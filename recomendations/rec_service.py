import logging
import pathlib
import pickle
import shutil
import tempfile
import asyncio
import grpc
from typing import Tuple, Optional, Any
from threading import RLock
from contextlib import contextmanager

import warnings
from lightfm import LightFM
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel, load_model
from rectools.models.base import ModelBase
from rectools.models.lightfm import LightFMWrapperModelConfig

from grpcs.services.data_preparer import DataPrepareService

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None
    _model: Optional[ModelBase] = None
    _dataset: Optional[Dataset] = None
    _version = 0
    _rw_lock = RLock()
    _file_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        model_path, dataset_path = self._current_paths
        if model_path.exists() and dataset_path.exists():
            self._load_from_disk(model_path, dataset_path)

    @property
    def _current_paths(self) -> Tuple[pathlib.Path, pathlib.Path]:
        return (
            pathlib.Path("data/cur/model.csv"),
            pathlib.Path("data/cur/dataset.pkl")
        )

    def _load_from_disk(self, model_path: pathlib.Path, dataset_path: pathlib.Path):
        with self._rw_lock:
            try:
                with open(dataset_path, 'rb') as f:
                    self._dataset = pickle.load(f)
                self._model = load_model(str(model_path))
                self._version += 1
                logger.info(f"Loaded model version {self._version}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise

    @contextmanager
    def _atomic_writer(self, path: pathlib.Path):
        temp = path.with_suffix(".tmp")
        try:
            yield temp
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()

    async def update_model(self, model: ModelBase, dataset: Dataset):
        async with self._file_lock:
            with self._rw_lock:
                model_path, dataset_path = self._current_paths
                prev_dir = pathlib.Path("data/prev")
                prev_dir.mkdir(exist_ok=True, parents=True)

                with self._atomic_writer(model_path) as tmp_model, \
                        self._atomic_writer(dataset_path) as tmp_dataset:

                    model.save(str(tmp_model))
                    with open(tmp_dataset, 'wb') as f:
                        pickle.dump(dataset, f)

                self._model = model
                self._dataset = dataset
                self._version += 1
                logger.info(f"Updated model to version {self._version}")

                if model_path.exists():
                    shutil.move(str(model_path), str(prev_dir / "model.csv"))
                if dataset_path.exists():
                    shutil.move(str(dataset_path), str(prev_dir / "dataset.pkl"))

    async def get_model(self) -> Tuple[ModelBase, Dataset]:
        if self._model is None or self._dataset is None:
            raise ValueError("Model not initialized")
        return self._model, self._dataset

    @property
    def current_version(self) -> int:
        return self._version


class RecService:
    @staticmethod
    async def _handle_request(context: grpc.ServicerContext):
        try:
            if ModelManager()._model is None:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    @staticmethod
    async def relevant_2_item(item_id: int, context: grpc.ServicerContext):
        await RecService._handle_request(context)
        try:
            model, dataset = await ModelManager().get_model()
            recos = model.recommend_to_items(
                target_items=[item_id],
                dataset=dataset,
                k=40,
                filter_itself=True,
            )
            return recos['item_id'].tolist()
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, f"Recommendation error: {str(e)}")

    @staticmethod
    async def rec(user_id: int, context: grpc.ServicerContext):
        await RecService._handle_request(context)
        try:
            model, dataset = await ModelManager().get_model()
            recos = model.recommend(
                users=[user_id],
                dataset=dataset,
                k=40,
                filter_viewed=True,
            )
            return recos['item_id'].tolist()
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, f"Recommendation error: {str(e)}")

    @staticmethod
    async def train(context: grpc.ServicerContext):
        try:
            user_features = await DataPrepareService.get_users_features()
            items_features = await DataPrepareService.get_titles_features()
            interactions = await DataPrepareService.get_interactions()

            model = LightFMWrapperModel(LightFM(
                no_components=10,
                loss="bpr",
                random_state=60
            ))

            dataset = Dataset.construct(
                interactions_df=interactions,
                user_features_df=user_features,
                cat_user_features=["age_group", "sex", "preference"],
                item_features_df=items_features,
                cat_item_features=["type_id", "genres", "categories", "count_chapters", "age_limit"],
            )

            model.fit(dataset)
            await ModelManager().update_model(model, dataset)

            return {"status": "success", "version": ModelManager().current_version}
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Training failed: {str(e)}")
