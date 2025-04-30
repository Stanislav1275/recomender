import logging
import pathlib
import pickle
import shutil
import asyncio
import grpc
from typing import Tuple, Optional
from threading import RLock

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

import warnings
from lightfm import LightFM
from pandas import DataFrame
from rectools import Columns
from rectools.dataset import Dataset, Interactions
from rectools.model_selection import TimeRangeSplitter
from rectools.models import LightFMWrapperModel, load_model
from rectools.models.base import ModelBase

from recommendations.data_preparer import DataPrepareService, BlacklistManager

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
        return cls._instance

    async def initialize(self):
        model_path, dataset_path = self._current_paths
        if model_path.exists() and dataset_path.exists():
            try:
                self._load_from_disk(model_path, dataset_path)
                if not self._model.is_fitted:
                    logger.warning("Загруженная модель не обучена! Начинаем переобучение...")
                    await self.train()
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}. Запускаем переобучение...")
                await self.train()
        else:
            logger.warning("Модель или датасет отсутствуют. Начинаем обучение...")
            await self.train()

    #         title_id:int
    # typer-imp_read_chapter, imp_view_chapter, exp_rating, imp_view, exp_custom
    #  exp_weight?: int
    # dasdsadasdsad
    async def fit_partial(self, new_interactions: DataFrame = None, new_user_features: DataFrame = None):
        model, currentDataset = await self.get_model()
        if not model.is_fitted:
            print("model is not fitted to partial fit")
            return

        await self.update_model(model)

    async def train(self):
        try:
            await BlacklistManager.refresh_blacklist()

            user_features = await DataPrepareService.get_users_features()
            items_features = await DataPrepareService.get_titles_features()
            interactions = await DataPrepareService.get_interactions()
            if interactions.empty:
                raise ValueError("Данные взаимодействий пусты. Обучение невозможно.")

            dataset = Dataset.construct(
                interactions_df=interactions,

                user_features_df=user_features,
                cat_user_features=["age_group", "sex", "preference"],
                item_features_df=items_features,
                cat_item_features=["type_id", "genres", "categories", "count_chapters", "age_limit", "relation_list"],
            )

            model = LightFMWrapperModel(LightFM(no_components=100, loss="bpr", random_state=60), num_threads=3,
                                        epochs=30)

            logger.info("Начинаем обучение модели...")
            model.fit(dataset)
            if not model.is_fitted:
                raise ValueError("Ошибка! Модель не обучилась!")

            await self.update_model(model, dataset)
            logger.info("Модель успешно обучена и сохранена.")

        except Exception as e:
            logger.error(f"Ошибка обучения модели: {str(e)}")
            raise

    @property
    def _current_paths(self) -> Tuple[pathlib.Path, pathlib.Path]:
        return pathlib.Path("data/cur/model.pkl"), pathlib.Path("data/cur/dataset.pkl")

    def _load_from_disk(self, model_path: pathlib.Path, dataset_path: pathlib.Path):
        with self._rw_lock:
            try:
                with open(dataset_path, 'rb') as f:
                    self._dataset = pickle.load(f)
                self._model = load_model(f=model_path)
                if not self._model.is_fitted:
                    raise ValueError("Загруженная модель не обучена.")
                self._version += 1
                logger.info(f"Загружена модель версии {self._version}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise

    async def update_model(self, model: ModelBase, dataset: Dataset = None):
        async with self._file_lock:
            with self._rw_lock:
                model_path, dataset_path = self._current_paths
                prev_dir = pathlib.Path("data/prev")
                cur_dir = pathlib.Path("data/cur")
                prev_dir.mkdir(parents=True, exist_ok=True)
                cur_dir.mkdir(parents=True, exist_ok=True)
                try:
                    logger.info("Сохраняем новую модель...")
                    # Сохраняем модель
                    tmp_model_path = model_path.with_suffix(".tmp")
                    model.save(str(tmp_model_path))

                    # Сохраняем датасет, только если он передан
                    if dataset is not None:
                        tmp_dataset_path = dataset_path.with_suffix(".tmp")
                        with open(tmp_dataset_path, 'wb') as dataset_file:
                            pickle.dump(dataset, dataset_file)

                    # Перемещаем старые файлы в папку prev (если они существуют)
                    if model_path.exists():
                        shutil.move(str(model_path), str(prev_dir / "model.pkl"))
                        # Перемещаем датасет только если он обновляется
                        if dataset is not None and dataset_path.exists():
                            shutil.move(str(dataset_path), str(prev_dir / "dataset.pkl"))

                    # Заменяем временные файлы на актуальные
                    tmp_model_path.replace(model_path)
                    if dataset is not None:
                        tmp_dataset_path.replace(dataset_path)

                    # Обновляем модель и датасет (если передан)
                    self._model = model
                    if dataset is not None:
                        self._dataset = dataset
                    self._version += 1
                    logger.info(f"Модель обновлена до версии {self._version}")
                except Exception as e:
                    logger.error(f"Ошибка обновления модели: {e}")
                    raise

    async def get_model(self) -> Tuple[ModelBase, Dataset]:
        if self._model is None or self._dataset is None:
            raise ValueError("Модель не инициализирована")
        return self._model, self._dataset

    @property
    def current_version(self) -> int:
        return self._version

    @property
    def model(self):
        return self._model


class RecService:
    _scheduler: Optional[AsyncIOScheduler] = None

    @classmethod
    def start_scheduler(cls):
        if cls._scheduler is None:
            cls._scheduler = AsyncIOScheduler(timezone=pytz.timezone("Europe/Moscow"))
            cls._scheduler.add_job(cls._scheduled_train, trigger=CronTrigger(hour=4, minute=0), max_instances=1)
            cls._scheduler.start()
            logger.info("✅ Scheduler started")

    @staticmethod
    async def _handle_request(context: grpc.ServicerContext):
        try:
            if ModelManager().model is None:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "❌ Model not initialized")
        except Exception as e:
            logger.error(f"🔥 Internal error: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    @staticmethod
    async def rec(user_id: int, context: grpc.ServicerContext):
        await RecService._handle_request(context)

        try:
            model, dataset = await ModelManager().get_model()
            logger.debug(f"Using model: {model}")

            recos = model.recommend(users=[user_id], dataset=dataset, k=40, filter_viewed=True)
            return recos['item_id'].tolist()
        except Exception as e:
            logger.error(f"🚨 Recommendation error: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Recommendation error: {str(e)}")

    @classmethod
    async def _scheduled_train(cls):
        try:
            logger.info("⏳ Starting scheduled training...")
            await cls.train()
            logger.info("✅ Scheduled training completed successfully")
        except Exception as e:
            logger.error(f"🚨 Scheduled training failed: {str(e)}")

    @staticmethod
    async def relevant(item_id: int, context: grpc.ServicerContext):
        await RecService._handle_request(context)
        try:
            model, dataset = await ModelManager().get_model()
            logger.debug(f"Using model: {model}")

            recos = model.recommend_to_items(
                target_items=[item_id],
                dataset=dataset,
                k=40,
                filter_itself=True,
            )
            return recos['item_id'].tolist()
        except Exception as e:
            logger.error(f"🚨 Recommendation error: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Recommendation error: {str(e)}")

    @staticmethod
    async def train(context: Optional[grpc.ServicerContext] = None):
        try:
            await BlacklistManager.refresh_blacklist()
            user_features = await DataPrepareService.get_users_features()
            items_features = await DataPrepareService.get_titles_features()
            interactions = await DataPrepareService.get_interactions()
            print(items_features.head())
            print(interactions.head(50))

            model = LightFMWrapperModel(LightFM(no_components=100, loss="bpr", random_state=60), num_threads=3,
                                        epochs=30)
            dataset = Dataset.construct(

                interactions_df=interactions,
                user_features_df=user_features,
                cat_user_features=["age_group", "sex", "preference"],
                item_features_df=items_features,
                cat_item_features=["type_id", "genres", "categories", "count_chapters", "age_limit", "relation_list"],
            )

            model.fit(dataset)

            if not model.is_fitted:
                logger.error("❌ Model is not fitted before saving! Something went wrong.")
                raise ValueError("Model is not fitted.")

            await ModelManager().update_model(model, dataset)
            logger.info("✅ Model trained and updated successfully")

            if context:
                return {"status": "success", "version": ModelManager().current_version}
        except Exception as e:
            logger.error(f"🔥 Training failed: {str(e)}")
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, f"Training failed: {str(e)}")
            else:
                raise
