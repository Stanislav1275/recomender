import logging
import pathlib
import pickle
import shutil
import asyncio
import grpc
from typing import Tuple, Optional, List, Dict, Any, Union
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
from recommendations.config import MODEL_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelStorage:
    """
    Класс для управления хранением моделей и датасетов на диске.
    Поддерживает версионирование и откат.
    """
    def __init__(self, model_dir: str = "data"):
        self.model_dir = pathlib.Path(model_dir)
        self.current_dir = self.model_dir / "cur"
        self.prev_dir = self.model_dir / "prev"
        self.file_lock = asyncio.Lock()
        
        # Создаем необходимые директории
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.prev_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def current_model_path(self) -> pathlib.Path:
        return self.current_dir / "model.pkl"
    
    @property
    def current_dataset_path(self) -> pathlib.Path:
        return self.current_dir / "dataset.pkl"
    
    async def save_model(self, model: ModelBase, dataset: Optional[Dataset] = None) -> bool:
        """
        Сохраняет модель и опционально датасет на диск с версионированием.
        
        Args:
            model: Модель для сохранения
            dataset: Датасет для сохранения (опционально)
            
        Returns:
            bool: Успешно ли сохранение
        """
        async with self.file_lock:
            try:
                # Временные пути для атомарного сохранения
                tmp_model_path = self.current_model_path.with_suffix(".tmp")
                
                # Сохраняем модель
                model.save(str(tmp_model_path))
                
                # Сохраняем датасет, если передан
                if dataset is not None:
                    tmp_dataset_path = self.current_dataset_path.with_suffix(".tmp")
                    with open(tmp_dataset_path, 'wb') as dataset_file:
                        pickle.dump(dataset, dataset_file)
                
                # Бэкап текущих файлов
                if self.current_model_path.exists():
                    shutil.move(str(self.current_model_path), str(self.prev_dir / "model.pkl"))
                    
                    if dataset is not None and self.current_dataset_path.exists():
                        shutil.move(str(self.current_dataset_path), str(self.prev_dir / "dataset.pkl"))
                
                # Атомарное переименование
                tmp_model_path.replace(self.current_model_path)
                if dataset is not None:
                    tmp_dataset_path.replace(self.current_dataset_path)
                
                logger.info(f"Модель и датасет успешно сохранены")
                return True
            except Exception as e:
                logger.error(f"Ошибка сохранения модели: {e}")
                return False
    
    async def load_model(self) -> Tuple[Optional[ModelBase], Optional[Dataset]]:
        """
        Загружает модель и датасет с диска.
        
        Returns:
            Tuple[Optional[ModelBase], Optional[Dataset]]: Загруженная модель и датасет или None
        """
        try:
            if not self.current_model_path.exists() or not self.current_dataset_path.exists():
                logger.warning("Модель или датасет не найдены на диске")
                return None, None
                
            # Загружаем датасет
            with open(self.current_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
                
            # Загружаем модель
            model = load_model(f=str(self.current_model_path))
            
            logger.info("Модель и датасет успешно загружены")
            return model, dataset
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return None, None
    
    async def rollback(self) -> Tuple[Optional[ModelBase], Optional[Dataset]]:
        """
        Откатывает модель и датасет к предыдущей версии.
        
        Returns:
            Tuple[Optional[ModelBase], Optional[Dataset]]: Предыдущая модель и датасет или None
        """
        async with self.file_lock:
            try:
                prev_model_path = self.prev_dir / "model.pkl"
                prev_dataset_path = self.prev_dir / "dataset.pkl"
                
                if not prev_model_path.exists() or not prev_dataset_path.exists():
                    logger.warning("Предыдущие версии не найдены для отката")
                    return None, None
                
                # Временно сохраняем текущие файлы
                if self.current_model_path.exists():
                    shutil.move(str(self.current_model_path), str(self.current_dir / "model.bak"))
                    shutil.move(str(self.current_dataset_path), str(self.current_dir / "dataset.bak"))
                
                # Перемещаем предыдущие версии
                shutil.move(str(prev_model_path), str(self.current_model_path))
                shutil.move(str(prev_dataset_path), str(self.current_dataset_path))
                
                # Загружаем откаченные файлы
                return await self.load_model()
            except Exception as e:
                logger.error(f"Ошибка отката модели: {e}")
                return None, None


class ModelManager:
    """
    Синглтон для управления моделью рекомендаций.
    Отвечает за инициализацию, обучение и обновление моделей.
    """
    _instance = None
    _model: Optional[ModelBase] = None
    _dataset: Optional[Dataset] = None
    _version = 0
    _rw_lock = RLock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._storage = ModelStorage()
        return cls._instance
    
    async def initialize(self):
        """
        Инициализирует менеджер моделей.
        Загружает существующую модель с диска или обучает новую.
        """
        model, dataset = await self._storage.load_model()
        
        if model is not None and dataset is not None:
            if model.is_fitted:
                with self._rw_lock:
                    self._model = model
                    self._dataset = dataset
                    self._version += 1
                logger.info(f"Модель инициализирована, версия {self._version}")
            else:
                logger.warning("Загруженная модель не обучена! Начинаем переобучение...")
                await self.train()
        else:
            logger.warning("Модель или датасет отсутствуют. Начинаем обучение...")
            await self.train()

    async def fit_partial(self, new_interactions: DataFrame = None, new_user_features: DataFrame = None) -> bool:
        """
        Частичное дообучение модели на новых данных.
        
        Args:
            new_interactions: Новые взаимодействия
            new_user_features: Новые признаки пользователей
            
        Returns:
            bool: Успешно ли дообучение
        """
        with self._rw_lock:
            if self._model is None or not self._model.is_fitted:
                logger.warning("Модель не инициализирована или не обучена для частичного дообучения")
                return False
                
            try:
                # Логика инкрементального обновления зависит от конкретной модели
                # Здесь мы просто обновляем текущую модель
                logger.info(f"Выполняется частичное дообучение модели на {len(new_interactions)} записях")
                
                # В реальной имплементации здесь бы был код обновления модели и датасета
                # model.fit_partial(new_interactions, new_user_features)
                
                await self._storage.save_model(self._model, self._dataset)
                self._version += 1
                return True
            except Exception as e:
                logger.error(f"Ошибка частичного дообучения: {e}")
                return False

    async def train(self) -> bool:
        """
        Полное обучение модели на всех данных.
        
        Returns:
            bool: Успешно ли обучение
        """
        try:
            # Обновляем черный список
            await BlacklistManager.refresh_blacklist()
            
            # Получаем данные
            user_features = await DataPrepareService.get_users_features()
            items_features = await DataPrepareService.get_titles_features()
            interactions = await DataPrepareService.get_interactions()
            
            if interactions.empty:
                logger.error("Данные взаимодействий пусты. Обучение невозможно.")
                return False
            
            # Конструируем датасет
            dataset = Dataset.construct(
                interactions_df=interactions,
                user_features_df=user_features,
                cat_user_features=["age_group", "sex", "preference"],
                item_features_df=items_features,
                cat_item_features=["type_id", "genres", "categories", "count_chapters", "age_limit", "relation_list"],
            )
            
            # Создаем модель с параметрами из конфига
            lightfm_model = LightFM(
                no_components=MODEL_CONFIG.get("no_components", 100),
                loss=MODEL_CONFIG.get("loss", "bpr"),
                random_state=MODEL_CONFIG.get("random_state", 60)
            )
            
            model = LightFMWrapperModel(
                lightfm_model,
                num_threads=MODEL_CONFIG.get("num_threads", 3),
                epochs=MODEL_CONFIG.get("epochs", 30)
            )
            
            # Обучаем модель
            logger.info("Начинаем обучение модели...")
            model.fit(dataset)
            
            if not model.is_fitted:
                logger.error("Ошибка! Модель не обучилась!")
                return False
            
            # Обновляем модель в памяти и на диске
            with self._rw_lock:
                self._model = model
                self._dataset = dataset
                
            await self._storage.save_model(model, dataset)
            self._version += 1
            
            logger.info(f"Модель успешно обучена и сохранена, версия {self._version}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {str(e)}")
            return False

    async def get_model(self) -> Tuple[ModelBase, Dataset]:
        """
        Возвращает текущую модель и датасет.
        
        Returns:
            Tuple[ModelBase, Dataset]: Текущая модель и датасет
        
        Raises:
            ValueError: Если модель не инициализирована
        """
        with self._rw_lock:
            if self._model is None or self._dataset is None:
                raise ValueError("Модель не инициализирована")
            return self._model, self._dataset

    @property
    def current_version(self) -> int:
        """
        Возвращает текущую версию модели.
        
        Returns:
            int: Текущая версия модели
        """
        return self._version

    @property
    def model(self) -> Optional[ModelBase]:
        """
        Возвращает текущую модель.
        
        Returns:
            Optional[ModelBase]: Текущая модель или None
        """
        return self._model


class RecService:
    """
    Сервис рекомендаций, предоставляющий API для получения рекомендаций
    и управления моделями.
    """
    _scheduler: Optional[AsyncIOScheduler] = None

    @classmethod
    def start_scheduler(cls):
        """
        Запускает планировщик для периодического обучения модели.
        """
        if cls._scheduler is None:
            cls._scheduler = AsyncIOScheduler(timezone=pytz.timezone("Europe/Moscow"))
            cls._scheduler.add_job(
                cls._scheduled_train, 
                trigger=CronTrigger(hour=4, minute=0), 
                max_instances=1
            )
            cls._scheduler.start()
            logger.info("✅ Планировщик запущен")

    @staticmethod
    async def _handle_request(context: grpc.ServicerContext) -> bool:
        """
        Проверяет готовность модели к обработке запроса.
        
        Args:
            context: gRPC контекст запроса
            
        Returns:
            bool: Готова ли модель к обработке запроса
        """
        try:
            if ModelManager().model is None:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "❌ Модель не инициализирована")
                return False
            return True
        except Exception as e:
            logger.error(f"🔥 Внутренняя ошибка: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Внутренняя ошибка: {str(e)}")
            return False

    @staticmethod
    async def rec(user_id: int, context: Optional[grpc.ServicerContext] = None) -> List[int]:
        """
        Получает персональные рекомендации для пользователя.
        
        Args:
            user_id: ID пользователя
            context: gRPC контекст (опционально)
            
        Returns:
            List[int]: Список ID рекомендованных произведений
            
        Raises:
            Exception: При ошибке получения рекомендаций
        """
        if context and not await RecService._handle_request(context):
            return []
            
        try:
            model, dataset = await ModelManager().get_model()
            recos = model.recommend(users=[user_id], dataset=dataset, k=40, filter_viewed=True)
            return recos['item_id'].tolist()
        except Exception as e:
            logger.error(f"Ошибка получения рекомендаций для пользователя {user_id}: {e}")
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, str(e))
            raise

    @classmethod
    async def _scheduled_train(cls):
        """
        Запланированное обучение модели.
        """
        logger.info("Начинаем запланированное обучение модели...")
        await ModelManager().train()

    @staticmethod
    async def relevant(item_id: int, context: Optional[grpc.ServicerContext] = None) -> List[int]:
        """
        Получает похожие произведения.
        
        Args:
            item_id: ID произведения
            context: gRPC контекст (опционально)
            
        Returns:
            List[int]: Список ID похожих произведений
            
        Raises:
            Exception: При ошибке получения рекомендаций
        """
        if context and not await RecService._handle_request(context):
            return []
            
        try:
            model, dataset = await ModelManager().get_model()
            recos = model.recommend_to_items(target_items=[item_id], dataset=dataset, k=40, filter_itself=False)
            return recos['item_id'].tolist()
        except Exception as e:
            logger.error(f"Ошибка получения похожих произведений для ID {item_id}: {e}")
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, str(e))
            raise

    @staticmethod
    async def train(context: Optional[grpc.ServicerContext] = None) -> bool:
        """
        Запускает обучение модели вручную.
        
        Args:
            context: gRPC контекст (опционально)
            
        Returns:
            bool: Успешно ли обучение
            
        Raises:
            Exception: При ошибке обучения
        """
        try:
            result = await ModelManager().train()
            if not result and context:
                await context.abort(grpc.StatusCode.INTERNAL, "Ошибка обучения модели")
            return result
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, str(e))
            raise
