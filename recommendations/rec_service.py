import logging
import pathlib
import pickle
import shutil
import asyncio
import grpc
from typing import Tuple, Optional, Dict, List, Any
from threading import RLock
import datetime
import time
import copy
import numpy as np

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
from recommendations.model_registry import ModelRegistry
from recommendations.cache_service import RecommendationCache
from recommendations.auth import auth_required

# Импортируем настройки конфигурации
from recommendations.config import Config

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

    def __init__(self, dp_service: DataPrepareService, model_config=None) -> None:
        self.dp_service = dp_service
        self.logger = logging.getLogger(__name__)
        self.user_embeddings_ = None
        self.item_embeddings_ = None
        if model_config is None:
            config_dict = Config.MODEL_PARAMS['default']
        else:
            config_dict = model_config
        self.config = config_dict
        self.num_threads = config_dict.get("num_threads", 4)  # Используем параметр из конфига или значение по умолчанию

    async def initialize(self):
        """Инициализирует менеджер моделей, загружая активную модель из реестра"""
        registry = ModelRegistry()
        # Получаем информацию об активной модели
        active_model = registry.get_active_model()
        
        if active_model and active_model.get('file_path') and active_model.get('dataset_path'):
            try:
                # Загружаем модель и датасет
                self._load_from_registry(active_model)
                
                if not self._model.is_fitted:
                    logger.warning("Загруженная модель не обучена! Начинаем переобучение...")
                    await self.train()
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}. Запускаем переобучение...")
                await self.train()
        else:
            logger.warning("Активная модель не найдена в реестре. Начинаем обучение...")
            await self.train()
        
        # Инициализируем сервис кеширования
        cache = RecommendationCache()
        try:
            await cache.initialize()
            logger.info("Сервис кеширования успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации сервиса кеширования: {e}")

    def _load_from_registry(self, model_info: Dict[str, Any]):
        """Загружает модель и датасет из реестра моделей"""
        with self._rw_lock:
            try:
                model_path = model_info.get('file_path')
                dataset_path = model_info.get('dataset_path')
                
                with open(dataset_path, 'rb') as f:
                    self._dataset = pickle.load(f)
                self._model = load_model(f=model_path)
                
                if not self._model.is_fitted:
                    raise ValueError("Загруженная модель не обучена.")
                    
                self._version = model_info.get('version', 0)
                logger.info(f"Загружена модель версии {self._version}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise
    
    @auth_required("fit_partial")
    async def fit_partial(self, new_interactions: DataFrame = None, new_user_features: DataFrame = None, context = None):
        """
        Частично дообучает модель на новых данных, не переобучая на всем датасете.
        
        Args:
            new_interactions: DataFrame с новыми взаимодействиями пользователей
            new_user_features: DataFrame с новыми характеристиками пользователей
            context: Контекст gRPC запроса
        """
        model, current_dataset = await self.get_model()
        if not model.is_fitted:
            logger.warning("Модель не обучена, невозможно выполнить частичное дообучение.")
            return False
        
        try:
            logger.info("Начинаем частичное дообучение модели...")
            
            # Проверяем, что данные не пустые
            if new_interactions is None or new_interactions.empty:
                logger.warning("Нет новых взаимодействий для дообучения модели")
                return False
            
            # Добавляем новые данные в датасет
            updated_dataset = current_dataset.clone()
            
            # Добавляем новые взаимодействия
            if new_interactions is not None and not new_interactions.empty:
                interactions_to_add = Interactions(new_interactions)
                updated_dataset.update_interactions(interactions_to_add)
            
            # Добавляем новые характеристики пользователей, если есть
            if new_user_features is not None and not new_user_features.empty:
                updated_dataset.update_user_features(new_user_features)
            
            # Частичное дообучение
            model.fit_partial(updated_dataset, epochs=3)
            
            # Сохраняем обновленную модель
            model_id = await self.register_model(model, updated_dataset, {
                'model_type': 'lightfm',
                'training_type': 'partial',
                'epochs': 3,
                'interactions_count': len(new_interactions)
            })
            
            # Инвалидируем кеш
            cache = RecommendationCache()
            await cache.invalidate_all_recommendations()
            
            logger.info(f"Модель успешно дообучена и зарегистрирована с ID: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка частичного дообучения модели: {str(e)}")
            raise

    @auth_required("train")
    async def train(self, parameters: Optional[Dict[str, Any]] = None, context = None):
        """
        Обучает новую модель с указанными параметрами.
        
        Args:
            parameters: Параметры модели (если None, используются параметры по умолчанию)
            context: Контекст gRPC запроса
        
        Returns:
            ID новой модели в реестре
        """
        try:
            # Устанавливаем параметры по умолчанию
            params = {
                'model_type': 'lightfm',
                'no_components': 100,
                'loss': 'bpr',
                'random_state': 60,
                'num_threads': self.num_threads,
                'epochs': self.config["epochs"],
                'item_alpha': 0.0,
                'user_alpha': 0.0
            }
            
            # Обновляем параметры, если они переданы
            if parameters:
                params.update(parameters)
            
            # Получаем данные
            await BlacklistManager.refresh_blacklist()
            user_features = await self.dp_service.get_users_features()
            items_features = await self.dp_service.get_titles_features()
            interactions = await self.dp_service.get_interactions()
            
            if interactions.empty:
                raise ValueError("Данные взаимодействий пусты. Обучение невозможно.")

            # Создаем датасет
            dataset = Dataset.construct(
                interactions_df=interactions,
                user_features_df=user_features,
                cat_user_features=["age_group", "sex", "preference"],
                item_features_df=items_features,
                cat_item_features=["type_id", "genres", "categories", "count_chapters", "age_limit", "relation_list"],
            )

            # Создаем и обучаем модель
            model = LightFMWrapperModel(
                LightFM(
                    no_components=params['no_components'],
                    loss=params['loss'],
                    random_state=params['random_state'],
                    item_alpha=params['item_alpha'],
                    user_alpha=params['user_alpha']
                ),
                num_threads=params['num_threads'],
                epochs=params['epochs']
            )

            logger.info("Начинаем обучение модели...")
            start_time = time.time()
            model.fit(dataset)
            
            if not model.is_fitted:
                raise ValueError("Ошибка! Модель не обучилась!")

            # Регистрируем модель в реестре
            model_id = await self.register_model(model, dataset, params)
            
            # Инвалидируем кеш рекомендаций
            cache = RecommendationCache()
            await cache.invalidate_all_recommendations()
            
            logger.info(f"Модель успешно обучена и зарегистрирована с ID: {model_id}")
            
            # Сохраняем эмбеддинги для быстрого доступа
            self.user_embeddings_ = model.get_user_embeddings()
            self.item_embeddings_ = model.get_item_embeddings()
            
            return model_id

        except Exception as e:
            logger.error(f"Ошибка обучения модели: {str(e)}")
            raise

    async def register_model(self, model: ModelBase, dataset: Dataset, parameters: Dict[str, Any]) -> str:
        """
        Регистрирует новую модель в реестре и устанавливает ее как активную.
        
        Args:
            model: Объект модели
            dataset: Объект датасета
            parameters: Параметры модели
            
        Returns:
            ID новой модели в реестре
        """
        registry = ModelRegistry()
        
        # Регистрируем модель в реестре
        model_id = registry.register_model(
            name="lightfm_model",
            parameters=parameters
        )
        
        # Сохраняем файлы модели
        model_path, dataset_path = registry.save_model_files(model_id, model, dataset)
        
        # Устанавливаем модель как активную
        registry.set_active_model(model_id)
        
        # Обновляем текущую модель в памяти
        with self._rw_lock:
            self._model = model
            self._dataset = dataset
            self._version = parameters.get('version', self._version + 1)
        
        return model_id

    async def get_model(self) -> Tuple[ModelBase, Dataset]:
        """
        Получает текущую активную модель и датасет.
        
        Returns:
            Кортеж с моделью и датасетом
        """
        if self._model is None or self._dataset is None:
            raise ValueError("Модель не инициализирована")
        return self._model, self._dataset
    
    @auth_required("list_models")
    async def list_models(self, limit: int = 20, offset: int = 0, context = None) -> List[Dict[str, Any]]:
        """
        Получает список моделей из реестра.
        
        Args:
            limit: Максимальное количество моделей
            offset: Смещение для пагинации
            context: Контекст gRPC запроса
            
        Returns:
            Список словарей с информацией о моделях
        """
        registry = ModelRegistry()
        return registry.list_models(limit=limit, offset=offset)
    
    @auth_required("get_model_info")
    async def get_model_info(self, model_id: str, context = None) -> Dict[str, Any]:
        """
        Получает информацию о модели из реестра.
        
        Args:
            model_id: ID модели
            context: Контекст gRPC запроса
            
        Returns:
            Словарь с информацией о модели
        """
        registry = ModelRegistry()
        return registry.get_model_info(model_id)
    
    @auth_required("set_active_model")
    async def set_active_model(self, model_id: str, context = None) -> bool:
        """
        Устанавливает модель как активную.
        
        Args:
            model_id: ID модели
            context: Контекст gRPC запроса
            
        Returns:
            True, если модель успешно установлена как активная
        """
        registry = ModelRegistry()
        model_info = registry.get_model_info(model_id)
        
        if not model_info:
            return False
            
        # Устанавливаем модель как активную в реестре
        if not registry.set_active_model(model_id):
            return False
            
        # Загружаем модель в память
        try:
            self._load_from_registry(model_info)
            
            # Инвалидируем кеш рекомендаций
            cache = RecommendationCache()
            await cache.invalidate_all_recommendations()
            
            return True
        except Exception as e:
            logger.error(f"Ошибка установки активной модели: {e}")
            return False
    
    @auth_required("schedule_training")
    async def schedule_training(self, parameters: Optional[Dict[str, Any]] = None, scheduled_at: Optional[datetime.datetime] = None, context = None) -> str:
        """
        Планирует обучение модели.
        
        Args:
            parameters: Параметры модели
            scheduled_at: Время запланированного обучения (если None, используется текущее время)
            context: Контекст gRPC запроса
            
        Returns:
            ID задания на обучение
        """
        registry = ModelRegistry()
        
        if scheduled_at is None:
            scheduled_at = datetime.datetime.now()
            
        # Планируем задание на обучение
        job_id = registry.schedule_training_job(
            parameters=parameters or {},
            scheduled_at=scheduled_at
        )
        
        return job_id
    
    @property
    def current_version(self) -> int:
        return self._version

    def _get_model(self):
        model_config = model_config_class(**self.config["model"])
        model = model_cls(config=model_config)
        return model

    def train(self, train_user_idx, train_item_idx, train_weights):
        model = self._get_model()
        self.logger.info('Training model...')
        start_time = time.time()
        
        # Используем параметр threads из конфига
        model.fit(
            interactions=(train_user_idx, train_item_idx, train_weights),
            num_threads=self.num_threads,
            epochs=self.config["epochs"]
        )
        
        self.logger.info(f'Model training completed in {time.time() - start_time:.2f} seconds')
        self.model = model
        
        # Сохраняем эмбеддинги для быстрого доступа
        self.user_embeddings_ = model.get_user_embeddings()
        self.item_embeddings_ = model.get_item_embeddings()
        
        return model

    def train_with_validation(self, train_user_idx, train_item_idx, train_weights, 
                            val_user_idx, val_item_idx, val_weights, k=10):
        model = self._get_model()
        self.logger.info('Training model with validation...')
        
        best_ndcg = 0
        best_epoch = 0
        best_model = None
        
        epochs = self.config["epochs"]
        # Обучаем модель с ранней остановкой
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Обучаем одну эпоху
            model.fit(
                interactions=(train_user_idx, train_item_idx, train_weights),
                num_threads=self.num_threads,
                epochs=1
            )
            
            # Оцениваем на валидационных данных
            val_ndcg = self._evaluate_ndcg(model, val_user_idx, val_item_idx, val_weights, k=k)
            
            epoch_time = time.time() - start_time
            self.logger.info(f'Epoch {epoch}/{epochs}, NDCG@{k}: {val_ndcg:.4f}, Time: {epoch_time:.2f}s')
            
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                self.logger.info(f'New best model at epoch {epoch} with NDCG@{k}: {val_ndcg:.4f}')
            
            # Ранняя остановка, если нет улучшения в течение 3 эпох
            if epoch - best_epoch >= 3:
                self.logger.info(f'Early stopping at epoch {epoch}, best epoch: {best_epoch}')
                break
        
        self.model = best_model if best_model else model
        
        # Сохраняем эмбеддинги для быстрого доступа
        self.user_embeddings_ = self.model.get_user_embeddings()
        self.item_embeddings_ = self.model.get_item_embeddings()
        
        return self.model

    def _evaluate_ndcg(self, model, user_idx, item_idx, weights, k=10):
        """Оценивает NDCG@k на валидационном наборе данных"""
        from rectools.metrics import calc_ndcg
        
        # Получаем уникальных пользователей
        unique_users = np.unique(user_idx)
        
        # Собираем истинные релевантности
        true_relevance = {}
        for u, i, w in zip(user_idx, item_idx, weights):
            if u not in true_relevance:
                true_relevance[u] = {}
            true_relevance[u][i] = w
        
        # Делаем предсказания
        all_items = np.unique(item_idx)
        predictions = []
        
        for user in unique_users:
            user_preds = model.predict(user, all_items, num_threads=self.num_threads)
            top_items_idx = np.argsort(-user_preds)[:k]
            top_items = all_items[top_items_idx]
            
            user_relevance = [true_relevance.get(user, {}).get(item, 0) for item in top_items]
            predictions.append((user, top_items, user_relevance))
        
        # Вычисляем NDCG
        ndcg_sum = 0
        for user, items, relevance in predictions:
            ndcg_sum += calc_ndcg(relevance, k)
        
        return ndcg_sum / len(unique_users)

    def predict(self, users, items=None, k=None, filtered_items=None):
        if not hasattr(self, 'model'):
            raise ValueError("Model is not trained yet.")
        
        # Делаем предсказания для всех комбинаций пользователей и элементов
        if items is None:
            if filtered_items is not None:
                items = filtered_items
            else:
                items = np.arange(self.model.get_item_embeddings().shape[0])
        
        # Используем параллельные вычисления для предсказаний
        scores = self.model.predict(users, items, num_threads=self.num_threads)
        
        if k is not None:
            # Для каждого пользователя находим топ-k элементов
            if len(users) == 1:
                # Если один пользователь, просто возвращаем топ-k элементов
                top_indices = np.argsort(-scores)[:k]
                return items[top_indices], scores[top_indices]
            else:
                # Если несколько пользователей, находим топ-k для каждого
                result_items = []
                result_scores = []
                
                for i, user in enumerate(users):
                    user_scores = scores[i]
                    top_indices = np.argsort(-user_scores)[:k]
                    result_items.append(items[top_indices])
                    result_scores.append(user_scores[top_indices])
                
                return result_items, result_scores
        
        return items, scores

    def get_similar_items(self, item_ids, k=10):
        """
        Возвращает k наиболее похожих элементов для каждого элемента в item_ids
        на основе сходства векторов в пространстве признаков
        """
        if self.item_embeddings_ is None:
            self.item_embeddings_ = self.model.get_item_embeddings()
        
        n_items = self.item_embeddings_.shape[0]
        
        result = []
        # Используем эффективное векторное вычисление косинусного сходства
        for item_id in item_ids:
            if item_id >= n_items:
                self.logger.warning(f"Item ID {item_id} is out of range. Skipping.")
                result.append(([], []))
                continue
            
            item_vector = self.item_embeddings_[item_id]
            
            # Вычисляем косинусное сходство со всеми элементами
            dot_products = np.dot(self.item_embeddings_, item_vector)
            item_norm = np.linalg.norm(item_vector)
            all_norms = np.linalg.norm(self.item_embeddings_, axis=1)
            
            # Избегаем деления на ноль
            similarities = np.zeros_like(dot_products)
            nonzero_indices = all_norms > 0
            similarities[nonzero_indices] = dot_products[nonzero_indices] / (all_norms[nonzero_indices] * item_norm)
            
            # Исключаем сам элемент
            similarities[item_id] = -1
            
            # Получаем топ-k наиболее похожих элементов
            top_indices = np.argsort(-similarities)[:k]
            top_similarities = similarities[top_indices]
            
            result.append((top_indices, top_similarities))
        
        return result


class RecService:
    _scheduler: Optional[AsyncIOScheduler] = None

    @classmethod
    def start_scheduler(cls):
        """Запускает планировщик заданий"""
        if cls._scheduler is None:
            cls._scheduler = AsyncIOScheduler(timezone=pytz.timezone("Europe/Moscow"))
            
            # Планируем ежедневное обучение модели в 3:00
            cls._scheduler.add_job(
                cls._scheduled_train,
                trigger=CronTrigger(hour=3, minute=0),
                max_instances=1
            )
            
            # Добавляем задание для проверки запланированных заданий каждые 5 минут
            cls._scheduler.add_job(
                cls._check_scheduled_jobs,
                trigger=CronTrigger(minute='*/5'),
                max_instances=1
            )
            
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
    @auth_required("rec")
    async def rec(user_id: int, context: grpc.ServicerContext = None):
        """
        Получает рекомендации для пользователя
        
        Args:
            user_id: ID пользователя
            context: Контекст gRPC (опционально)
            
        Returns:
            Список ID рекомендованных элементов
        """
        if context:
            await RecService._handle_request(context)
            
        try:
            # Сначала пытаемся получить рекомендации из кеша
            cache = RecommendationCache()
            cached_recommendations = await cache.get_user_recommendations(user_id)
            
            if cached_recommendations:
                logger.debug(f"Найдены кешированные рекомендации для пользователя {user_id}")
                return cached_recommendations
                
            # Если кеш пуст, вычисляем рекомендации
            logger.debug(f"Кеш пуст для пользователя {user_id}, вычисляем рекомендации")
            model, dataset = await ModelManager().get_model()
            
            recos = model.recommend(users=[user_id], dataset=dataset, k=40, filter_viewed=True)
            recommendations = recos['item_id'].tolist()
            
            # Сохраняем рекомендации в кеш
            await cache.cache_user_recommendations(user_id, recommendations)
            
            return recommendations
            
        except Exception as e:
            error_msg = f"🚨 Recommendation error: {str(e)}"
            logger.error(error_msg)
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, error_msg)
            raise

    @classmethod
    async def _scheduled_train(cls):
        """Выполняет запланированное обучение модели"""
        try:
            logger.info("⏳ Starting scheduled training...")
            await ModelManager().train()
            logger.info("✅ Scheduled training completed successfully")
        except Exception as e:
            logger.error(f"🚨 Scheduled training failed: {str(e)}")
    
    @classmethod
    async def _check_scheduled_jobs(cls):
        """Проверяет и выполняет запланированные задания на обучение"""
        try:
            registry = ModelRegistry()
            pending_jobs = registry.get_pending_jobs()
            
            for job in pending_jobs:
                job_id = job['id']
                parameters = job['parameters']
                
                try:
                    # Обновляем статус задания
                    registry.update_training_job(
                        job_id=job_id,
                        status='running',
                        started_at=datetime.datetime.now()
                    )
                    
                    # Выполняем обучение
                    model_id = await ModelManager().train(parameters)
                    
                    # Обновляем статус задания
                    registry.update_training_job(
                        job_id=job_id,
                        status='completed',
                        model_id=model_id,
                        completed_at=datetime.datetime.now()
                    )
                    
                    logger.info(f"✅ Scheduled job {job_id} completed successfully, model ID: {model_id}")
                    
                except Exception as e:
                    # В случае ошибки обновляем статус
                    registry.update_training_job(
                        job_id=job_id,
                        status='failed',
                        completed_at=datetime.datetime.now()
                    )
                    
                    logger.error(f"🚨 Scheduled job {job_id} failed: {str(e)}")
                    
        except Exception as e:
            logger.error(f"🚨 Error checking scheduled jobs: {str(e)}")

    @staticmethod
    @auth_required("relevant")
    async def relevant(item_id: int, context: grpc.ServicerContext = None):
        """
        Получает список похожих элементов
        
        Args:
            item_id: ID элемента
            context: Контекст gRPC (опционально)
            
        Returns:
            Список ID похожих элементов
        """
        if context:
            await RecService._handle_request(context)
            
        try:
            # Сначала пытаемся получить рекомендации из кеша
            cache = RecommendationCache()
            cached_recommendations = await cache.get_item_recommendations(item_id)
            
            if cached_recommendations:
                logger.debug(f"Найдены кешированные рекомендации для элемента {item_id}")
                return cached_recommendations
                
            # Если кеш пуст, вычисляем рекомендации
            logger.debug(f"Кеш пуст для элемента {item_id}, вычисляем рекомендации")
            model, dataset = await ModelManager().get_model()
            
            recos = model.recommend_to_items(
                target_items=[item_id],
                dataset=dataset,
                k=40,
                filter_itself=True,
            )
            recommendations = recos['item_id'].tolist()
            
            # Сохраняем рекомендации в кеш
            await cache.cache_item_recommendations(item_id, recommendations)
            
            return recommendations
            
        except Exception as e:
            error_msg = f"🚨 Recommendation error: {str(e)}"
            logger.error(error_msg)
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, error_msg)
            raise

    @staticmethod
    @auth_required("train")
    async def train(context: Optional[grpc.ServicerContext] = None):
        """
        Запускает обучение модели
        
        Args:
            context: Контекст gRPC (опционально)
            
        Returns:
            Словарь с информацией о результате обучения
        """
        try:
            model_id = await ModelManager().train()
            
            if context:
                return {"status": "success", "model_id": model_id, "version": ModelManager().current_version}
            return model_id
            
        except Exception as e:
            error_msg = f"🔥 Training failed: {str(e)}"
            logger.error(error_msg)
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, error_msg)
            raise
            
    @staticmethod
    @auth_required("get_user_recent_interactions")
    async def get_user_recent_interactions(user_id: int, limit: int = 10, context: grpc.ServicerContext = None):
        """
        Получает последние взаимодействия пользователя
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество взаимодействий
            context: Контекст gRPC (опционально)
            
        Returns:
            DataFrame с последними взаимодействиями пользователя
        """
        try:
            dp_service = DataPrepareService(session_maker=None)  # Получаем из DI
            interactions = await dp_service.get_user_recent_interactions(user_id, limit)
            return interactions
            
        except Exception as e:
            error_msg = f"🚨 Error getting user interactions: {str(e)}"
            logger.error(error_msg)
            if context:
                await context.abort(grpc.StatusCode.INTERNAL, error_msg)
            raise
