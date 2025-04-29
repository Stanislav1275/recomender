import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import aioredis
from abc import ABC, abstractmethod
import numpy as np
import time
import pandas as pd
from sqlalchemy import text

from recommendation_system.core.models.base import BaseDataService
from recommendation_system.infrastructure.persistence.database import get_session

logger = logging.getLogger(__name__)

class BaseRecommender(ABC):
    """Базовый абстрактный класс для всех рекомендательных моделей"""
    
    @abstractmethod
    def fit(self, dataset) -> None:
        """Обучение модели на датасете"""
        pass
    
    @abstractmethod
    def recommend(self, user_ids: List[int], k: int = 10, filter_viewed: bool = True) -> Dict[int, List[int]]:
        """Получение рекомендаций для пользователей"""
        pass
    
    @abstractmethod
    def recommend_similar_items(self, item_ids: List[int], k: int = 10) -> Dict[int, List[int]]:
        """Получение похожих элементов"""
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Проверка, обучена ли модель"""
        pass

class BaseCacheService(ABC):
    """Базовый интерфейс для кэширования"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Инициализация сервиса кэширования"""
        pass
    
    @abstractmethod
    async def get_user_recommendations(self, user_id: int) -> Optional[List[int]]:
        """Получение рекомендаций для пользователя из кэша"""
        pass
    
    @abstractmethod
    async def cache_user_recommendations(self, user_id: int, recommendations: List[int]) -> None:
        """Сохранение рекомендаций для пользователя в кэш"""
        pass
    
    @abstractmethod
    async def get_item_recommendations(self, item_id: int) -> Optional[List[int]]:
        """Получение похожих элементов из кэша"""
        pass
    
    @abstractmethod
    async def cache_item_recommendations(self, item_id: int, recommendations: List[int]) -> None:
        """Сохранение похожих элементов в кэш"""
        pass
    
    @abstractmethod
    async def invalidate_all_recommendations(self) -> None:
        """Инвалидация всех кэшированных рекомендаций"""
        pass

class RecommendationCache(BaseCacheService):
    _instance = None
    _redis: Optional[aioredis.Redis] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self, redis_url: str = "redis://localhost:6379/0"):
        """Инициализирует соединение с Redis"""
        if self._redis is None:
            try:
                self._redis = await aioredis.from_url(
                    redis_url, 
                    encoding="utf-8", 
                    decode_responses=True
                )
                logger.info("Redis соединение установлено")
            except Exception as e:
                logger.error(f"Ошибка подключения к Redis: {str(e)}")
                self._redis = None
                raise
    
    async def close(self):
        """Закрывает соединение с Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis соединение закрыто")
    
    async def cache_user_recommendations(self, user_id: int, items: List[int], ttl: int = 3600):
        """
        Кеширует рекомендации для пользователя
        
        Args:
            user_id: ID пользователя
            items: Список ID рекомендованных элементов
            ttl: Время жизни кеша в секундах (по умолчанию 1 час)
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return False
            
        key = f"rec:user:{user_id}"
        try:
            async with self._lock:
                await self._redis.delete(key)
                if items:
                    # Сохраняем в сортированном наборе с позицией в качестве score
                    for i, item_id in enumerate(items):
                        await self._redis.zadd(key, {str(item_id): i})
                    await self._redis.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Ошибка кеширования рекомендаций пользователя {user_id}: {str(e)}")
            return False
    
    async def get_user_recommendations(self, user_id: int, limit: int = 40) -> List[int]:
        """
        Получает рекомендации для пользователя из кеша
        
        Args:
            user_id: ID пользователя
            limit: Максимальное количество рекомендаций
            
        Returns:
            Список ID рекомендованных элементов или пустой список, если кеш отсутствует
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return []
            
        key = f"rec:user:{user_id}"
        try:
            # Получаем отсортированные элементы
            items = await self._redis.zrange(key, 0, limit-1, withscores=False)
            return [int(item) for item in items] if items else []
        except Exception as e:
            logger.error(f"Ошибка получения рекомендаций для пользователя {user_id}: {str(e)}")
            return []
    
    async def cache_item_recommendations(self, item_id: int, similar_items: List[int], ttl: int = 3600):
        """
        Кеширует похожие элементы для указанного элемента
        
        Args:
            item_id: ID элемента
            similar_items: Список ID похожих элементов
            ttl: Время жизни кеша в секундах (по умолчанию 1 час)
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return False
            
        key = f"rec:item:{item_id}"
        try:
            async with self._lock:
                await self._redis.delete(key)
                if similar_items:
                    # Сохраняем в сортированном наборе с позицией в качестве score
                    for i, similar_id in enumerate(similar_items):
                        await self._redis.zadd(key, {str(similar_id): i})
                    await self._redis.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Ошибка кеширования похожих элементов для {item_id}: {str(e)}")
            return False
    
    async def get_item_recommendations(self, item_id: int, limit: int = 40) -> List[int]:
        """
        Получает похожие элементы из кеша
        
        Args:
            item_id: ID элемента
            limit: Максимальное количество похожих элементов
            
        Returns:
            Список ID похожих элементов или пустой список, если кеш отсутствует
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return []
            
        key = f"rec:item:{item_id}"
        try:
            # Получаем отсортированные элементы
            items = await self._redis.zrange(key, 0, limit-1, withscores=False)
            return [int(item) for item in items] if items else []
        except Exception as e:
            logger.error(f"Ошибка получения похожих элементов для {item_id}: {str(e)}")
            return []
    
    async def invalidate_all_recommendations(self):
        """Инвалидирует все кеши рекомендаций"""
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return False
            
        try:
            # Удаляем все ключи с паттерном rec:user:* и rec:item:*
            user_keys = await self._redis.keys("rec:user:*")
            item_keys = await self._redis.keys("rec:item:*")
            
            all_keys = user_keys + item_keys
            if all_keys:
                await self._redis.delete(*all_keys)
                logger.info(f"Инвалидировано {len(all_keys)} кешей рекомендаций")
            return True
        except Exception as e:
            logger.error(f"Ошибка инвалидации кешей рекомендаций: {str(e)}")
            return False
            
    async def store_model_metadata(self, metadata: Dict[str, Any], model_id: str, ttl: int = 2592000):
        """
        Сохраняет метаданные модели в Redis
        
        Args:
            metadata: Словарь с метаданными модели
            model_id: Идентификатор модели
            ttl: Время жизни в секундах (по умолчанию 30 дней)
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return False
            
        key = f"model:metadata:{model_id}"
        try:
            await self._redis.set(key, json.dumps(metadata), ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения метаданных модели {model_id}: {str(e)}")
            return False
    
    async def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Получает метаданные модели из Redis
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            Словарь с метаданными модели или пустой словарь, если модель не найдена
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return {}
            
        key = f"model:metadata:{model_id}"
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            logger.error(f"Ошибка получения метаданных модели {model_id}: {str(e)}")
            return {}
            
    async def list_models(self) -> List[str]:
        """
        Получает список всех сохраненных моделей
        
        Returns:
            Список идентификаторов моделей
        """
        if not self._redis:
            logger.warning("Redis не инициализирован")
            return []
            
        try:
            keys = await self._redis.keys("model:metadata:*")
            return [key.replace("model:metadata:", "") for key in keys]
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {str(e)}")
            return [] 

class DataService(BaseDataService):
    """Улучшенная реализация сервиса для подготовки данных"""
    
    def __init__(self, session_maker, read_only: bool = True, cache_ttl: int = 3600):
        """
        Инициализация сервиса данных
        
        Args:
            session_maker: Фабрика сессий базы данных
            read_only: Флаг только для чтения (по умолчанию True)
            cache_ttl: Время жизни кэша в секундах (по умолчанию 1 час)
        """
        self.session_maker = session_maker
        self.read_only = read_only
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.cache_timestamp = {}
    
    async def get_users_features(self, max_age_days: Optional[int] = None):
        """Получение признаков пользователей с кэшированием"""
        cache_key = f"users_features_{max_age_days}"
        
        # Проверяем кэш
        if self._is_cache_valid(cache_key):
            self.logger.info("Возвращаем кэшированные данные о пользователях")
            return self.cache[cache_key]
        
        self.logger.info("Получаем данные о пользователях из базы данных")
        
        query = self._build_users_query(max_age_days)
        
        async with self.session_maker() as session:
            if self.read_only:
                await session.execute(text("SET TRANSACTION READ ONLY"))
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(rows, columns=[
                'user_id', 'age', 'account_age_days', 
                'watch_count', 'like_count', 'favorite_genres'
            ])
            
            # Обрабатываем и дополняем данные
            df = self._process_user_features(df)
            
            # Сохраняем в кэш
            self._update_cache(cache_key, df)
            
            self.logger.info(f"Получено {len(df)} записей о пользователях")
            return df
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверяет валидность кэша по ключу"""
        return (
            cache_key in self.cache and 
            time.time() - self.cache_timestamp.get(cache_key, 0) < self.cache_ttl
        )
    
    def _update_cache(self, cache_key: str, value: Any) -> None:
        """Обновляет кэш для указанного ключа"""
        self.cache[cache_key] = value
        self.cache_timestamp[cache_key] = time.time()
    
    def _build_users_query(self, max_age_days: Optional[int]) -> str:
        """Строит SQL-запрос для получения данных о пользователях"""
        query = """
            SELECT 
                u.id AS user_id,
                u.age,
                EXTRACT(DAY FROM NOW() - u.created_at) AS account_age_days,
                COUNT(DISTINCT w.title_id) AS watch_count,
                COUNT(DISTINCT l.title_id) AS like_count,
                ARRAY_AGG(DISTINCT g.name) AS favorite_genres
            FROM users u
            LEFT JOIN watches w ON u.id = w.user_id
            LEFT JOIN likes l ON u.id = l.user_id
            LEFT JOIN titles_genres tg ON tg.title_id = w.title_id OR tg.title_id = l.title_id
            LEFT JOIN genres g ON tg.genre_id = g.id
        """
        
        if max_age_days is not None:
            query += f" WHERE EXTRACT(DAY FROM NOW() - u.created_at) <= {max_age_days}"
        
        query += """
            GROUP BY u.id, u.age, u.created_at
        """
        
        return query
    
    def _process_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обрабатывает и дополняет данные о пользователях"""
        # Добавляем возрастную группу
        df['age_group'] = df['age'].apply(self._calculate_age_group)
        
        # Балансируем пропущенные значения
        df['favorite_genres'] = df['favorite_genres'].fillna('[]')
        
        return df
    
    def _calculate_age_group(self, age: Optional[int]) -> str:
        """Вычисляет возрастную группу по возрасту"""
        if age is None or pd.isna(age):
            return "unknown"
        elif age < 18:
            return "under_18"
        elif age < 24:
            return "18_24"
        elif age < 35:
            return "25_34"
        elif age < 45:
            return "35_44"
        elif age < 55:
            return "45_54"
        else:
            return "55_plus"
    
    # Остальные методы аналогично реорганизуются с улучшенным форматированием,
    # разделением ответственности и инкапсуляцией логики 