import asyncio
import logging
from datetime import  datetime
import time  # Добавляем импорт time для кэширования

import numpy as np
from rectools import Columns
from sqlalchemy import func, select, distinct, text  # Добавляем импорт text
from sqlalchemy.exc import SQLAlchemyError

from models import RawUsers, Bookmarks, Rating, UserTitleData, \
    TitlesTitleRelation, Comments, UserBuys, TitleChapter, TitlesSites
import warnings
import os
import threadpoolctl

from utils.age_group import calculate_age_group
from utils.cat_chapters import categorize_chapters

warnings.filterwarnings('ignore')
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas")
USERS_LIMIT = 1000000
DAYS = 1000000
test_users_ids = [34]
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from core.database import SessionLocal
from models import Titles, TitlesCategories, TitlesGenres

MIN_VOTES = 100


def fetch_data_in_parallel():
    with SessionLocal() as db:
        with ThreadPoolExecutor() as executor:
            # Define tasks for parallel execution
            titles_future = executor.submit(
                pd.read_sql_query,
                db.query(Titles).filter_by(is_erotic=0, is_yaoi=0, uploaded=1).order_by(None).statement,
                db.bind
            )
            categories_future = executor.submit(
                pd.read_sql_query,
                db.query(TitlesCategories).statement,
                db.bind
            )
            genres_future = executor.submit(
                pd.read_sql_query,
                db.query(TitlesGenres).statement,
                db.bind
            )
            # similar_future = executor.submit(
            #     pd.read_sql_query,
            #     db.query(SimilarTitles).limit(USERS_LIMIT).statement,
            #     db.bind
            # )
            relations_future = executor.submit(
                pd.read_sql_query,
                db.query(TitlesTitleRelation).statement,
                db.bind
            )

            titles_pd = titles_future.result()
            titles_categories_pd = categories_future.result()
            titles_genres_pd = genres_future.result()
            titles_relations_pd = relations_future.result()
            # similar_titles_pd = similar_future.result()

    return titles_pd, titles_categories_pd, titles_genres_pd, titles_relations_pd


def map_ratings(rating: int):
    if rating <= 2:
        return -rating * 0.3
    if rating == 3:
        return -1
    if rating == 4:
        return 0.8
    return 3 - (rating * 0.7)


def map_bookmark_type_id(bookmark_type_id):
    switcher = {
        1: 7,
        2: 8,
        3: 7,
        4: -0.1,
        5: 3,
        6: -0.1,
    }
    return switcher.get(bookmark_type_id, 1)


class BlacklistManager:
    _black_titles_ids = None

    @classmethod
    def has_block_title(cls, model: Titles):
        return (model.count_chapters == 0) or model.is_yaoi == 1 or model.uploaded == 0

    @classmethod
    async def _fetch_black_titles_ids(cls):
        with SessionLocal() as db:
            black_query = db.query(Titles.id).filter(
                (Titles.count_chapters == 0) |
                (Titles.is_yaoi == 1) |
                (Titles.uploaded == 0)
            )

            site_filter_query = db.query(TitlesSites.title_id).filter(
                TitlesSites.site_id != 1
            )

            result = black_query.union(site_filter_query)
            ids = {row.id for row in result.all()}
            return ids

    @classmethod
    async def get_black_titles_ids(cls) -> set:
        if cls._black_titles_ids is None:
            cls._black_titles_ids = await cls._fetch_black_titles_ids()
        return cls._black_titles_ids

    @classmethod
    async def refresh_blacklist(cls):
        cls._black_titles_ids = None


class DataPrepareService:
    """Сервис для подготовки данных для рекомендательной системы"""

    def __init__(self, session_maker, read_only=True):
        """
        Инициализация сервиса подготовки данных
        
        Args:
            session_maker: Фабрика сессий базы данных
            read_only: Флаг только для чтения (по умолчанию True)
        """
        self.session_maker = session_maker
        self.read_only = read_only
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_ttl = 3600  # кэш на 1 час
        self.cache_timestamp = {}

    async def get_users_features(self, max_age_days=None):
        """Получение признаков пользователей"""
        cache_key = f"users_features_{max_age_days}"
        
        # Проверяем кэш
        if cache_key in self.cache:
            if time.time() - self.cache_timestamp.get(cache_key, 0) < self.cache_ttl:
                self.logger.info("Returning cached user features")
                return self.cache[cache_key]
        
        self.logger.info("Fetching user features from database")
        async with self.session_maker() as session:
            if self.read_only:
                # Устанавливаем сессию только для чтения
                session.execute(text("SET TRANSACTION READ ONLY"))
                
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
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(rows, columns=[
                'user_id', 'age', 'account_age_days', 
                'watch_count', 'like_count', 'favorite_genres'
            ])
            
            # Сохраняем в кэш
            self.cache[cache_key] = df
            self.cache_timestamp[cache_key] = time.time()
            
            self.logger.info(f"Fetched {len(df)} user records")
            return df

    async def get_titles_features(self, min_rating=None):
        """Получение признаков фильмов"""
        cache_key = f"titles_features_{min_rating}"
        
        # Проверяем кэш
        if cache_key in self.cache:
            if time.time() - self.cache_timestamp.get(cache_key, 0) < self.cache_ttl:
                self.logger.info("Returning cached title features")
                return self.cache[cache_key]
        
        self.logger.info("Fetching title features from database")
        async with self.session_maker() as session:
            if self.read_only:
                # Устанавливаем сессию только для чтения
                session.execute(text("SET TRANSACTION READ ONLY"))
                
            query = """
                SELECT 
                    t.id AS title_id,
                    t.title,
                    t.release_year,
                    t.rating,
                    t.duration_minutes,
                    COUNT(DISTINCT w.user_id) AS watch_count,
                    COUNT(DISTINCT l.user_id) AS like_count,
                    ARRAY_AGG(DISTINCT g.name) AS genres
                FROM titles t
                LEFT JOIN watches w ON t.id = w.title_id
                LEFT JOIN likes l ON t.id = l.title_id
                LEFT JOIN titles_genres tg ON t.id = tg.title_id
                LEFT JOIN genres g ON tg.genre_id = g.id
            """
            
            if min_rating is not None:
                query += f" WHERE t.rating >= {min_rating}"
            
            query += """
                GROUP BY t.id, t.title, t.release_year, t.rating, t.duration_minutes
            """
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(rows, columns=[
                'title_id', 'title', 'release_year', 'rating', 
                'duration_minutes', 'watch_count', 'like_count', 'genres'
            ])
            
            # Сохраняем в кэш
            self.cache[cache_key] = df
            self.cache_timestamp[cache_key] = time.time()
            
            self.logger.info(f"Fetched {len(df)} title records")
            return df

    async def get_interactions(self, min_days=None, interaction_types=None):
        """Получение взаимодействий пользователей с фильмами"""
        if interaction_types is None:
            interaction_types = ["watch", "like"]
            
        cache_key = f"interactions_{min_days}_{'-'.join(interaction_types)}"
        
        # Проверяем кэш
        if cache_key in self.cache:
            if time.time() - self.cache_timestamp.get(cache_key, 0) < self.cache_ttl:
                self.logger.info("Returning cached interactions")
                return self.cache[cache_key]
        
        self.logger.info(f"Fetching interactions from database: {interaction_types}")
        
        # Подготавливаем задачи для получения данных параллельно
        tasks = []
        
        # Получаем просмотры
        if "watch" in interaction_types:
            tasks.append(self._get_watches(min_days))
            
        # Получаем лайки
        if "like" in interaction_types:
            tasks.append(self._get_likes(min_days))
            
        # Ожидаем выполнения всех задач
        results = await asyncio.gather(*tasks)
        
        # Объединяем результаты
        interactions = pd.concat(results, ignore_index=True)
        
        # Сохраняем в кэш
        self.cache[cache_key] = interactions
        self.cache_timestamp[cache_key] = time.time()
        
        self.logger.info(f"Fetched {len(interactions)} interaction records")
        return interactions

    async def _get_watches(self, min_days=None):
        """Получение просмотров"""
        async with self.session_maker() as session:
            if self.read_only:
                # Устанавливаем сессию только для чтения
                session.execute(text("SET TRANSACTION READ ONLY"))
                
            query = """
                SELECT 
                    w.user_id,
                    w.title_id,
                    1.0 AS weight,
                    'watch' AS interaction_type,
                    w.watched_at
                FROM watches w
            """
            
            if min_days is not None:
                query += f" WHERE EXTRACT(DAY FROM NOW() - w.watched_at) <= {min_days}"
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            return pd.DataFrame(rows, columns=[
                'user_id', 'title_id', 'weight', 'interaction_type', 'timestamp'
            ])

    async def _get_likes(self, min_days=None):
        """Получение лайков"""
        async with self.session_maker() as session:
            if self.read_only:
                # Устанавливаем сессию только для чтения
                session.execute(text("SET TRANSACTION READ ONLY"))
                
            query = """
                SELECT 
                    l.user_id,
                    l.title_id,
                    2.0 AS weight,
                    'like' AS interaction_type,
                    l.liked_at
                FROM likes l
            """
            
            if min_days is not None:
                query += f" WHERE EXTRACT(DAY FROM NOW() - l.liked_at) <= {min_days}"
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            return pd.DataFrame(rows, columns=[
                'user_id', 'title_id', 'weight', 'interaction_type', 'timestamp'
            ])

    async def prepare_data_for_model(self, min_interaction_days=90, min_title_rating=None):
        """Подготовка данных для обучения модели"""
        self.logger.info("Preparing data for model training")
        
        # Параллельно получаем все необходимые данные
        users_features, titles_features, interactions = await asyncio.gather(
            self.get_users_features(),
            self.get_titles_features(min_rating=min_title_rating),
            self.get_interactions(min_days=min_interaction_days)
        )
        # Оставляем только взаимодействия с фильмами, которые есть в наборе фильмов
        valid_titles = set(titles_features['title_id'])
        interactions = interactions[interactions['title_id'].isin(valid_titles)]
        
        # Оставляем только взаимодействия пользователей, которые есть в наборе пользователей
        valid_users = set(users_features['user_id'])
        interactions = interactions[interactions['user_id'].isin(valid_users)]
        
        # Создаем отображения ID пользователей и фильмов в индексы для модели
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users_features['user_id'].unique())}
        title_id_to_idx = {title_id: idx for idx, title_id in enumerate(titles_features['title_id'].unique())}
        
        # Преобразуем ID в индексы
        interactions['user_idx'] = interactions['user_id'].map(user_id_to_idx)
        interactions['title_idx'] = interactions['title_id'].map(title_id_to_idx)
        
        # Подготавливаем данные для обучения модели
        train_data = (
            interactions['user_idx'].values,
            interactions['title_idx'].values,
            interactions['weight'].values
        )
        
        # Создаем обратные отображения для использования в предсказаниях
        idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
        idx_to_title_id = {idx: title_id for title_id, idx in title_id_to_idx.items()}
        
        return {
            'train_data': train_data,
            'users_features': users_features,
            'titles_features': titles_features,
            'interactions': interactions,
            'user_mappings': {
                'user_id_to_idx': user_id_to_idx,
                'idx_to_user_id': idx_to_user_id
            },
            'title_mappings': {
                'title_id_to_idx': title_id_to_idx,
                'idx_to_title_id': idx_to_title_id
            }
        }

    async def prepare_train_val_data(self, val_ratio=0.2, min_interaction_days=90, min_title_rating=None):
        """Подготовка данных для обучения и валидации модели"""
        all_data = await self.prepare_data_for_model(
            min_interaction_days=min_interaction_days, 
            min_title_rating=min_title_rating
        )
        
        interactions = all_data['interactions']
        
        # Сортируем по времени, чтобы разделение было хронологическим
        interactions = interactions.sort_values('timestamp')
        
        # Определяем точку разделения (последние val_ratio % данных - для валидации)
        split_idx = int(len(interactions) * (1 - val_ratio))
        
        # Разделяем данные
        train_interactions = interactions.iloc[:split_idx]
        val_interactions = interactions.iloc[split_idx:]
        
        # Подготавливаем данные для обучения
        train_data = (
            train_interactions['user_idx'].values,
            train_interactions['title_idx'].values,
            train_interactions['weight'].values
        )
        
        # Подготавливаем данные для валидации
        val_data = (
            val_interactions['user_idx'].values,
            val_interactions['title_idx'].values,
            val_interactions['weight'].values
        )
        
        # Обновляем словарь с данными
        all_data['train_data'] = train_data
        all_data['val_data'] = val_data
        
        return all_data

    async def get_blacklisted_titles(self):
        """Получение списка заблокированных фильмов"""
        async with self.session_maker() as session:
            if self.read_only:
                # Устанавливаем сессию только для чтения
                session.execute(text("SET TRANSACTION READ ONLY"))
                
            query = """
                SELECT id FROM titles WHERE is_blacklisted = TRUE
            """
            result = await session.execute(text(query))
            rows = result.fetchall()
            return [row[0] for row in rows]

    async def get_user_recent_interactions(self, user_id, limit=10):
        """Получение последних взаимодействий пользователя"""
        # Получаем последние просмотры
        async with self.session_maker() as session:
            if self.read_only:
                # Устанавливаем сессию только для чтения
                session.execute(text("SET TRANSACTION READ ONLY"))
                
            # Объединяем просмотры и лайки в одном запросе
            query = """
                (SELECT 
                    title_id, 
                    'watch' AS interaction_type, 
                    watched_at AS timestamp
                FROM watches 
                WHERE user_id = :user_id)
                
                UNION ALL
                
                (SELECT 
                    title_id, 
                    'like' AS interaction_type, 
                    liked_at AS timestamp
                FROM likes 
                WHERE user_id = :user_id)
                
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            
            result = await session.execute(
                text(query), 
                {"user_id": user_id, "limit": limit}
            )
            
            rows = result.fetchall()
            
            return pd.DataFrame(rows, columns=['title_id', 'interaction_type', 'timestamp'])

    # Добавляем метод для проверки доступа
    async def check_read_only(self):
        """Проверяет, работает ли сервис в режиме только для чтения"""
        return self.read_only
