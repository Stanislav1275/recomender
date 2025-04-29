import asyncio
import logging
from datetime import  datetime

import numpy as np
from rectools import Columns
from sqlalchemy import func, select, distinct
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
    @staticmethod
    async def boost_int_weights_by_paid(df: pd.DataFrame, boost_factor: float = 1.2) -> pd.DataFrame:
        """
        Boost interaction weights for titles with any paid chapters.
        Applies careful boost to avoid over-weighting.

        Args:
            df: Input DataFrame with interactions
            boost_factor: Multiplicative factor for paid titles (default: 1.2 = +20%)

        Returns:
            Modified DataFrame with boosted weights
        """
        if Columns.Weight not in df.columns or Columns.Item not in df.columns:
            return df

        black_list = await BlacklistManager.get_black_titles_ids()

        with SessionLocal() as db:
            paid_titles = db.query(
                distinct(TitleChapter.title_id)
            ).filter(
                TitleChapter.is_paid == True,
                TitleChapter.title_id.in_(df[Columns.Item].unique()),
                ~TitleChapter.title_id.in_(black_list)
            ).all()

            paid_title_ids = {t[0] for t in paid_titles}

        if paid_title_ids:
            paid_mask = df[Columns.Item].isin(paid_title_ids)
            df.loc[paid_mask, Columns.Weight] = df.loc[paid_mask, Columns.Weight] * boost_factor

        return df

    @staticmethod
    async def _get_user_title_data(black_list: set):

        with SessionLocal() as db:
            cur_date = datetime.now()
            # thirty_days_ago = cur_date - timedelta(days=DAYS)
            user_title_data_pd = pd.read_sql_query(
                db.query(UserTitleData)
                .filter(
                    # UserTitleData.last_read_date >= thirty_days_ago,
                    ~UserTitleData.title_id.in_(black_list))
                .statement, db.bind)
            user_title_data_pd.rename({"last_read_date": Columns.Datetime}, inplace=True)
            new_rows = []

            for index, row in user_title_data_pd.iterrows():
                chapter_votes = row['chapter_votes'] if 'chapter_votes' in row else []
                chapter_views = row['chapter_views'] if 'chapter_views' in row else []

                for _ in chapter_votes:
                    new_rows.append({
                        Columns.User: row[Columns.User],
                        Columns.Item: row["title_id"],
                        Columns.Weight: 6,
                        Columns.Datetime: row['last_read_date']
                    })

                for _ in chapter_views:
                    new_rows.append({
                        Columns.User: row[Columns.User],
                        Columns.Item: row["title_id"],
                        Columns.Weight: 3,
                        Columns.Datetime: row['last_read_date']
                    })

            final_df = pd.DataFrame(new_rows)
            final_df = final_df[[Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]]
            return final_df

    @staticmethod
    async def _get_bookmarks(black_list: set):
        with SessionLocal() as db:
            bookmarks_pd = pd.read_sql_query(db.query(Bookmarks).filter(Bookmarks.is_default == 1,
                                                                        ~Bookmarks.title_id.in_(black_list)).limit(
                USERS_LIMIT).statement,
                                             db.bind)
            bookmarks_pd.rename({"title_id": Columns.Item, "bookmark_type_id": Columns.Weight}, axis='columns',
                                inplace=True)
            bookmarks_pd[Columns.Datetime] = pd.to_datetime("now")
            bookmarks_pd[Columns.Weight] = bookmarks_pd[Columns.Weight].apply(map_bookmark_type_id)
            bookmarks_pd.drop(columns=["id", "is_default"], inplace=True)
            bookmarks_pd = bookmarks_pd[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]
            return bookmarks_pd

    @staticmethod
    async def _get_comments(black_list: set):
        with SessionLocal() as db:
            try:
                query = db.query(Comments).filter(
                    Comments.title_id.isnot(None),
                    ~Comments.title_id.in_(black_list),
                    Comments.is_blocked == 0,
                    Comments.is_deleted == 0
                ).limit(USERS_LIMIT)

                comments_pd = pd.read_sql_query(query.statement, db.bind)

                if not comments_pd.empty:
                    comments_pd = (
                        comments_pd
                        .assign(**{Columns.Weight: 1.1})
                        .rename({
                            'user_id': Columns.User,
                            'title_id': Columns.Item,
                            'date': Columns.Datetime
                        }, axis='columns')
                    )
                else:
                    comments_pd = pd.DataFrame(columns=[
                        Columns.User,
                        Columns.Item,
                        Columns.Weight,
                        Columns.Datetime
                    ])
                comments_pd = comments_pd[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]

                return comments_pd

            except SQLAlchemyError as e:
                logger.error(f"Database error: {str(e)}")
                return pd.DataFrame()

    @staticmethod
    async def _get_user_buys(black_list: set):
        with (SessionLocal() as db):
            query = db.query(UserBuys.user_id, UserBuys.date, TitleChapter.title_id
                             ).join(TitleChapter, UserBuys.chapter_id == TitleChapter.id).filter(
                ~TitleChapter.title_id.in_(black_list)
            ).limit(1000)

            user_buys_pd = pd.read_sql_query(query.statement, db.bind)
            user_buys_pd.rename({"date": Columns.Datetime, "title_id": Columns.Item, "user_id": Columns.User},
                                axis='columns',
                                inplace=True)
            user_buys_pd[Columns.Weight] = 5
            return user_buys_pd

    @staticmethod
    async def _get_ratings(black_list: set):
        with SessionLocal() as db:
            cur_date = datetime.now()
            # thirty_days_ago = cur_date - timedelta(days=DAYS)
            # .filter(Rating.date.between(thirty_days_ago, cur_date))
            query = db.query(Rating).filter(
                # Rating.date >= thirty_days_ago,
                ~Rating.title_id.in_(black_list)).limit(1000).statement
            ratings_pd = pd.read_sql_query(query, db.bind)
            ratings_pd.rename({"date": Columns.Datetime, "title_id": Columns.Item, "rating": Columns.Weight},
                              axis='columns',
                              inplace=True)
            ratings_pd.drop(columns=["id"], inplace=True)
            ratings_pd[Columns.Weight] = ratings_pd[Columns.Weight].apply(map_ratings)
            ratings_pd = ratings_pd[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]
            return ratings_pd

    @staticmethod
    async def _apply_temporal_decay(df, omega: float = 0.01, max_decay: float = 0.7):
        df.head(10)
        """
            Apply improved temporal decay to interaction weights using formula:
             W = B * (max_decay + (1-max_decay) * e^(-ω * dif^0.5))
    
            Args:
                df: Input DataFrame
                omega: Decay coefficient (default 0.01) - более мягкий параметр
                max_decay: Минимальная доля веса (0.7 = сохраняем минимум 70% исходного веса)
        """
        if Columns.Weight not in df.columns or Columns.Item not in df.columns:
            return df

        black_list = await BlacklistManager.get_black_titles_ids()

        with SessionLocal() as db:
            result = db.query(Titles.id, Titles.last_chapter_uploaded).where(
                Titles.id.in_(df[Columns.Item].unique()),
                ~Titles.id.in_(black_list),
                Titles.last_chapter_uploaded != None
            )
        title_dates = {id_: date for id_, date in result.all()}
        current_date = pd.to_datetime('today')
        mask = (df[Columns.Weight] > 1) & (df[Columns.Item].isin(title_dates))

        last_upload_dates = df[Columns.Item].map(title_dates)
        dif = (current_date - pd.to_datetime(last_upload_dates)).dt.days
        decay_factor = max_decay + (1 - max_decay) * np.exp(-omega * np.sqrt(dif))
        df.loc[mask, Columns.Weight] = df.loc[mask, Columns.Weight] * decay_factor[mask]
        df.head(10)

        return df

    @staticmethod
    async def get_interactions():
        black_list = await BlacklistManager.get_black_titles_ids()
        bookmarks, ratings, user_title_data, comments, user_buys = await asyncio.gather(
            DataPrepareService._get_bookmarks(black_list),
            DataPrepareService._get_ratings(black_list),
            DataPrepareService._get_user_title_data(black_list),
            DataPrepareService._get_comments(black_list),
            DataPrepareService._get_user_buys(black_list),

        )

        combined_df = pd.concat([bookmarks, ratings, user_title_data, comments, user_buys], ignore_index=True)
        print(combined_df.head())
        await DataPrepareService._apply_temporal_decay(combined_df)
        await DataPrepareService.boost_int_weights_by_paid(combined_df)
        print(combined_df.head())
        combined_df.to_pickle("data/intereaction.pickle")

        return combined_df

    @staticmethod
    async def get_users_features():
        with SessionLocal() as db:
            users_pd = pd.read_sql_query(
                db.query(RawUsers).filter_by(is_banned=0).limit(3000).statement,
                db.bind
            )
            users_pd.fillna('Unknown', inplace=True)
            users_pd['birthday'] = pd.to_datetime(users_pd['birthday'], errors='coerce')
            users_pd["age_group"] = users_pd["birthday"].apply(calculate_age_group)

            user_features_frames = []
            for feature in ["sex", "age_group", "preference"]:
                feature_frame = users_pd.reindex(columns=["id", feature])
                feature_frame.columns = ["id", "value"]
                feature_frame["feature"] = feature
                user_features_frames.append(feature_frame)

            user_features = pd.concat(user_features_frames, ignore_index=True)
            user_features.to_pickle("data/user_features.pickle")
            return user_features

    @staticmethod
    async def get_titles_features():
        titles_pd, titles_categories_pd, titles_genres_pd, relations_pd = fetch_data_in_parallel()

        # Обработка relations
        if not relations_pd.empty:
            relations_agg = (
                relations_pd
                .sort_values('id')
                .groupby('title_id')['relation_list_id']
                .first()
                .reset_index(name='relation_list')
            )
        else:
            relations_agg = pd.DataFrame(columns=['title_id', 'relation_list'])

        titles_genres_agg = titles_genres_pd.groupby('title_id')['genre_id'].apply(list).reset_index()
        titles_categories_agg = titles_categories_pd.groupby('title_id')['category_id'].apply(list).reset_index()

        titles_features = pd.merge(titles_pd, titles_categories_agg, left_on='id', right_on='title_id', how='left')
        titles_features = pd.merge(titles_features, titles_genres_agg, left_on='id', right_on='title_id', how='left')
        titles_features = pd.merge(titles_features, relations_agg, left_on='id', right_on='title_id', how='left')

        titles_features['category_id'] = titles_features['category_id'].fillna('0')
        titles_features['genre_id'] = titles_features['genre_id'].fillna('0')
        titles_features.rename(columns={'category_id': 'categories', 'genre_id': 'genres'}, inplace=True)

        titles_features['count_chapters'] = titles_features['count_chapters'].fillna(0).apply(categorize_chapters)

        if 'relation_list' not in titles_features.columns:
            titles_features['relation_list'] = 0
        titles_features['relation_list'] = titles_features['relation_list'].fillna(0).astype('Int64')

        features = ['status_id', 'age_limit', 'count_chapters', 'type_id', 'categories', 'genres', 'relation_list']
        titles_features_long = titles_features.melt(
            id_vars=['id'],
            value_vars=features,
            var_name='feature',
            value_name='value'
        ).explode('value')
        # titles_features_long[Columns.Weight] = 2
        titles_features_long.to_pickle("data/title_features.pickle")

        return titles_features_long
