from pandas.core.interchange.dataframe_protocol import Column
from rectools import Columns

from models import RawUsers, Titles, TitlesCategories, TitlesGenres, Bookmarks
import warnings
import os
import threadpoolctl

from utils.age_group import calculate_age_group
from utils.cat_chapters import categorize_chapters

warnings.filterwarnings('ignore')
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas")
USERS_LIMIT = 10000


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from core.database import SessionLocal
from models import Titles, TitlesCategories, TitlesGenres


def fetch_data_in_parallel():
    with SessionLocal() as db:
        with ThreadPoolExecutor() as executor:
            # Define tasks for parallel execution
            titles_future = executor.submit(
                pd.read_sql_query,
                db.query(Titles).filter_by(is_erotic=False, is_yaoi=False).limit(USERS_LIMIT).statement,
                db.bind
            )
            categories_future = executor.submit(
                pd.read_sql_query,
                db.query(TitlesCategories).limit(USERS_LIMIT).statement,
                db.bind
            )
            genres_future = executor.submit(
                pd.read_sql_query,
                db.query(TitlesGenres).limit(USERS_LIMIT).statement,
                db.bind
            )

            titles_pd = titles_future.result()
            titles_categories_pd = categories_future.result()
            titles_genres_pd = genres_future.result()

    return titles_pd, titles_categories_pd, titles_genres_pd
def map_bookmark_type_id(bookmark_type_id):
    switcher = {
        1: 7,
        2: 8,
        3: 7,
        4: 0.1,
        5: 3,
        6: 0.1,
    }
    return switcher.get(bookmark_type_id, 1)
class BookmarksPreparer:
    @staticmethod
    async def to_interact():
        with SessionLocal() as db:
            print(db.query(Bookmarks).limit(5).statement)
            bookmarks_pd = pd.read_sql_query(db.query(Bookmarks).where(is_default=1).limit(USERS_LIMIT).statement, db.bind)
            bookmarks_pd.rename({"title_id":Columns.Item, "bookmark_type_id":Columns.Weight}, axis='columns', inplace=True)
            bookmarks_pd[Columns.Weight] = bookmarks_pd[Columns.Weight].apply(map_bookmark_type_id)

            bookmarks_pd.drop(columns=["id"], inplace=True)
            return bookmarks_pd

class DataPrepareService:
    @staticmethod
    async def get_interactions():
        return await BookmarksPreparer.to_interact()
    @staticmethod
    async def get_users_features():
        with SessionLocal() as db:
            users_pd = pd.read_sql_query(
                db.query(RawUsers).filter_by(is_banned=False, is_active=True).limit(USERS_LIMIT).statement,
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
            return user_features

    @staticmethod
    async def get_titles_features():
        titles_pd, titles_categories_pd, titles_genres_pd = fetch_data_in_parallel()

        titles_pd.fillna('Unknown', inplace=True)

        titles_categories_agg = titles_categories_pd.groupby('title_id')['category_id'].apply(
            lambda x: ' '.join(map(str, sorted(x)))).reset_index()
        titles_genres_agg = titles_genres_pd.groupby('title_id')['genre_id'].apply(
            lambda x: ' '.join(map(str, sorted(x)))).reset_index()

        titles_features = pd.merge(titles_pd, titles_categories_agg, left_on='id', right_on='title_id', how='left')
        titles_features = pd.merge(titles_features, titles_genres_agg, left_on='id', right_on='title_id', how='left')

        titles_features['category_id'].fillna('0', inplace=True)
        titles_features['genre_id'].fillna('0', inplace=True)
        titles_features.rename(columns={'category_id': 'categories', 'genre_id': 'genres'}, inplace=True)
        titles_features['count_chapters'] = titles_features['count_chapters'].apply(categorize_chapters)

        features = ['status_id', 'age_limit', 'count_chapters', 'type_id', 'categories', 'genres']
        titles_features_long = pd.melt(titles_features, id_vars=['id'], value_vars=features,
                                       var_name='feature', value_name='value', ignore_index=True)
        return titles_features_long
