from datetime import timedelta, datetime

import pytz
from pandas.core.interchange.dataframe_protocol import Column
from rectools import Columns
from sqlalchemy import func

from models import RawUsers, Titles, TitlesCategories, TitlesGenres, Bookmarks, Rating, UserTitleData
import warnings
import os
import threadpoolctl
import json
from utils.age_group import calculate_age_group
from utils.cat_chapters import categorize_chapters

warnings.filterwarnings('ignore')
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas")
USERS_LIMIT = 20000000


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


def map_ratings(rating: int):
    if rating <= 2:
        return rating * 0.3
    if rating == 3:
        return 2
    if rating == 4:
        return 3
    return rating * 0.7


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


class UserTitleDataPreparer:
    @staticmethod
    async def to_interact():
        with SessionLocal() as db:
            user_title_data_pd = pd.read_sql_query(db.query(UserTitleData).statement.limit(USERS_LIMIT), db.bind)
            user_title_data_pd.rename({"last_read_date": Columns.Datetime}, inplace=True)
            new_rows = []

            for index, row in user_title_data_pd.iterrows():
                chapter_votes = row['chapter_votes'] if 'chapter_votes' in row else []
                chapter_views = row['chapter_views'] if 'chapter_views' in row else []

                for _ in chapter_votes:
                    new_rows.append({
                        Columns.User: row[Columns.User],
                        Columns.Item: row["title_id"],
                        Columns.Weight: 3,
                        Columns.Datetime: row['last_read_date']
                    })

                for _ in chapter_views:
                    new_rows.append({
                        Columns.User: row[Columns.User],
                        Columns.Item: row["title_id"],
                        Columns.Weight: 1.2,
                        Columns.Datetime: row['last_read_date']
                    })

            final_df = pd.DataFrame(new_rows)
            final_df = final_df[[Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]]
            return final_df


class BookmarksPreparer:
    @staticmethod
    async def to_interact():
        with SessionLocal() as db:
            bookmarks_pd = pd.read_sql_query(db.query(Bookmarks).filter_by(is_default=1).limit(USERS_LIMIT).statement,
                                             db.bind)
            bookmarks_pd.rename({"title_id": Columns.Item, "bookmark_type_id": Columns.Weight}, axis='columns',
                                inplace=True)
            bookmarks_pd[Columns.Datetime] = pd.to_datetime("now")
            bookmarks_pd[Columns.Weight] = bookmarks_pd[Columns.Weight].apply(map_bookmark_type_id)
            bookmarks_pd.drop(columns=["id", "is_default"], inplace=True)
            bookmarks_pd = bookmarks_pd[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]
            return bookmarks_pd


class RatingPraparer:
    @staticmethod
    async def to_interact():
        with SessionLocal() as db:
            cur_date = func.current_date()
            thirty_days_ago = cur_date - timedelta(days=1000)
            # .filter(Rating.date.between(thirty_days_ago, cur_date))
            query = db.query(Rating).limit(USERS_LIMIT).statement
            ratings_pd = pd.read_sql_query(query, db.bind)
            ratings_pd.rename({"date": Columns.Datetime, "title_id": Columns.Item, "rating": Columns.Weight},
                              axis='columns',
                              inplace=True)
            ratings_pd.drop(columns=["id"], inplace=True)
            ratings_pd[Columns.Weight] = ratings_pd[Columns.Weight].apply(map_ratings)
            ratings_pd = ratings_pd[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]
            return ratings_pd


class DataPrepareService:
    @staticmethod
    async def get_interactions():
        # todo parallel
        bookmarks = await BookmarksPreparer.to_interact()
        ratings = await RatingPraparer.to_interact()
        user_title_data = await UserTitleDataPreparer.to_interact()
        combined_df = pd.concat([bookmarks, ratings, user_title_data], ignore_index=True)
        return combined_df

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
