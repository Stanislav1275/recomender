import pickle

import pandas
import warnings
import numpy as np
#
from lightfm import LightFM
#

import faulthandler
from grpcs.services.data_preparer import DataPrepareService
from implicit.nearest_neighbours import TFIDFRecommender

warnings.filterwarnings('ignore')
# from lightfm import LightFM
import logging
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel, LightFMWrapperModel, load_model

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RecService:
    @staticmethod
    async def rec(user_id:int):
        model = LightFMWrapperModel(LightFM(no_components=10, loss="bpr", random_state=60))
        m = model.load(f='model/model.csv')
        print(m.is_fitted)
        with open("model/dataset.pkl", 'rb') as f:
            dataset = pickle.load(f)
            # popular = m._recommend_cold(dataset=dataset)
            recos = m.recommend(
                users=[user_id],
                dataset=dataset,
                k=40,

                filter_viewed=True,
            )

            print(recos)
            return recos
        # return recos.head(10)

    @staticmethod
    async def train():
        user_features = await DataPrepareService.get_users_features()
        items_features = await DataPrepareService.get_titles_features()
        interactions = pandas.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8],
                                         Columns.Datetime: [pandas.to_datetime("now"), pandas.to_datetime("now"),
                                                            pandas.to_datetime("now")], Columns.Weight: [1, 3, 5]})

        RANDOM_STATE = 60

        model = LightFMWrapperModel(LightFM(no_components=10, loss="bpr", random_state=RANDOM_STATE))
        try:
            dataset = Dataset.construct(
                interactions_df=interactions,
                user_features_df=user_features,
                cat_user_features=["age_group", "sex", "preference"],
                item_features_df=items_features,
                cat_item_features=["type_id", "genres", "categories", "count_chapters", "age_limit"],
            )

            with open('model/dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)
            model.fit(dataset)


        except Exception as e:
            logger.error(f"Error during training: {e}")
            print(model)
        print(1)
        model.save(f="model/model.csv")

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import os
# import threadpoolctl

# warnings.filterwarnings('ignore')

# from implicit.als import AlternatingLeastSquares
# from implicit.bpr import BayesianPersonalizedRanking
# from implicit.nearest_neighbours import CosineRecommender
# lightfm extension is required for the LighFM section. You can install it with `pip install rectools[lightfm]`
# try:
#     from lightfm import LightFM
# except ModuleNotFoundError:
#     pass
#
# from rectools import Columns
# from rectools.dataset import Dataset
# from rectools.models import (
#     LightFMWrapperModel,
# )
#
# # For vector models optimized ranking
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# threadpoolctl.threadpool_limits(1, "blas")
# DATA_PATH = Path("")
# users = pd.read_csv('users_en.csv', index_col=0)
# print(users.shape)
# users.head(2)
# items = pd.read_csv('items_en.csv', index_col=0)
# print(items.shape)
# items.head(2)
# interactions = (
#     pd.read_csv('interactions.csv', parse_dates=["last_watch_dt"])
#     .rename(columns={"last_watch_dt": Columns.Datetime})
# )
# print(interactions.shape)
# interactions.head(2)
# interactions[Columns.Weight] = np.where(interactions['watched_pct'] > 10, 3, 1)
# interactions = interactions[["user_id", "item_id", "datetime", "weight"]]
# print(interactions.shape)
# interactions.head(2)
# users = users[["user_id", "age", "sex"]]
# users.fillna('Unknown', inplace=True)
# user_features_frames = []
# for feature in ["sex", "age"]:
#     feature_frame = users.reindex(columns=[Columns.User, feature])
#     feature_frame.columns = ["id", "value"]
#     feature_frame["feature"] = feature
#     user_features_frames.append(feature_frame)
# user_features = pd.concat(user_features_frames)
# print(user_features.shape)
# user_features.head(2)
# items = items.loc[items[Columns.Item].isin(interactions[Columns.Item])].copy()
# items["genre"] = items["genres"].str.lower().str.replace(", ", ",", regex=False).str.split(",")
# genre_feature = items[["item_id", "genre"]].explode("genre")
# genre_feature.columns = ["id", "value"]
# genre_feature["feature"] = "genre"
# item_features = genre_feature
# print(item_features.shape)
# item_features.head(2)
# test_hot_users = [176549]  # have interactions and features
# print(interactions[interactions["user_id"] == test_hot_users[0]].shape)
# interactions[interactions["user_id"] == test_hot_users[0]].head(2)
# print(user_features[user_features["id"] == test_hot_users[0]].shape)
# var = user_features[user_features["id"] == test_hot_users[0]]
# test_warm_users = [1097541]  # have features but don't have interactions
# interactions[interactions["user_id"] == test_warm_users[0]].shape
# print(user_features[user_features["id"] == test_warm_users[0]].shape)
# user_features[user_features["id"] == test_warm_users[0]]
# test_cold_users = [99999999]  # don't have features or interactions
# interactions[interactions["user_id"] == test_cold_users[0]].shape
# user_features[user_features["id"] == test_cold_users[0]].shape
# dataset = Dataset.construct(
#     interactions_df=interactions,
#     user_features_df=user_features,
#     cat_user_features=["sex", "age"],
#     item_features_df=item_features,
#     cat_item_features=["genre"],
# )
# RANDOM_STATE=60
# model = LightFMWrapperModel(LightFM(no_components=10, loss="bpr", random_state=RANDOM_STATE))
# model.fit(dataset)
# recos = model.recommend(
#     users=test_hot_users + test_warm_users + test_cold_users,
#     dataset=dataset,
#     k=3,
#     filter_viewed=True,
# )
# recos.merge(items[["item_id", "title_orig"]], on="item_id").sort_values(["user_id", "rank"])
