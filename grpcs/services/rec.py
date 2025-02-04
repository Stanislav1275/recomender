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
    async def relavant_2_item(item_id: int):
        model = LightFMWrapperModel(LightFM(no_components=10, loss="bpr", random_state=60))
        m = model.load(f='data/data.csv')
        with open("data/dataset.pkl", 'rb') as f:
            dataset = pickle.load(f)
            recos = m.recommend_to_items(

                target_items=[item_id],
                dataset=dataset,
                k=40,
                filter_itself=True,
            )
            return recos
    @staticmethod
    async def rec(user_id: int):
        model = LightFMWrapperModel(LightFM(no_components=10, loss="bpr", random_state=60))
        m = model.load(f='data/data.csv')
        with open("data/dataset.pkl", 'rb') as f:
            dataset = pickle.load(f)
            # popular = m._recommend_cold(dataset=dataset)
            recos = m.recommend(

                users=[user_id],
                dataset=dataset,
                k=40,
                filter_viewed=True,
            )

            return recos
        # return recos.head(10)

    @staticmethod
    async def train():
        user_features = await DataPrepareService.get_users_features()
        items_features = await DataPrepareService.get_titles_features()
        interactions = await DataPrepareService.get_interactions()

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
            with open('data/dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)
                model.fit(dataset)
                model.save(f="data/data.csv")
                print(model.is_fitted)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            print(model)
