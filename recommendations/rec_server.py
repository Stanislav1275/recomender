import grpc
from grpc import aio as grpc_aio
from concurrent import futures
from recommendations.protos import recommendations_pb2_grpc
from recommendations.protos.recommendations_pb2 import RecommendationResponse, TrainResponse, InteractionResponse
from recommendations.rec_service import ModelManager, RecService
import pandas as pd
from rectools.dataset import Dataset
import logging


class RecommenderService(recommendations_pb2_grpc.RecommenderServicer):
    def __init__(self, model_manager=None, config=None):
        self.model_manager = model_manager or ModelManager()
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
    async def _handle_request(self, action_func, context, **kwargs):
        """Общий обработчик запросов с унифицированной обработкой ошибок"""
        try:
            result = await action_func(**kwargs, context=context)
            return result
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            
    async def Train(self, request, context):
        """Метод был неправильно реализован - исправлено"""
        await RecService._handle_request(context)
        try:
            await RecService.train(context)
            return TrainResponse(
                success=True,
                message="Training completed",
                version=ModelManager().current_version
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetUserRecommendations(self, request, context):
        return await self._handle_request(
            RecService.rec,
            context,
            user_id=request.user_id
        )
            
    async def GetTitleRelevant(self, request, context):
        return await self._handle_request(
            RecService.relevant,
            context,
            item_id=request.title_id
        )

    async def TrainModel(self, request, context):
        try:
            await RecService.train(context)
            return TrainResponse(
                success=True,
                message="Training completed",
                version=ModelManager().current_version
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RegisterInteraction(self, request, context):
        """Регистрирует новое взаимодействие пользователя и обновляет быстрый ранжировщик"""
        try:
            # Создаем сообщение о взаимодействии
            from recommendations.core.messaging.message import InteractionMessage
            interaction = InteractionMessage(
                user_id=request.user_id,
                item_id=request.item_id,
                interaction_type=request.interaction_type,
                weight=request.weight
            )
            
            # Отправляем в очередь сообщений, если настроена
            try:
                from recommendations.core.messaging.producer import MessageProducer
                producer = MessageProducer(bootstrap_servers=self.config.kafka_servers)
                await producer.send_interaction(interaction)
            except ImportError:
                self.logger.warning("MessageProducer not available, skipping message queue")
            
            # Обрабатываем взаимодействие в быстром ранжировщике
            try:
                from recommendations.core.fast_ranker.ranker import FastRanker
                ranker = FastRanker(self.model_manager, None, None, {})
                await ranker.hot_interact(
                    user_id=request.user_id,
                    item_id=request.item_id,
                    interaction_type=request.interaction_type,
                    weight=request.weight
                )
                self.logger.info(f"Processed hot interaction: user={request.user_id}, item={request.item_id}")
            except ImportError:
                self.logger.warning("FastRanker not available, skipping hot interaction processing")
            
            return InteractionResponse(success=True, message="Interaction registered")
            
        except Exception as e:
            error_msg = f"Error registering interaction: {str(e)}"
            self.logger.error(error_msg)
            await context.abort(grpc.StatusCode.INTERNAL, error_msg)

    def export_dataset_to_pandas(self, dataset: Dataset):
        # Экспортируем взаимодействия
        interactions_df = pd.DataFrame({
            'user_id': dataset.interactions.user_ids,
            'item_id': dataset.interactions.item_ids,
            'rating': dataset.interactions.ratings,
            'timestamp': dataset.interactions.timestamps
        })
        
        # Экспортируем фичи пользователей и элементов
        user_features = dataset.user_features_df if hasattr(dataset, 'user_features_df') else None
        item_features = dataset.item_features_df if hasattr(dataset, 'item_features_df') else None
        
        return interactions_df, user_features, item_features

    async def _update_model(self, interactions_df: pd.DataFrame):
        """Обновляет модель с новыми взаимодействиями"""
        try:
            # Получаем текущий датасет
            dataset = self.model_manager._dataset
            if dataset is None:
                self.logger.warning("No dataset available for model update")
                return False
            
            # Создаем новый датасет с добавленными взаимодействиями
            from rectools import Columns
            
            # Преобразуем колонки для соответствия RecTools
            interactions_df = interactions_df.rename(columns={
                'user_id': Columns.User,
                'item_id': Columns.Item,
                'weight': Columns.Weight,
                'timestamp': Columns.Datetime
            })
            
            # Обновляем датасет
            new_dataset = dataset.update_interactions(interactions_df)
            
            # Обучаем модель на обновленном датасете
            if hasattr(self.model_manager, 'fit') and callable(self.model_manager.fit):
                success = await self.model_manager.fit(new_dataset)
                if success:
                    self.logger.info("Model successfully updated with new interactions")
                else:
                    self.logger.warning("Failed to update model with new interactions")
                return success
            else:
                self.logger.warning("Model manager does not support fit method")
                return False
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False

    async def initialize(self):
        """Инициализирует быстрый рекомендатель"""
        try:
            # Получаем модель и датасет
            model = self.model_manager._model
            
            if model and hasattr(model, 'model') and hasattr(model.model, 'item_embeddings'):
                # Извлекаем векторные представления элементов из LightFM модели в rectools
                self.item_vectors = {
                    item_id: model.model.item_embeddings[int_id] 
                    for item_id, int_id in model._item_id_map.external_to_internal.items()
                }
                self.logger.info(f"Loaded {len(self.item_vectors)} item vectors")
                
                # Извлекаем векторные представления пользователей
                if hasattr(model.model, 'user_embeddings'):
                    self.user_vectors = {
                        user_id: model.model.user_embeddings[int_id]
                        for user_id, int_id in model._user_id_map.external_to_internal.items()
                    }
                    self.logger.info(f"Loaded {len(self.user_vectors)} user vectors")
            else:
                self.logger.warning("Model does not support embeddings extraction")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
