import asyncio
import logging
import os
import grpc
from concurrent import futures
from typing import Dict, Any

from grpc.aio import server as grpc_aio_server
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from recommendations.rec_service import RecService, ModelManager
from recommendations.data_preparer import DataPrepareService
from recommendations.auth import AuthService
from recommendations.cache_service import RecommendationCache

# Генерируем код для gRPC сервера из proto-файла
from recommendations.proto import recommendation_service_pb2, recommendation_service_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получаем настройки из переменных окружения
DB_URL = os.environ.get("REC_DB_URL", "postgresql+asyncpg://user:password@localhost:5432/rec_db")
GRPC_HOST = os.environ.get("REC_GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.environ.get("REC_GRPC_PORT", "50051"))
CACHE_URL = os.environ.get("REC_CACHE_URL", "redis://localhost:6379/0")


class RecommendationServicer(recommendation_service_pb2_grpc.RecommendationServiceServicer):
    """
    Реализация gRPC сервиса рекомендаций
    """
    
    def __init__(self, session_maker):
        self.session_maker = session_maker
        self.data_preparer = DataPrepareService(session_maker=session_maker, read_only=True)
        self.model_manager = ModelManager(dp_service=self.data_preparer)
        self.logger = logging.getLogger(__name__)
    
    async def GetRecommendations(self, request, context):
        """
        Получение персональных рекомендаций для пользователя
        """
        try:
            user_id = request.user_id
            limit = request.limit or 40
            filter_viewed = request.filter_viewed
            
            self.logger.info(f"Запрос рекомендаций для пользователя {user_id}")
            
            recommendations = await RecService.rec(user_id, context=context)
            
            # Ограничиваем количество рекомендаций
            if limit > 0 and len(recommendations) > limit:
                recommendations = recommendations[:limit]
            
            response = recommendation_service_pb2.RecommendationsResponse(
                item_ids=recommendations,
                scores=[1.0] * len(recommendations)  # Просто заполняем единицами, так как не используем оценки
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении рекомендаций: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Внутренняя ошибка: {str(e)}")
    
    async def GetSimilarItems(self, request, context):
        """
        Получение похожих элементов для заданного элемента
        """
        try:
            item_id = request.item_id
            limit = request.limit or 40
            
            self.logger.info(f"Запрос похожих элементов для элемента {item_id}")
            
            similar_items = await RecService.relevant(item_id, context=context)
            
            # Ограничиваем количество рекомендаций
            if limit > 0 and len(similar_items) > limit:
                similar_items = similar_items[:limit]
            
            response = recommendation_service_pb2.RecommendationsResponse(
                item_ids=similar_items,
                scores=[1.0] * len(similar_items)  # Просто заполняем единицами
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении похожих элементов: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Внутренняя ошибка: {str(e)}")
    
    async def TrainModel(self, request, context):
        """
        Запуск обучения модели
        """
        try:
            parameters = {}
            # Преобразуем параметры из строк в нужные типы
            for key, value in request.parameters.items():
                if key in ['no_components', 'epochs', 'num_threads', 'random_state']:
                    parameters[key] = int(value)
                elif key in ['item_alpha', 'user_alpha', 'learning_rate']:
                    parameters[key] = float(value)
                else:
                    parameters[key] = value
            
            self.logger.info(f"Запуск обучения модели с параметрами: {parameters}")
            
            model_id = await self.model_manager.train(parameters=parameters, context=context)
            
            response = recommendation_service_pb2.TrainModelResponse(
                model_id=model_id,
                success=True,
                message="Модель успешно обучена",
                version=self.model_manager.current_version
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {e}")
            return recommendation_service_pb2.TrainModelResponse(
                model_id="",
                success=False,
                message=f"Ошибка при обучении модели: {str(e)}",
                version=self.model_manager.current_version
            )
    
    async def GetModelInfo(self, request, context):
        """
        Получение информации о модели
        """
        try:
            model_id = request.model_id
            
            self.logger.info(f"Запрос информации о модели {model_id}")
            
            model_info = await self.model_manager.get_model_info(model_id, context=context)
            
            if not model_info:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Модель с ID {model_id} не найдена")
            
            # Преобразуем все значения в строки для передачи через gRPC
            parameters = {k: str(v) for k, v in model_info.get('parameters', {}).items()}
            metrics = {k: str(v) for k, v in model_info.get('metrics', {}).items()}
            
            response = recommendation_service_pb2.ModelInfoResponse(
                model_id=model_info.get('id', ''),
                name=model_info.get('name', ''),
                parameters=parameters,
                created_at=model_info.get('created_at', ''),
                is_active=model_info.get('is_active', False),
                metrics=metrics,
                file_path=model_info.get('file_path', '')
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении информации о модели: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Внутренняя ошибка: {str(e)}")
    
    async def ListModels(self, request, context):
        """
        Получение списка моделей
        """
        try:
            limit = request.limit or 20
            offset = request.offset or 0
            
            self.logger.info(f"Запрос списка моделей (limit={limit}, offset={offset})")
            
            models = await self.model_manager.list_models(limit=limit, offset=offset, context=context)
            
            # Создаем ответы для каждой модели
            model_responses = []
            for model_info in models:
                # Преобразуем все значения в строки для передачи через gRPC
                parameters = {k: str(v) for k, v in model_info.get('parameters', {}).items()}
                metrics = {k: str(v) for k, v in model_info.get('metrics', {}).items()}
                
                model_response = recommendation_service_pb2.ModelInfoResponse(
                    model_id=model_info.get('id', ''),
                    name=model_info.get('name', ''),
                    parameters=parameters,
                    created_at=model_info.get('created_at', ''),
                    is_active=model_info.get('is_active', False),
                    metrics=metrics,
                    file_path=model_info.get('file_path', '')
                )
                model_responses.append(model_response)
            
            response = recommendation_service_pb2.ListModelsResponse(
                models=model_responses,
                total_count=len(models)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка моделей: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Внутренняя ошибка: {str(e)}")
    
    async def SetActiveModel(self, request, context):
        """
        Установка активной модели
        """
        try:
            model_id = request.model_id
            
            self.logger.info(f"Запрос на установку активной модели {model_id}")
            
            success = await self.model_manager.set_active_model(model_id, context=context)
            
            response = recommendation_service_pb2.SetActiveModelResponse(
                success=success,
                message="Модель успешно установлена как активная" if success else "Ошибка установки активной модели"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке активной модели: {e}")
            return recommendation_service_pb2.SetActiveModelResponse(
                success=False,
                message=f"Ошибка: {str(e)}"
            )
    
    async def GetUserRecentInteractions(self, request, context):
        """
        Получение последних взаимодействий пользователя
        """
        try:
            user_id = request.user_id
            limit = request.limit or 10
            
            self.logger.info(f"Запрос последних взаимодействий пользователя {user_id}")
            
            interactions_df = await RecService.get_user_recent_interactions(
                user_id=user_id, 
                limit=limit, 
                context=context
            )
            
            # Преобразуем DataFrame в структуру для ответа
            interactions = []
            for _, row in interactions_df.iterrows():
                interaction = recommendation_service_pb2.UserInteraction(
                    item_id=int(row['title_id']),
                    interaction_type=row['interaction_type'],
                    timestamp=str(row['timestamp'])
                )
                interactions.append(interaction)
            
            response = recommendation_service_pb2.UserInteractionsResponse(
                interactions=interactions
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении взаимодействий пользователя: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Внутренняя ошибка: {str(e)}")


async def init_db():
    """Инициализирует соединение с базой данных"""
    engine = create_async_engine(DB_URL, echo=False)
    
    # Создаем фабрику асинхронных сессий
    async_session = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    return async_session


async def init_services():
    """Инициализирует сервисы и компоненты системы"""
    # Инициализируем базу данных
    session_maker = await init_db()
    
    # Инициализируем сервис аутентификации
    auth_service = AuthService()
    
    # Инициализируем сервис кеширования
    cache = RecommendationCache(url=CACHE_URL)
    await cache.initialize()
    
    # Инициализируем сервис подготовки данных
    data_preparer = DataPrepareService(session_maker=session_maker, read_only=True)
    
    # Инициализируем менеджер моделей
    model_manager = ModelManager(dp_service=data_preparer)
    await model_manager.initialize()
    
    # Запускаем планировщик задач
    RecService.start_scheduler()
    
    return session_maker


async def serve():
    """Запускает gRPC сервер"""
    # Инициализируем сервисы
    session_maker = await init_services()
    
    # Создаем сервер gRPC
    server = grpc_aio_server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    # Регистрируем сервис
    recommendation_service_pb2_grpc.add_RecommendationServiceServicer_to_server(
        RecommendationServicer(session_maker), 
        server
    )
    
    # Запускаем сервер
    server.add_insecure_port(f"{GRPC_HOST}:{GRPC_PORT}")
    await server.start()
    
    logger.info(f"Сервер gRPC запущен на {GRPC_HOST}:{GRPC_PORT}")
    
    # Держим сервер запущенным
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Завершение работы сервера...")
        await server.stop(0)


if __name__ == "__main__":
    # Запускаем сервер
    asyncio.run(serve()) 