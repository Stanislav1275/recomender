"""
gRPC сервер для рекомендательной системы.
Предоставляет методы для получения рекомендаций и управления моделью.
"""
import logging
import grpc
from grpc import aio as grpc_aio
from concurrent import futures
from recommendations.protos import recommendations_pb2_grpc
from recommendations.protos.recommendations_pb2 import RecommendationResponse, TrainResponse
from recommendations.rec_service import ModelManager, RecService

logger = logging.getLogger(__name__)


class RecommenderService(recommendations_pb2_grpc.RecommenderServicer):
    """
    Реализация gRPC сервиса рекомендаций.
    Предоставляет методы для получения рекомендаций и управления моделью.
    """
    
    async def GetUserRecommendations(self, request, context):
        """
        Получение персональных рекомендаций для пользователя.
        
        Args:
            request: gRPC запрос с user_id
            context: gRPC контекст
            
        Returns:
            RecommendationResponse: Ответ с рекомендациями
        """
        try:
            logger.info(f"Получен запрос рекомендаций для пользователя {request.user_id}")
            recos = await RecService.rec(request.user_id, context)
            return RecommendationResponse(item_ids=recos)
        except Exception as e:
            logger.error(f"Ошибка получения рекомендаций: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            
    async def GetTitleRelevant(self, request, context):
        """
        Получение похожих произведений.
        
        Args:
            request: gRPC запрос с title_id
            context: gRPC контекст
            
        Returns:
            RecommendationResponse: Ответ с рекомендациями
        """
        try:
            logger.info(f"Получен запрос похожих произведений для ID {request.title_id}")
            recos = await RecService.relevant(request.title_id, context)
            return RecommendationResponse(item_ids=recos)
        except Exception as e:
            logger.error(f"Ошибка получения похожих произведений: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def TrainModel(self, request, context):
        """
        Ручной запуск обучения модели.
        
        Args:
            request: gRPC запрос
            context: gRPC контекст
            
        Returns:
            TrainResponse: Ответ с результатом обучения
        """
        try:
            logger.info("Получен запрос на обучение модели")
            result = await RecService.train(context)
            return TrainResponse(
                success=result,
                message="Обучение завершено успешно" if result else "Ошибка обучения",
                version=ModelManager().current_version
            )
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


async def serve(host="[::]:50051", max_workers=10):
    """
    Запуск gRPC сервера.
    
    Args:
        host: Хост и порт для прослушивания
        max_workers: Максимальное количество рабочих потоков
    """
    # Инициализация модели
    manager = ModelManager()
    await manager.initialize()

    # Настройка gRPC сервера
    server = grpc_aio.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    recommendations_pb2_grpc.add_RecommenderServicer_to_server(RecommenderService(), server)

    # Прослушивание порта
    server.add_insecure_port(host)
    await server.start()
    logger.info(f"gRPC сервер запущен на {host}")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки сервера")
        await server.stop(5)
    except Exception as e:
        logger.error(f"Ошибка сервера: {e}")
        await server.stop(0)


if __name__ == '__main__':
    import asyncio
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск сервера
    asyncio.run(serve())
