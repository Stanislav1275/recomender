import logging
from concurrent import futures

import grpc
from recommendations.protos.recommendations_pb2_grpc import add_RecommenderServicer_to_server
from recommendations.rec_server import RecommenderService
from recommendations.rec_service import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def serve():
    # Инициализация модели
    manager = ModelManager()
    await manager.initialize()

    # Настройка gRPC сервера
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RecommenderServicer_to_server(RecommenderService(), server)

    # Прослушивание порта
    server.add_insecure_port('[::]:50051')
    await server.start()
    logger.info("gRPC server started on port 50051")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(5)


if __name__ == '__main__':
    import asyncio

    asyncio.run(serve())