import asyncio

import grpc
from  recommendations.rec_service import RecService, ModelManager

from recommendations.protos.recommendations_pb2 import RecommendationResponse, TrainResponse
from recommendations.protos.recommendations_pb2_grpc import RecommenderServicer
class RecommenderService(RecommenderServicer):
    async def GetUserRecommendations(self, request, context):
        recos = await RecService.rec(request.user_id, context)
        return RecommendationResponse(item_ids=recos)

    async def GetTitleRelevant(self, request, context):
        recos = await RecService.relevant_2_item(request.title_id, context)
        return RecommendationResponse(item_ids=recos)

    async def TrainModel(self, request, context):
        await RecService.train(context)
        return TrainResponse(
            success=True,
            message="Training initiated",
            version=ModelManager().current_version
        )


def serve():
    from concurrent import futures
    from recommendations.protos.recommendations_pb2_grpc import add_RecommenderServicer_to_server

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RecommenderServicer_to_server(RecommenderService(), server)
    server.add_insecure_port('[::]:50051')
    return server


async def main():
    # Инициализация модели
    ModelManager()

    # Запуск планировщика
    RecService.start_scheduler()

    # Запуск сервера
    server = serve()
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(main())
