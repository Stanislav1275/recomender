from recomendations.protos import recommendations_pb2
from recomendations.protos.recommendations_pb2 import RecommendationResponse
from recomendations.protos.recommendations_pb2_grpc import RecommenderServicer
from recomendations.rec_service import RecService


class RecommenderServer(RecommenderServicer):
    async def GetUserRecommendations(self, request, context):
        recos = await RecService.rec(request.user_id, context)
        return RecommendationResponse(item_ids=recos.item_ids)

    async def GetRelevantItems(self, request, context):
        recos = await RecService.relevant_2_item(request.user_id, context)
        return RecommendationResponse(item_ids=recos)

    async def Train(self, request, context):
        await RecService.train(context)
        # return recommendations_pb2.RecommendationResponse(item_ids=item_ids)

    async def TrainModel(self, request, context):
        # Пример реализации
        return recommendations_pb2.TrainResponse(
            success=True,
            message="Model trained successfully",
            version=1
        )
