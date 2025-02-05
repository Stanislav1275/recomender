import grpc
from grpc import aio as grpc_aio
from concurrent import futures
from recommendations.protos import recommendations_pb2_grpc
from recommendations.protos.recommendations_pb2 import RecommendationResponse, TrainResponse
from recommendations.rec_service import ModelManager


class RecommenderService(recommendations_pb2_grpc.RecommenderServicer):
    async def GetUserRecommendations(self, request, context):
        from recommendations.rec_service import RecService
        try:
            recos = await RecService.rec(request.user_id, context)
            return RecommendationResponse(item_ids=recos)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def TrainModel(self, request, context):
        from recommendations.rec_service import RecService
        try:
            await RecService.train(context)
            return TrainResponse(
                success=True,
                message="Training completed",
                version=ModelManager().current_version
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
