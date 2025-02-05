# main.py
from fastapi import FastAPI
import grpc
from concurrent import futures
import asyncio
from recommendations.protos.recommendations_pb2_grpc import add_RecommenderServicer_to_server
from recommendations.rec_server import RecommenderService
from recommendations.rec_service import ModelManager, RecService

app = FastAPI()

# gRPC server setup
grpc_server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=1))


@app.on_event("startup")
async def startup_event():
    manager = ModelManager()
    await manager.initialize()

    RecService.start_scheduler()

    add_RecommenderServicer_to_server(RecommenderService(), grpc_server)
    grpc_server.add_insecure_port('[::]:50051')
    await grpc_server.start()

    if RecService._scheduler:
        RecService._scheduler._eventloop = asyncio.get_running_loop()

    print("gRPC server started on port 50051")


@app.on_event("shutdown")
async def shutdown_event():
    await grpc_server.stop(grace=5)
    if RecService._scheduler:
        RecService._scheduler.shutdown()
    print("Services stopped")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "grpc_server": "running",
        "model_version": ModelManager().current_version
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)