# main.py
import logging
import os
import sys
import warnings
from typing import Literal, List, Optional

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import Depends, HTTPException
import grpc
from concurrent import futures
import asyncio
import mongoengine
from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId

from rectools import Columns
from sqlalchemy import select
from sqlalchemy.orm import Session

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from external_db.models import Titles
from recommendations.data_preparer import get_db, map_ratings, DataPrepareService
from recommendations.protos.recommendations_pb2_grpc import add_RecommenderServicer_to_server
from recommendations.rec_server import RecommenderService
from recommendations.rec_service import ModelManager, RecService
from src.routes.admin_routes import router as admin_router
from internal.config.mongo_adapter import MongoAdapter
from internal.models import RecommendationItem, RecommendationResult

import uvicorn
from internal.app import create_app

# Инициализация MongoDB через MongoAdapter
mongo_config = MongoAdapter()
mongoengine.connect(
    db=mongo_config.db_name,
    host=mongo_config.host,
    port=mongo_config.port,
    username=mongo_config.user,
    password=mongo_config.password,
    alias='default'
)

app = create_app()

# gRPC server setup
grpc_server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=1))

app.include_router(admin_router)

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


@app.post('/api/train')
async def train():
    await RecService.train()


class CoverModel(BaseModel):
    high: Optional[str] = None
    low: Optional[str] = None
    mid: Optional[str] = None

class TitleResponse(BaseModel):
    id: int
    main_name: str
    dir: str
    status_id: int
    type_id: int
    age_limit: int
    count_chapters: int
    issue_year: int
    is_yaoi: int
    is_erotic: int
    is_legal: int
    is_licensed: int
    total_views: int
    total_votes: int
    avg_rating: float = Field(ge=0, le=10)
    cover: Optional[CoverModel] = None
    upload_date: Optional[datetime] = None
    last_chapter_uploaded: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            ObjectId: str
        }

@app.get("/api/titles/recommendations", response_model=List[TitleResponse])
async def rec_by_users(user_id: int, db: Session = Depends(get_db)) -> List[TitleResponse]:
    try:
        model, dataset = await ModelManager().get_model()
        recos = model.recommend(users=[user_id], dataset=dataset, k=10, filter_viewed=True)
        item_ids = recos['item_id'].tolist()
        stmt = select(Titles).where(Titles.id.in_(item_ids))

        order = {id_: idx for idx, id_ in enumerate(item_ids)}
        result = db.execute(stmt).scalars().all()

        sorted_result = sorted(result, key=lambda x: order.get(x.id, len(item_ids)))

        await DataPrepareService._get_user_buys({})
        return sorted_result

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/titles/relavant", response_model=List[TitleResponse])
async def rec_by_title(title_id: int, db: Session = Depends(get_db)) -> List[TitleResponse]:
    try:
        model, dataset = await ModelManager().get_model()
        recos = model.recommend_to_items(target_items=[title_id], dataset=dataset, k=40, filter_itself=False)
        item_ids = recos['item_id'].tolist()

        stmt = select(Titles).where(Titles.id.in_(item_ids))
        order = {id_: idx for idx, id_ in enumerate(item_ids)}
        result = db.execute(stmt).scalars().all()

        sorted_result = sorted(result, key=lambda x: order.get(x.id, len(item_ids)))
        return sorted_result

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "grpc_server": "running",
        "model_version": ModelManager().current_version
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
