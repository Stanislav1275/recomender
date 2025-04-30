#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Основной модуль приложения рекомендательного сервиса.
Запускает FastAPI приложение и gRPC сервер.
"""

import logging
import warnings
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request
import grpc
from concurrent import futures
import asyncio

from rectools import Columns
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# Отключаем предупреждения, которые могут появляться при работе с pandas и rectools
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорты из локальных модулей
from core.database import get_db_session
from models import Titles
from recommendations.data_preparer import map_ratings, DataPrepareService
from recommendations.protos.recommendations_pb2_grpc import add_RecommenderServicer_to_server
from recommendations.rec_server import RecommenderService
from recommendations.rec_service import ModelManager, RecService
from recommendations.config import API_CONFIG, GRPC_CONFIG

# Создание FastAPI приложения
app = FastAPI(
    title="Рекомендательный сервис API",
    description="API для получения рекомендаций на основе RecTools",
    version="1.0.0",
)

# Добавление CORS middleware для поддержки кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3001", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Настройка gRPC сервера
grpc_server = grpc.aio.server(
    futures.ThreadPoolExecutor(max_workers=GRPC_CONFIG["max_workers"])
)


@app.on_event("startup")
async def startup_event():
    """
    Инициализация всех компонентов при запуске приложения.
    """
    logger.info("Инициализация ModelManager...")
    manager = ModelManager()
    await manager.initialize()

    logger.info("Запуск планировщика...")
    RecService.start_scheduler()

    logger.info("Настройка gRPC сервера...")
    add_RecommenderServicer_to_server(RecommenderService(), grpc_server)
    grpc_server.add_insecure_port(f'{GRPC_CONFIG["host"]}:{GRPC_CONFIG["port"]}')
    await grpc_server.start()

    # Задаем event_loop для планировщика если он существует
    if RecService._scheduler:
        RecService._scheduler._eventloop = asyncio.get_running_loop()

    logger.info(f"gRPC сервер запущен на порту {GRPC_CONFIG['port']}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Корректное завершение всех компонентов при остановке приложения.
    """
    logger.info("Останавливаем gRPC сервер...")
    await grpc_server.stop(grace=5)
    
    logger.info("Останавливаем планировщик...")
    if RecService._scheduler:
        RecService._scheduler.shutdown()
    
    logger.info("Все сервисы остановлены")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware для логирования всех HTTP запросов.
    """
    logger.debug(f"Запрос {request.method} {request.url}")
    response = await call_next(request)
    logger.debug(f"Ответ {request.method} {request.url} - {response.status_code}")
    return response


@app.get('/train')
async def train():
    """
    Эндпоинт для ручного запуска обучения модели.
    """
    logger.info("Запущено ручное обучение модели")
    result = await RecService.train()
    if result:
        return {"status": "success", "message": "Модель успешно обучена"}
    else:
        raise HTTPException(status_code=500, detail="Ошибка обучения модели")


@app.get("/titles/recommendations", response_model=List[dict])
async def get_user_recommendations(user_id: int, db: Session = Depends(get_db_session)):
    """
    Получение персональных рекомендаций для пользователя.
    
    Args:
        user_id: ID пользователя
        db: Сессия БД
        
    Returns:
        List[dict]: Список рекомендованных произведений
        
    Raises:
        HTTPException: При ошибке получения рекомендаций
    """
    try:
        # Получаем модель и датасет
        model, dataset = await ModelManager().get_model()
        
        # Получаем рекомендации
        recos = model.recommend(
            users=[user_id], 
            dataset=dataset, 
            k=40, 
            filter_viewed=True
        )
        
        # Получаем ID рекомендованных произведений
        item_ids = recos['item_id'].tolist()
        
        # Формируем запрос к БД
        stmt = select(Titles).where(Titles.id.in_(item_ids))
        
        # Создаем словарь для сортировки результатов в том же порядке, что и рекомендации
        order = {id_: idx for idx, id_ in enumerate(item_ids)}
        
        # Выполняем запрос
        result = db.execute(stmt).scalars().all()
        
        # Сортируем результаты в порядке рекомендаций
        sorted_result = sorted(result, key=lambda x: order.get(x.id, len(item_ids)))
        
        return sorted_result

    except Exception as e:
        logger.error(f"Ошибка получения рекомендаций для пользователя {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/titles/relavant", response_model=List[dict])
async def get_similar_titles(title_id: int, db: Session = Depends(get_db_session)):
    """
    Получение похожих произведений.
    
    Args:
        title_id: ID произведения
        db: Сессия БД
        
    Returns:
        List[dict]: Список похожих произведений
        
    Raises:
        HTTPException: При ошибке получения рекомендаций
    """
    try:
        # Получаем модель и датасет
        model, dataset = await ModelManager().get_model()
        
        # Получаем рекомендации
        recos = model.recommend_to_items(
            target_items=[title_id], 
            dataset=dataset, 
            k=40, 
            filter_itself=False
        )
        
        # Получаем ID рекомендованных произведений
        item_ids = recos['item_id'].tolist()

        # Формируем запрос к БД
        stmt = select(Titles).where(Titles.id.in_(item_ids))
        
        # Создаем словарь для сортировки результатов в том же порядке, что и рекомендации
        order = {id_: idx for idx, id_ in enumerate(item_ids)}
        
        # Выполняем запрос
        result = db.execute(stmt).scalars().all()
        
        # Сортируем результаты в порядке рекомендаций
        sorted_result = sorted(result, key=lambda x: order.get(x.id, len(item_ids)))

        return sorted_result

    except Exception as e:
        logger.error(f"Ошибка получения похожих произведений для ID {title_id}: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/rec/hot/interact")
async def hot_update_interact(
    user_id: int, 
    title_id: int, 
    int_type: str,  # 'rating' или 'view'
    raw_score: Optional[int] = None,
    db: Session = Depends(get_db_session)
):
    """
    Эндпоинт для горячего обновления взаимодействий пользователя с произведениями.
    
    Args:
        user_id: ID пользователя
        title_id: ID произведения
        int_type: Тип взаимодействия ('rating' или 'view')
        raw_score: Оценка (для rating) или значение просмотра (для view)
        db: Сессия БД
        
    Returns:
        dict: Результат операции
        
    Raises:
        HTTPException: При ошибке обработки запроса
    """
    try:
        # Проверка корректности типа взаимодействия
        if int_type not in ['rating', 'view']:
            raise ValueError("Неверный тип взаимодействия. Допустимые значения: 'rating', 'view'")
        
        # Формируем DataFrame с новым взаимодействием
        weight = map_ratings(raw_score) if int_type == 'rating' else raw_score
        
        interact_pd = pd.DataFrame({
            Columns.User: [user_id],
            Columns.Item: [title_id],
            Columns.Weight: [weight],
            Columns.Datetime: [pd.Timestamp.now()],
        })
        
        logger.info(f"Получено новое взаимодействие: {interact_pd}")
        
        # Пустой DataFrame для признаков пользователя (в данном случае не используется)
        user_feature_pd = pd.DataFrame(columns=['id', 'feature', 'value'])
        
        # Выполняем частичное дообучение модели
        result = await ModelManager().fit_partial(
            new_interactions=interact_pd, 
            new_user_features=user_feature_pd
        )
        
        if result:
            return {"status": "success", "message": "Взаимодействие успешно обработано"}
        else:
            raise HTTPException(
                status_code=500, 
                detail="Ошибка обработки взаимодействия"
            )
            
    except ValueError as ve:
        logger.error(f"Ошибка валидации: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        logger.error(f"Ошибка обработки взаимодействия: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.get("/health")
async def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    """
    model_version = 0
    try:
        model_version = ModelManager().current_version
    except Exception:
        pass
        
    return {
        "status": "ok",
        "grpc_server": "running",
        "model_version": model_version
    }


if __name__ == "__main__":
    import uvicorn
    
    # Запуск FastAPI приложения
    uvicorn.run(
        app, 
        host=API_CONFIG["host"], 
        port=API_CONFIG["port"],
        log_level="info"
    )
