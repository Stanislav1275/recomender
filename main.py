# main.py
import logging
import warnings
from typing import Literal, List, Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Query
import grpc
from concurrent import futures
import asyncio

from rectools import Columns
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from core.database import SessionLocal
from models import Titles
from recommendations.data_preparer import get_db, map_ratings, DataPrepareService
from recommendations.protos.recommendations_pb2_grpc import add_RecommenderServicer_to_server
from recommendations.rec_server import RecommenderService
from recommendations.rec_service import ModelManager, RecService
from recommendations.cache_service import RecommendationCache
from recommendations.model_registry import ModelRegistry

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3001", "http://localhost:3000"],  # Разрешить все домены
    allow_methods=["*"],  # Разрешить все методы (GET, POST и т.д.)
    allow_headers=["*"]
)
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

    # Инициализируем сервис кеширования
    cache = RecommendationCache()
    await cache.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    await grpc_server.stop(grace=5)
    if RecService._scheduler:
        RecService._scheduler.shutdown()
        
    # Закрываем соединение с Redis
    cache = RecommendationCache()
    await cache.close()
        
    print("Services stopped")


@app.get('/admin/train', tags=["Admin"])
async def train_model(
    no_components: int = Query(100, description="Размерность векторов представления (факторов)"),
    loss: str = Query("bpr", description="Функция потерь: bpr, warp, logistic, warp-kos"),
    epochs: int = Query(30, description="Количество эпох обучения"),
    item_alpha: float = Query(0.0, description="L2 регуляризация для элементов"),
    user_alpha: float = Query(0.0, description="L2 регуляризация для пользователей")
):
    """
    Запускает обучение новой модели с указанными параметрами.
    """
    try:
        # Проверяем параметры
        if loss not in ["bpr", "warp", "logistic", "warp-kos"]:
            raise HTTPException(status_code=400, detail="Неверное значение параметра loss")
            
        parameters = {
            'no_components': no_components,
            'loss': loss,
            'epochs': epochs,
            'item_alpha': item_alpha,
            'user_alpha': user_alpha
        }
        
        model_id = await ModelManager().train(parameters)
        
        return {
            "status": "success",
            "message": "Модель успешно обучена",
            "model_id": model_id,
            "version": ModelManager().current_version
        }
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обучения модели: {str(e)}")


@app.get('/admin/schedule-training', tags=["Admin"])
async def schedule_training(
    scheduled_at: Optional[datetime] = None,
    no_components: int = Query(100, description="Размерность векторов представления (факторов)"),
    loss: str = Query("bpr", description="Функция потерь: bpr, warp, logistic, warp-kos"),
    epochs: int = Query(30, description="Количество эпох обучения"),
    item_alpha: float = Query(0.0, description="L2 регуляризация для элементов"),
    user_alpha: float = Query(0.0, description="L2 регуляризация для пользователей")
):
    """
    Планирует обучение новой модели на указанное время.
    Если время не указано, берется текущее время + 5 минут.
    """
    try:
        # Проверяем параметры
        if loss not in ["bpr", "warp", "logistic", "warp-kos"]:
            raise HTTPException(status_code=400, detail="Неверное значение параметра loss")
            
        parameters = {
            'no_components': no_components,
            'loss': loss,
            'epochs': epochs,
            'item_alpha': item_alpha,
            'user_alpha': user_alpha
        }
        
        # Если время не указано, планируем на текущее время + 5 минут
        if scheduled_at is None:
            scheduled_at = datetime.now() + timedelta(minutes=5)
        
        job_id = await ModelManager().schedule_training(parameters, scheduled_at)
        
        return {
            "status": "success",
            "message": f"Обучение запланировано на {scheduled_at.isoformat()}",
            "job_id": job_id,
            "scheduled_at": scheduled_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Ошибка планирования обучения: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка планирования обучения: {str(e)}")


@app.get('/admin/models', tags=["Admin"])
async def list_models(limit: int = 20, offset: int = 0):
    """
    Получает список всех моделей из реестра.
    """
    try:
        models = await ModelManager().list_models(limit, offset)
        return {
            "total": len(models),
            "models": models
        }
    except Exception as e:
        logger.error(f"Ошибка получения списка моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка моделей: {str(e)}")


@app.get('/admin/models/{model_id}', tags=["Admin"])
async def get_model_info(model_id: str):
    """
    Получает информацию о конкретной модели.
    """
    try:
        model_info = await ModelManager().get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Модель с ID {model_id} не найдена")
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения информации о модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации о модели: {str(e)}")


@app.post('/admin/models/{model_id}/activate', tags=["Admin"])
async def activate_model(model_id: str):
    """
    Устанавливает указанную модель как активную.
    """
    try:
        success = await ModelManager().set_active_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Модель с ID {model_id} не найдена или не может быть активирована")
        
        return {
            "status": "success",
            "message": f"Модель {model_id} успешно активирована",
            "current_version": ModelManager().current_version
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка активации модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка активации модели: {str(e)}")


@app.get('/admin/jobs', tags=["Admin"])
async def list_training_jobs(status: Optional[str] = None, limit: int = 20, offset: int = 0):
    """
    Получает список заданий на обучение.
    Можно фильтровать по статусу (scheduled, running, completed, failed).
    """
    try:
        registry = ModelRegistry()
        jobs = registry.list_training_jobs(status, limit, offset)
        
        return {
            "total": len(jobs),
            "jobs": jobs
        }
    except Exception as e:
        logger.error(f"Ошибка получения списка заданий: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка заданий: {str(e)}")


@app.get('/admin/jobs/{job_id}', tags=["Admin"])
async def get_training_job(job_id: str):
    """
    Получает информацию о конкретном задании на обучение.
    """
    try:
        registry = ModelRegistry()
        job_info = registry.get_training_job(job_id)
        
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Задание с ID {job_id} не найдено")
            
        return job_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения информации о задании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации о задании: {str(e)}")


@app.post('/admin/cache/invalidate', tags=["Admin"])
async def invalidate_cache():
    """
    Инвалидирует весь кеш рекомендаций.
    """
    try:
        cache = RecommendationCache()
        success = await cache.invalidate_all_recommendations()
        
        if not success:
            raise HTTPException(status_code=500, detail="Ошибка инвалидации кеша")
            
        return {
            "status": "success",
            "message": "Кеш рекомендаций успешно инвалидирован"
        }
    except Exception as e:
        logger.error(f"Ошибка инвалидации кеша: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка инвалидации кеша: {str(e)}")


@app.get("/titles/recommendations")
async def rec_by_users(user_id: int, db: Session = Depends(get_db)):
    """
    Получает персональные рекомендации для пользователя и возвращает соответствующие тайтлы.
    
    Args:
        user_id: ID пользователя
        db: Сессия базы данных
    
    Returns:
        List[Titles]: Список рекомендованных тайтлов
    """
    try:
        # Получаем рекомендации (метод RecService.rec уже использует кеширование)
        item_ids = await RecService.rec(user_id=user_id)
        
        # Если нет рекомендаций, возвращаем пустой список
        if not item_ids:
            return []
            
        # Запрашиваем информацию о тайтлах из базы данных
        titles_info = await _fetch_titles_by_ids(item_ids, db)
        
        return titles_info

    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.get("/titles/relavant")
async def rec_by_title(title_id: int, db: Session = Depends(get_db)):
    """
    Получает список похожих тайтлов для указанного тайтла.
    
    Args:
        title_id: ID тайтла
        db: Сессия базы данных
    
    Returns:
        List[Titles]: Список похожих тайтлов
    """
    try:
        # Получаем рекомендации (метод RecService.relevant уже использует кеширование)
        item_ids = await RecService.relevant(item_id=title_id)
        
        # Если нет рекомендаций, возвращаем пустой список
        if not item_ids:
            return []
            
        # Запрашиваем информацию о тайтлах из базы данных
        titles_info = await _fetch_titles_by_ids(item_ids, db)
        
        return titles_info

    except Exception as e:
        logger.error(f"Ошибка при получении похожих тайтлов: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


async def _fetch_titles_by_ids(item_ids, db):
    """
    Вспомогательная функция для получения информации о тайтлах по их ID.
    
    Args:
        item_ids: Список ID тайтлов
        db: Сессия базы данных
    
    Returns:
        List[Titles]: Список тайтлов, отсортированный в соответствии с порядком ID
    """
    # Создаем запрос на получение тайтлов
    stmt = select(Titles).where(Titles.id.in_(item_ids))
    
    # Создаем словарь для сохранения порядка ID
    order = {id_: idx for idx, id_ in enumerate(item_ids)}
    
    # Выполняем запрос
    result = db.execute(stmt).scalars().all()
    
    # Сортируем результат в соответствии с порядком ID
    sorted_result = sorted(result, key=lambda x: order.get(x.id, len(item_ids)))
    
    return sorted_result


@app.get("/rec/hot/interact")
async def hot_update_interact(user_id: int, title_id: int, int_type: Literal['rating', 'view'], raw_score: int = None,
                              db: Session = Depends(get_db)):
    """
    Обновляет взаимодействие пользователя с контентом и обучает модель на новых данных.
    
    Args:
        user_id: ID пользователя
        title_id: ID контента
        int_type: Тип взаимодействия ('rating' или 'view')
        raw_score: Числовое значение взаимодействия (оценка или время просмотра)
        db: Сессия базы данных
    """
    if raw_score is None and int_type == 'rating':
        raise HTTPException(status_code=400, detail="Rating requires a raw_score value")
        
    try:
        # Создаем DataFrame с новым взаимодействием
        interact_pd = pd.DataFrame({
            Columns.User: [user_id],
            Columns.Item: [title_id],
            Columns.Weight: [map_ratings(raw_score) if int_type == 'rating' else raw_score or 1],
            Columns.Datetime: [pd.Timestamp.now()],
        })
        
        # Пустой DataFrame пользовательских характеристик (их не нужно обновлять)
        user_features_pd = pd.DataFrame(columns=['id', 'feature', 'value'])
        
        # Частичное дообучение модели
        await ModelManager().fit_partial(new_interactions=interact_pd, new_user_features=user_features_pd)
        
        # Инвалидируем кеш рекомендаций для этого пользователя
        cache = RecommendationCache()
        await cache.cache_user_recommendations(user_id, [])  # Удаляем кеш пользователя
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Interaction of type '{int_type}' processed and model updated",
                "user_id": user_id,
                "title_id": title_id,
                "model_version": ModelManager().current_version
            }
        )
    except Exception as exc:
        logger.error("%s", exc, extra={"rich": True})
        logger.debug("Exception information:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update interaction: {str(exc)}")


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса.
    """
    # Проверяем состояние модели
    model_ok = await _check_model_health()
    
    # Проверяем состояние Redis
    cache_ok = await _check_cache_health()
    
    # Получаем информацию об активной модели
    registry = ModelRegistry()
    active_model = registry.get_active_model()
    model_info = {
        "version": ModelManager().current_version,
        "id": active_model.get("id", "unknown"),
        "created_at": active_model.get("created_at", "unknown")
    }
    
    return {
        "status": "ok" if model_ok and cache_ok else "degraded",
        "components": {
            "grpc_server": "running",
            "model": "ok" if model_ok else "error",
            "cache": "ok" if cache_ok else "error"
        },
        "model": model_info
    }


async def _check_model_health() -> bool:
    """Проверяет состояние модели"""
    try:
        model, _ = await ModelManager().get_model()
        return model.is_fitted
    except Exception:
        return False


async def _check_cache_health() -> bool:
    """Проверяет состояние кеша"""
    try:
        cache = RecommendationCache()
        if cache._redis is None:
            await cache.initialize()
        return cache._redis is not None
    except Exception:
        return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
