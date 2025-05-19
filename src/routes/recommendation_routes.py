from fastapi import APIRouter, Depends, HTTPException
from typing import List

from internal.types import (
    RecommendationRequest,
    RecommendationResult,
    TrainRequest,
    TrainResponse,
    ErrorResponse
)
from internal.services import RecommendationService
from internal.utils.error_handler import handle_error
from internal.models import RecommendationItem, RecommendationResult as RecommendationResultModel


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


async def get_recommendation_service() -> RecommendationService:
    """Получить сервис рекомендаций"""
    return RecommendationService()


@router.get(
    "/user/{user_id}",
    response_model=RecommendationResultModel,
    responses={
        200: {
            "description": "Успешное получение рекомендаций",
            "content": {
                "application/json": {
                    "schema": RecommendationResultModel.model_json_schema(),
                    "example": {
                        "items": [
                            {
                                "id": 12345,
                                "main_name": "Название тайтла",
                                "dir": "title-name",
                                "score": 0.95,
                                "status_id": 1,
                                "type_id": 1,
                                "age_limit": 0,
                                "count_chapters": 100,
                                "issue_year": 2023,
                                "is_yaoi": 0,
                                "is_erotic": 0,
                                "is_legal": 1,
                                "is_licensed": 1,
                                "total_views": 10000,
                                "total_votes": 1000,
                                "avg_rating": 8.5,
                                "cover": {
                                    "high": "https://example.com/cover.jpg",
                                    "low": "https://example.com/cover.jpg",
                                    "mid": "https://example.com/cover.jpg",
                                },
                                "upload_date": "2024-02-14T04:00:00Z",
                                "last_chapter_uploaded": "2024-02-14T04:00:00Z"
                            }
                        ],
                        "total": 1,
                        "page": 1,
                        "page_size": 10
                    }
                }
            }
        },
        422: {
            "description": "Ошибка валидации",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "loc": {
                                            "type": "array",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"}
                                                ]
                                            }
                                        },
                                        "msg": {"type": "string"},
                                        "type": {"type": "string"}
                                    },
                                    "required": ["loc", "msg", "type"]
                                }
                            }
                        },
                        "required": ["detail"]
                    },
                    "example": {
                        "detail": [
                            {
                                "loc": ["path", "user_id"],
                                "msg": "value is not a valid integer",
                                "type": "type_error.integer"
                            }
                        ]
                    }
                }
            }
        },
        404: {
            "description": "Пользователь не найден",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "detail": {"type": "string"}
                        }
                    },
                    "example": {
                        "error": "Пользователь не найден",
                        "detail": "User with id 123 not found"
                    }
                }
            }
        },
        500: {
            "description": "Внутренняя ошибка сервера",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "detail": {"type": "string"}
                        }
                    },
                    "example": {
                        "error": "Внутренняя ошибка сервера",
                        "detail": "Failed to get recommendations"
                    }
                }
            }
        }
    },
    summary="Получить рекомендации для пользователя",
    description="Возвращает список рекомендованных тайтлов для указанного пользователя"
)
async def get_user_recommendations(
    user_id: int,
    limit: int = 10,
    service: RecommendationService = Depends(get_recommendation_service)
) -> RecommendationResultModel:
    """Получить рекомендации для пользователя"""
    try:
        request = RecommendationRequest(user_id=user_id, limit=limit)
        return await service.get_user_recommendations(request)
    except Exception as e:
        raise handle_error(e)


@router.get(
    "/title/{title_id}/relevant",
    response_model=RecommendationResultModel,
    responses={
        200: {
            "description": "Успешное получение похожих тайтлов",
            "content": {
                "application/json": {
                    "schema": RecommendationResultModel.model_json_schema(),
                    "example": {
                        "items": [
                            {
                                "id": 12345,
                                "main_name": "Название тайтла",
                                "dir": "title-name",
                                "score": 0.95,
                                "status_id": 1,
                                "type_id": 1,
                                "age_limit": 0,
                                "count_chapters": 100,
                                "issue_year": 2023,
                                "is_yaoi": 0,
                                "is_erotic": 0,
                                "is_legal": 1,
                                "is_licensed": 1,
                                "total_views": 10000,
                                "total_votes": 1000,
                                "avg_rating": 8.5,
                                "cover": {
                                    "url": "https://example.com/cover.jpg",
                                    "low": "https://example.com/cover.jpg",
                                    "mid": "https://example.com/cover.jpg",
                                },
                                "upload_date": "2024-02-14T04:00:00Z",
                                "last_chapter_uploaded": "2024-02-14T04:00:00Z"
                            }
                        ],
                        "total": 1,
                        "page": 1,
                        "page_size": 10
                    }
                }
            }
        },
        422: {
            "description": "Ошибка валидации",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "loc": {
                                            "type": "array",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"}
                                                ]
                                            }
                                        },
                                        "msg": {"type": "string"},
                                        "type": {"type": "string"}
                                    },
                                    "required": ["loc", "msg", "type"]
                                }
                            }
                        },
                        "required": ["detail"]
                    },
                    "example": {
                        "detail": [
                            {
                                "loc": ["path", "title_id"],
                                "msg": "value is not a valid integer",
                                "type": "type_error.integer"
                            }
                        ]
                    }
                }
            }
        },
        404: {
            "description": "Тайтл не найден",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "detail": {"type": "string"}
                        }
                    },
                    "example": {
                        "error": "Тайтл не найден",
                        "detail": "Title with id 123 not found"
                    }
                }
            }
        },
        500: {
            "description": "Внутренняя ошибка сервера",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "detail": {"type": "string"}
                        }
                    },
                    "example": {
                        "error": "Внутренняя ошибка сервера",
                        "detail": "Failed to get relevant titles"
                    }
                }
            }
        }
    },
    summary="Получить похожие тайтлы",
    description="Возвращает список тайтлов, похожих на указанный"
)
async def get_title_relevant(
    title_id: int,
    limit: int = 10,
    service: RecommendationService = Depends(get_recommendation_service)
) -> RecommendationResultModel:
    """Получить похожие тайтлы"""
    try:
        request = RecommendationRequest(title_id=title_id, limit=limit)
        return await service.get_title_relevant(request)
    except Exception as e:
        raise handle_error(e)


@router.post(
    "/train",
    response_model=TrainResponse,
    responses={
        200: {
            "description": "Успешный ответ",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Model trained successfully",
                        "version": 1,
                        "metrics": {
                            "precision": 0.85,
                            "recall": 0.82,
                            "ndcg": 0.91
                        }
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Внутренняя ошибка сервера",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                        "details": None
                    }
                }
            }
        }
    },
    summary="Обучить модель рекомендаций",
    description="Запускает процесс обучения модели рекомендаций"
)
async def train_model(
    request: TrainRequest,
    service: RecommendationService = Depends(get_recommendation_service)
) -> TrainResponse:
    """Обучить модель рекомендаций"""
    try:
        return await service.train_model(request)
    except Exception as e:
        raise handle_error(e) 