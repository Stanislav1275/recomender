from datetime import datetime
from typing import List, Optional, Dict, Union, Literal, TypedDict
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from .models import (
    RecommendationConfig,
    RecommendationItem,
    RecommendationResult
)
from .constants import FIELD_OPTIONS
from external_db.data_service import ExternalDataService


class BaseEntity(BaseModel):
    """Базовая структура сущности"""
    id: Optional[str] = Field(None, description="Идентификатор сущности")
    created_at: Optional[datetime] = Field(None, description="Дата создания")
    updated_at: Optional[datetime] = Field(None, description="Дата обновления")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "created_at": "2024-02-14T04:00:00Z",
                "updated_at": "2024-02-14T04:00:00Z"
            }
        }
    )


class ScheduleType(BaseModel):
    """Тип расписания"""
    id: str = Field(..., description="Идентификатор типа расписания")
    name: str = Field(..., description="Название типа расписания")
    description: str = Field(..., description="Описание типа расписания")
    value: str = Field(..., description="Значение типа расписания")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "once_day",
                "name": "Ежедневно",
                "description": "Ежедневно в 00:00",
                "value": "once_day"
            }
        }
    )


class FieldValue(TypedDict):
    """Значение поля"""
    value: Union[str, int, float, bool]
    name: str


class FieldMetadata(BaseModel):
    """Метаданные поля"""
    name: str = Field(..., description="Имя поля")
    description: str = Field(..., description="Описание поля")
    type: Literal["boolean", "integer", "float", "reference"] = Field(..., description="Тип данных поля")
    operators: List[str] = Field(..., description="Доступные операторы")
    values: Optional[List[FieldValue]] = Field(None, description="Доступные значения для поля")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "is_erotic",
                "description": "Эротический контент",
                "type": "boolean",
                "operators": ["equals", "not_equals"],
                "values": [
                    {"value": 0, "name": "Нет"},
                    {"value": 1, "name": "Да"}
                ]
            }
        }
    )


class FieldOptions(BaseModel):
    """Опции полей для админ-панели"""
    title_fields: List[FieldMetadata] = Field(..., description="Метаданные полей для фильтрации")
    schedule_types: List[ScheduleType] = Field(..., description="Доступные типы расписаний")

    @classmethod
    async def from_constants(cls, external_service: ExternalDataService) -> "FieldOptions":
        """Создать опции полей из констант и внешнего сервиса"""
        title_fields = []
        for field_name, field_data in FIELD_OPTIONS.items():
            values = None
            if field_data["type"] == "reference":
                # Получаем значения из внешнего сервиса для справочных полей
                values = await external_service.get_field_metadata(field_name)
            elif field_data["type"] == "boolean":
                # Значения для булевых полей
                values = [
                    {"value": False, "name": "Нет"},
                    {"value": True, "name": "Да"}
                ]
            elif field_name == "age_limit":
                # Значения для возрастных ограничений
                values = [
                    {"value": 0, "name": "0+"},
                    {"value": 1, "name": "12+"},
                    {"value": 2, "name": "18+"}
                ]
            elif field_name == "issue_year":
                # Значения для годов выпуска (последние 50 лет)
                current_year = datetime.now().year
                values = [
                    {"value": year, "name": str(year)}
                    for year in range(current_year - 50, current_year + 1)
                ]
            elif field_name == "avg_rating":
                # Значения для рейтинга (от 0 до 10 с шагом 0.5)
                values = [
                    {"value": rating, "name": str(rating)}
                    for rating in [i/2 for i in range(21)]  # 0, 0.5, 1, ..., 10
                ]
            elif field_name == "site_id":
                # Значения для сайтов
                values = await external_service.get_sites()
            elif field_name == "status_id":
                # Значения для статусов
                values = await external_service.get_field_metadata(field_name)
            elif field_name == "type_id":
                # Значения для типов
                values = await external_service.get_field_metadata(field_name)
            
            title_fields.append(
                FieldMetadata(
                    name=field_name,
                    description=field_data["description"],
                    type=field_data["type"],
                    operators=field_data["operators"],
                    values=values
                )
            )
        
        schedule_types = [
            ScheduleType(
                id="once_day",
                name="Ежедневно",
                description="Ежедневно в 00:00",
                value="once_day"
            ),
            ScheduleType(
                id="once_year",
                name="Ежегодно",
                description="Ежегодно 14.02 в 04:00, приоритетнее ежедневных",
                value="once_year"
            )
        ]
        
        return cls(
            title_fields=title_fields,
            schedule_types=schedule_types
        )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title_fields": [
                    {
                        "name": "is_erotic",
                        "description": "Эротический контент",
                        "type": "boolean",
                        "operators": ["equals", "not_equals"],
                        "values": [
                            {"value": 0, "name": "Нет"},
                            {"value": 1, "name": "Да"}
                        ]
                    }
                ],
                "schedule_types": [
                    {
                        "id": "once_day",
                        "name": "Ежедневно",
                        "description": "Ежедневно в 00:00",
                        "value": "once_day"
                    }
                ]
            }
        }
    )


class ConfigResponse(BaseModel):
    """Ответ с конфигурацией"""
    config: RecommendationConfig = Field(..., description="Конфигурация рекомендаций")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "name": "default_recommendation_config",
                    "description": "Базовая конфигурация рекомендательной системы",
                    "is_active": True,
                    "title_field_filters": [
                        {
                            "field_name": "is_erotic",
                            "operatАor": "not_equals",
                            "values": [True],
                            "is_active": True
                        }
                    ],
                    "schedules_dates": [
                        {
                            "type": "once_day",
                            "date_like": "04:00",
                            "is_active": True
                        }
                    ]
                }
            }
        }
    )


class ErrorResponse(BaseModel):
    """Стандартизированный ответ с ошибкой"""
    error_code: str = Field(..., description="Код ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Union[str, int, bool]]] = Field(None, description="Дополнительные детали ошибки")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_code": "NOT_FOUND",
                "message": "Configuration not found",
                "details": {
                    "config_id": "507f1f77bcf86cd799439011"
                }
            }
        }
    )


class RecommendationRequest(BaseModel):
    """Запрос на получение рекомендаций"""
    user_id: Optional[int] = Field(None, description="ID пользователя")
    title_id: Optional[int] = Field(None, description="ID тайтла")
    limit: int = Field(10, description="Количество рекомендаций", ge=1, le=100)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 12345,
                "limit": 10
            }
        }
    )


class RecommendationResponse(BaseModel):
    """Ответ с рекомендациями"""
    items: List[RecommendationItem] = Field(..., description="Список рекомендованных тайтлов")
    total: int = Field(..., description="Общее количество рекомендаций")
    page: int = Field(1, description="Номер страницы", ge=1)
    page_size: int = Field(10, description="Размер страницы", ge=1, le=100)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 12345,
                        "title": "Название тайтла",
                        "score": 0.95,
                        "type": "manga",
                        "status": "ongoing",
                        "cover_url": "https://example.com/cover.jpg"
                    }
                ],
                "total": 100,
                "page": 1,
                "page_size": 10
            }
        }
    )


class TrainRequest(BaseModel):
    """Запрос на обучение модели"""
    force_retrain: bool = Field(False, description="Принудительное переобучение модели")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "force_retrain": False
            }
        }
    )


class TrainResponse(BaseModel):
    """Ответ на запрос обучения модели"""
    success: bool = Field(..., description="Успешность обучения")
    message: str = Field(..., description="Сообщение о результате")
    version: int = Field(..., description="Версия модели")
    metrics: Optional[Dict[str, float]] = Field(None, description="Метрики качества модели")

    model_config = ConfigDict(
        json_schema_extra={
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
    )
