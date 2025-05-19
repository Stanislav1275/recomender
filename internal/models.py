from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Annotated, TypeVar

from bson import ObjectId
from pydantic import BaseModel, Field, BeforeValidator, validator, ConfigDict

from .constants import FIELD_OPTIONS


class FilterOperator(str, Enum):
    """Операторы для фильтров"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"


T = TypeVar('T', bool, int, float)


def validate_object_id(v: Union[str, ObjectId]) -> str:
    """Валидатор для ObjectId"""
    if isinstance(v, ObjectId):
        return str(v)
    if not ObjectId.is_valid(v):
        raise ValueError("Invalid ObjectId")
    return str(v)


PyObjectId = Annotated[
    str,
    BeforeValidator(validate_object_id),
    Field(pattern="^[0-9a-fA-F]{24}$", description="MongoDB ObjectId")
]


class MongoBaseModel(BaseModel):
    """Базовая модель для MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )


class FieldFilter(BaseModel):
    """Фильтр по полю"""
    field_name: str = Field(..., description="Имя поля")
    operator: str = Field(..., description="Оператор сравнения")
    values: List[Union[bool, int, float, str]] = Field(..., description="Значения для сравнения")
    is_active: bool = Field(True, description="Активен ли фильтр")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field_name": "is_erotic",
                "operator": "equals",
                "values": [False],
                "is_active": True
            }
        }
    )


class ScheduleDate(BaseModel):
    """Расписание запуска"""
    type: str = Field(..., description="Тип расписания")
    date_like: str = Field(..., description="Дата/время в формате cron")
    is_active: bool = Field(True, description="Активно ли расписание")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "once_day",
                "date_like": "04:00",
                "is_active": True
            }
        }
    )


class RecommendationConfig(MongoBaseModel):
    """Конфигурация рекомендаций"""
    id: Optional[str] = Field(None, description="ID конфигурации")
    name: str = Field(..., description="Название конфигурации")
    description: str = Field(..., description="Описание конфигурации")
    is_active: bool = Field(True, description="Активна ли конфигурация")
    title_field_filters: List[FieldFilter] = Field(..., description="Фильтры по полям тайтлов")
    schedules_dates: List[ScheduleDate] = Field(..., description="Расписания запуска")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "default_recommendation_config",
                "description": "Базовая конфигурация рекомендационной системы",
                "is_active": True,
                "title_field_filters": [
                    {
                        "field_name": "is_erotic",
                        "operator": "not_equals",
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
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationConfig":
        """Создать конфигурацию из словаря"""
        if "_id" in data:
            data["id"] = str(data["_id"])
            del data["_id"]
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать конфигурацию в словарь"""
        data = self.model_dump()
        if "id" in data:
            del data["id"]
        return data


class ConfigExecutionLog(MongoBaseModel):
    """Лог выполнения конфигурации"""
    config_id: str = Field(..., description="ID конфигурации")
    status: str = Field(..., description="Статус выполнения")
    message: str = Field(..., description="Сообщение о результате")
    execution_time: float = Field(..., description="Время выполнения в секундах")
    items_processed: int = Field(..., description="Количество обработанных элементов")
    error: Optional[str] = Field(None, description="Текст ошибки, если есть")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config_id": "507f1f77bcf86cd799439011",
                "status": "success",
                "message": "Configuration executed successfully",
                "execution_time": 1.5,
                "items_processed": 1000,
                "error": None
            }
        }
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigExecutionLog":
        """Создать лог из словаря"""
        if "_id" in data:
            data["id"] = str(data["_id"])
        return cls(**data)


class RecommendationItem(BaseModel):
    """Элемент рекомендации"""
    id: int = Field(..., description="ID тайтла")
    main_name: str = Field(..., description="Название тайтла")
    dir: str = Field(..., description="URL-путь тайтла")
    score: float = Field(..., description="Оценка релевантности", ge=0, le=1)
    
    # Основные характеристики
    status_id: Optional[int] = Field(None, description="ID статуса тайтла")
    type_id: Optional[int] = Field(None, description="ID типа тайтла")
    age_limit: int = Field(..., description="Возрастное ограничение")
    count_chapters: int = Field(..., description="Количество глав")
    issue_year: Optional[int] = Field(None, description="Год выпуска")
    
    # Флаги
    is_yaoi: int = Field(..., description="Является ли яой")
    is_erotic: int = Field(..., description="Эротический контент")
    is_legal: int = Field(..., description="Легальный контент")
    is_licensed: int = Field(..., description="Лицензированный контент")
    
    # Статистика
    total_views: int = Field(..., description="Общее количество просмотров")
    total_votes: int = Field(..., description="Общее количество голосов")
    avg_rating: float = Field(..., description="Средний рейтинг", ge=0, le=10)
    
    # Метаданные
    cover: Optional[Dict[str, Any]] = Field(None, description="Обложка тайтла")
    upload_date: Optional[datetime] = Field(None, description="Дата загрузки")
    last_chapter_uploaded: Optional[datetime] = Field(None, description="Дата загрузки последней главы")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                    "mid": "https://example.com/cover.jpg"
                },
                "upload_date": "2024-02-14T04:00:00Z",
                "last_chapter_uploaded": "2024-02-14T04:00:00Z"
            }
        }
    )


class RecommendationResult(BaseModel):
    """Результат рекомендаций"""
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


FieldFilter = FieldFilter
RecommendationConfig = RecommendationConfig
ScheduleDate = ScheduleDate


