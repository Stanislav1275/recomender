from typing import TypedDict, List, Optional, Union, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class FieldFilter(BaseModel):
    """Фильтр для поля"""
    field_name: str
    operator: str
    values: List[Union[str, int, float, bool]]
    is_active: bool = True


class RelatedTableFilter(BaseModel):
    """Фильтр для связанной таблицы"""
    table_name: str
    field_name: str
    operator: str
    values: List[Union[str, int, float, bool]]
    is_active: bool = True


class ScheduleConfig(BaseModel):
    """Конфигурация расписания"""
    type: str
    date_like: str
    is_active: bool = True
    next_run: Optional[datetime] = None


class RecommendationConfig(BaseModel):
    """Основная конфигурация рекомендаций"""
    id: Optional[str] = None
    name: str
    description: str
    is_active: bool = True
    title_field_filters: List[FieldFilter] = Field(default_factory=list)
    related_table_filters: List[RelatedTableFilter] = Field(default_factory=list)
    schedules_dates: List[ScheduleConfig] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FieldMetadata(BaseModel):
    """Метаданные поля"""
    name: str
    description: str
    type: str
    values: Optional[List[Dict[str, Any]]] = None
    is_required: bool = False
    default_value: Optional[Any] = None


class RelatedTableMetadata(BaseModel):
    """Метаданные связанной таблицы"""
    name: str
    description: str
    fields: Dict[str, FieldMetadata]
    is_required: bool = False


class SiteConfig(BaseModel):
    """Конфигурация сайта"""
    id: int
    name: str
    is_active: bool = True


class CategoryConfig(BaseModel):
    """Конфигурация категории"""
    id: int
    name: str
    parent_id: Optional[int] = None
    is_active: bool = True


class ChapterConfig(BaseModel):
    """Конфигурация главы"""
    id: int
    title: str
    chapter_number: int
    is_active: bool = True


class ExternalDataMapping(BaseModel):
    """Маппинг для внешних данных"""
    field_name: str
    external_field: str
    transform_function: Optional[str] = None
    default_value: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Стандартизированный ответ с ошибкой"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Стандартизированный успешный ответ"""
    data: Any
    message: Optional[str] = None
