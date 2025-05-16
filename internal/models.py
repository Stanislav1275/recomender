from datetime import datetime
from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: Any,
        _handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string"}


class MongoBaseModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True


class FieldFilterModel(MongoBaseModel):
    field_name: str
    operator: str
    values: List[Any]
    is_active: bool = True


class RelatedTableFilterModel(MongoBaseModel):
    table_name: str
    field_name: str
    operator: str
    values: List[Any]
    is_active: bool = True


class ScheduleConfigModel(MongoBaseModel):
    type: str
    date_like: str
    is_active: bool = True
    next_run: Optional[datetime] = None


class RecommendationConfigModel(MongoBaseModel):
    name: str
    description: str
    is_active: bool = True
    title_field_filters: List[FieldFilterModel] = Field(default_factory=list)
    related_table_filters: List[RelatedTableFilterModel] = Field(default_factory=list)
    schedules_dates: List[ScheduleConfigModel] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать модель в словарь для MongoDB"""
        data = self.dict(by_alias=True, exclude_none=True)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationConfigModel":
        """Создать модель из словаря MongoDB"""
        if "_id" in data:
            data["_id"] = str(data["_id"])
        return cls(**data)


#
# 1,remanga.org,comics_ReManga
# 2,renovels.org,book_ReNovels
# 3,recomics.org,comics_ReComics
# 4,reanime.org,movie_ReAnime
# 5,rehentai.org,comics_ReHentai
# 6,neremanga.org,comics_NeReManga
class ConfigExecutionLog(MongoBaseModel):
    """Лог выполнения конфигураций"""
    config_id: PyObjectId
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    status: str
    error_message: Optional[str] = None
    titles_processed: int


