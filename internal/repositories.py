from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from internal.models import ConfigExecutionLog
from .models import RecommendationConfig
from .types import FieldOptions, ConfigResponse


class RecommendationConfigRepository:
    """Репозиторий для работы с конфигурациями рекомендаций"""
    
    def __init__(self, client: AsyncIOMotorClient):
        self.db = client.recommender_db
        self.collection = self.db.recommendation_configs
        self.logs_collection = self.db.config_execution_logs

    async def create_indexes(self):
        """Создать индексы в коллекции"""
        await self.collection.create_index("name", unique=True)
        await self.collection.create_index("is_active")
        await self.collection.create_index("created_at")
        await self.collection.create_index("updated_at")
        await self.collection.create_index("schedules_dates.next_run")
        
        await self.logs_collection.create_index("config_id")
        await self.logs_collection.create_index("executed_at")
        await self.logs_collection.create_index("status")

    async def get_by_id(self, config_id: str) -> Optional[RecommendationConfig]:
        """Получить конфигурацию по ID"""
        config = await self.collection.find_one({"_id": ObjectId(config_id)})
        if not config:
            return None
        return RecommendationConfig.from_dict(config)

    async def get_by_name(self, name: str) -> Optional[RecommendationConfig]:
        """Получить конфигурацию по имени"""
        config = await self.collection.find_one({"name": name})
        if not config:
            return None
        return RecommendationConfig.from_dict(config)

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[RecommendationConfig]:
        """Получить все конфигурации с пагинацией"""
        cursor = self.collection.find().skip(skip).limit(limit)
        configs = await cursor.to_list(length=limit)
        return [RecommendationConfig.from_dict(config) for config in configs]

    async def create(self, config: RecommendationConfig) -> RecommendationConfig:
        """Создать новую конфигурацию"""
        config_dict = config.to_dict()
        config_dict["created_at"] = datetime.utcnow()
        config_dict["updated_at"] = datetime.utcnow()
        
        # Устанавливаем next_run для расписаний
        for schedule in config_dict.get("schedules_dates", []):
            if schedule.get("is_active"):
                schedule["next_run"] = self._calculate_next_run(
                    schedule["type"],
                    schedule["date_like"]
                )
        
        result = await self.collection.insert_one(config_dict)
        config_dict["_id"] = result.inserted_id
        return RecommendationConfig.from_dict(config_dict)

    async def update(self, config_id: str, config: RecommendationConfig) -> RecommendationConfig:
        """Обновить конфигурацию"""
        config_dict = config.to_dict()
        config_dict["updated_at"] = datetime.utcnow()
        
        # Обновляем next_run для расписаний
        for schedule in config_dict.get("schedules_dates", []):
            if schedule.get("is_active"):
                schedule["next_run"] = self._calculate_next_run(
                    schedule["type"],
                    schedule["date_like"]
                )
        
        # Удаляем _id из словаря, так как он не должен обновляться
        if "_id" in config_dict:
            del config_dict["_id"]
            
        await self.collection.update_one(
            {"_id": ObjectId(config_id)},
            {"$set": config_dict}
        )
        
        # Получаем обновленный документ
        updated_config = await self.get_by_id(config_id)
        if not updated_config:
            raise ValueError(f"Config with id {config_id} not found after update")
        return updated_config

    async def delete(self, config_id: str) -> bool:
        """Удалить конфигурацию"""
        result = await self.collection.delete_one({"_id": ObjectId(config_id)})
        return result.deleted_count > 0

    async def get_active_configs(self) -> List[RecommendationConfig]:
        """Получить все активные конфигурации"""
        cursor = self.collection.find({"is_active": True})
        configs = await cursor.to_list(length=None)
        return [RecommendationConfig.from_dict(config) for config in configs]

    async def add_execution_log(self, log: ConfigExecutionLog) -> ConfigExecutionLog:
        """Добавить лог выполнения конфигурации"""
        log_dict = log.dict(by_alias=True)
        result = await self.logs_collection.insert_one(log_dict)
        log.id = result.inserted_id
        return log

    async def get_execution_logs(
        self,
        config_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ConfigExecutionLog]:
        """Получить логи выполнения конфигурации"""
        cursor = self.logs_collection.find({"config_id": config_id}).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        return [ConfigExecutionLog(**log) for log in logs]

    def _calculate_next_run(self, schedule_type: str, date_like: str) -> datetime:
        """Рассчитать следующее время выполнения для расписания"""
        now = datetime.utcnow()
        
        if schedule_type == "once_day":
            # Формат HH:MM
            hours, minutes = map(int, date_like.split(":"))
            next_run = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run.replace(day=next_run.day + 1)
                
        elif schedule_type == "once_year":
            # Формат DD.MM:HH:MM
            date_part, time_part = date_like.split(":")
            day, month = map(int, date_part.split("."))
            hours, minutes = map(int, time_part.split(":"))
            
            next_run = now.replace(
                month=month,
                day=day,
                hour=hours,
                minute=minutes,
                second=0,
                microsecond=0
            )
            
            if next_run <= now:
                next_run = next_run.replace(year=next_run.year + 1)
                
        return next_run

    async def get_config_by_uid(self, config_id: str) -> Optional[ConfigResponse]:
        """Получить конфигурацию с опциями полей"""
        config = await self.get_by_id(config_id)
        if not config:
            return None
            
        field_options = FieldOptions.from_constants()
        return ConfigResponse(config=config, field_options=field_options)
    @staticmethod
    async def get_field_options() -> FieldOptions:
        """Получить опции полей для админ-панели"""
        return FieldOptions.from_constants()

    async def get_configs_with_options(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ConfigResponse]:
        """Получить список конфигураций с опциями полей"""
        configs = await self.get_all(skip, limit)
        field_options = await self.get_field_options()
        
        return [
            ConfigResponse(config=config, field_options=field_options)
            for config in configs
        ]


