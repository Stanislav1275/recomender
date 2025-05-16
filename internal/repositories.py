from typing import List, Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from internal.models import RecommendationConfigModel, ConfigExecutionLog
from internal.utils.error_handler import NotFoundError
from internal.config.mongo_adapter import mongo_adapter


class ConfigRepository:
    """Репозиторий для работы с конфигурациями рекомендаций"""
    
    def __init__(self, client: AsyncIOMotorClient):
        self.db = client[mongo_adapter.db_name]
        self.collection = self.db.recommendation_configs
        self.logs_collection = self.db.config_execution_logs

    async def create_indexes(self):
        """Создать индексы в коллекции"""
        await self.collection.create_index("name", unique=True)
        await self.collection.create_index("is_active")
        await self.collection.create_index("created_at")
        await self.collection.create_index("schedules_dates.next_run")
        
        await self.logs_collection.create_index("config_id")
        await self.logs_collection.create_index("executed_at")
        await self.logs_collection.create_index("status")

    async def get_by_id(self, config_id: str) -> Optional[RecommendationConfigModel]:
        """Получить конфигурацию по ID"""
        config = await self.collection.find_one({"_id": config_id})
        if not config:
            return None
        return RecommendationConfigModel.from_dict(config)

    async def get_by_name(self, name: str) -> Optional[RecommendationConfigModel]:
        """Получить конфигурацию по имени"""
        config = await self.collection.find_one({"name": name})
        if not config:
            return None
        return RecommendationConfigModel.from_dict(config)

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[RecommendationConfigModel]:
        """Получить все конфигурации с пагинацией"""
        cursor = self.collection.find().skip(skip).limit(limit)
        configs = await cursor.to_list(length=limit)
        return [RecommendationConfigModel.from_dict(config) for config in configs]

    async def create(self, config: RecommendationConfigModel) -> RecommendationConfigModel:
        """Создать новую конфигурацию"""
        config_dict = config.to_dict()
        result = await self.collection.insert_one(config_dict)
        config.id = result.inserted_id
        return config

    async def update(self, config_id: str, config: RecommendationConfigModel) -> RecommendationConfigModel:
        """Обновить конфигурацию"""
        config_dict = config.to_dict()
        config_dict["updated_at"] = datetime.utcnow()
        
        result = await self.collection.update_one(
            {"_id": config_id},
            {"$set": config_dict}
        )
        
        if result.modified_count == 0:
            raise NotFoundError(f"Configuration with id {config_id} not found")
            
        return config

    async def delete(self, config_id: str) -> bool:
        """Удалить конфигурацию"""
        result = await self.collection.delete_one({"_id": config_id})
        return result.deleted_count > 0

    async def get_active_configs(self) -> List[RecommendationConfigModel]:
        """Получить все активные конфигурации"""
        cursor = self.collection.find({"is_active": True})
        configs = await cursor.to_list(length=None)
        return [RecommendationConfigModel.from_dict(config) for config in configs]

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


