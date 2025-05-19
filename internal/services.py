from typing import List, Optional
from datetime import datetime

from .models import (
    RecommendationConfig,
    ConfigExecutionLog,
    RecommendationItem,
    RecommendationResult
)
from .types import (
    ConfigResponse,
    FieldOptions,
    RecommendationRequest,
    TrainRequest,
    TrainResponse
)
from .repositories import RecommendationConfigRepository
from external_db.data_service import ExternalDataService


class RecommendationConfigService:
    """Сервис для работы с конфигурациями рекомендаций"""
    
    def __init__(self, repository: RecommendationConfigRepository, external_service: ExternalDataService):
        self.repository = repository
        self.external_service = external_service

    async def get_config(self, config_id: str) -> Optional[ConfigResponse]:
        """Получить конфигурацию"""
        config = await self.repository.get_by_id(config_id)
        if not config:
            return None
        return ConfigResponse(config=config)

    async def get_configs(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ConfigResponse]:
        """Получить список конфигураций"""
        configs = await self.repository.get_all(skip, limit)
        return [ConfigResponse(config=config) for config in configs]

    async def create_config(self, config: RecommendationConfig) -> ConfigResponse:
        """Создать новую конфигурацию"""
        created_config = await self.repository.create(config)
        return ConfigResponse(config=created_config)

    async def update_config(
        self,
        config_id: str,
        config: RecommendationConfig
    ) -> ConfigResponse:
        """Обновить конфигурацию"""
        updated_config = await self.repository.update(config_id, config)
        return ConfigResponse(config=updated_config)

    async def delete_config(self, config_id: str) -> bool:
        """Удалить конфигурацию"""
        return await self.repository.delete(config_id)

    async def get_config_logs(
        self,
        config_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ConfigExecutionLog]:
        """Получить логи выполнения конфигурации"""
        return await self.repository.get_execution_logs(config_id, skip, limit)

    async def add_execution_log(self, log: ConfigExecutionLog) -> ConfigExecutionLog:
        """Добавить лог выполнения конфигурации"""
        return await self.repository.add_execution_log(log)

    async def get_field_options(self) -> FieldOptions:
        """Получить опции полей для админ-панели"""
        return await FieldOptions.from_constants(self.external_service)

    async def get_active_configs(self) -> List[ConfigResponse]:
        """Получить все активные конфигурации"""
        configs = await self.repository.get_active_configs()
        return [ConfigResponse(config=config) for config in configs]


class RecommendationService:
    """Сервис для работы с рекомендациями"""
    
    def __init__(self):
        """Инициализация сервиса рекомендаций"""
        self.grpc_client = None  # TODO: Инициализировать gRPC клиент

    async def get_user_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResult:
        """Получить рекомендации для пользователя"""
        # TODO: Реализовать получение рекомендаций через gRPC
        raise NotImplementedError()

    async def get_title_relevant(
        self,
        request: RecommendationRequest
    ) -> RecommendationResult:
        """Получить похожие тайтлы"""
        # TODO: Реализовать получение похожих тайтлов через gRPC
        raise NotImplementedError()

    async def train_model(self, request: TrainRequest) -> TrainResponse:
        """Обучить модель рекомендаций"""
        # TODO: Реализовать обучение модели через gRPC
        raise NotImplementedError() 