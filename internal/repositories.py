from typing import List, Optional
from external_db.db_connecter import get_external_session
from external_db.models import DjangoSite
from internal.models import RecommendationConfig as DBRecommendationConfig
from datetime import datetime, timezone
from bson import ObjectId
from internal.types import (
    FieldOptions,
    RelatedTableMetadata,
    FieldMetadata
)

from internal.utils.field_mapping import RELATED_TABLE_MAPPINGS, TITLES_FIELD_MAPPINGS


class ConfigRepository:
    """Репозиторий для работы с конфигурациями рекомендаций"""

    @staticmethod
    def get_by_id(config_id: str) -> Optional[DBRecommendationConfig]:
        """Получить конфигурацию по ID"""
        try:
            return DBRecommendationConfig.objects.get(id=ObjectId(config_id))
        except (DBRecommendationConfig.DoesNotExist, ValueError):
            return None

    @staticmethod
    def get_all() -> List[DBRecommendationConfig]:
        """Получить все конфигурации"""
        return list(DBRecommendationConfig.objects.all())

    @staticmethod
    def get_active() -> List[DBRecommendationConfig]:
        """Получить активные конфигурации"""
        return list(DBRecommendationConfig.objects.filter(is_active=True))

    @staticmethod
    def create(config_data: dict) -> DBRecommendationConfig:
        """Создать новую конфигурацию"""
        config = DBRecommendationConfig(**config_data)
        config.save()
        return config

    @staticmethod
    def update(config_id: str, config_data: dict) -> Optional[DBRecommendationConfig]:
        """Обновить существующую конфигурацию"""
        config = ConfigRepository.get_by_id(config_id)
        if not config:
            return None

        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.now(timezone.utc)
        config.save()
        return config

    @staticmethod
    def delete(config_id: str) -> bool:
        """Удалить конфигурацию по ID"""
        config = ConfigRepository.get_by_id(config_id)
        if not config:
            return False

        config.delete()
        return True


class FieldMappingRepository:
    """Репозиторий для работы с маппингами полей"""

    @staticmethod
    def get_title_fields() -> dict[str, FieldMetadata]:
        """Получить все поля из таблицы Titles"""
        return TITLES_FIELD_MAPPINGS

    @staticmethod
    def get_related_table_fields(table_name: str) -> dict[str, FieldMetadata]:
        """Получить поля для связанной таблицы"""
        return RELATED_TABLE_MAPPINGS.get(table_name, {})

    @staticmethod
    def get_sites() -> List[DjangoSite]:
        """Получить список всех сайтов"""
        with get_external_session() as db:
            return db.query(DjangoSite).all()

    @staticmethod
    def get_site_by_id(site_id: int) -> Optional[DjangoSite]:
        """Получить сайт по ID"""
        with get_external_session() as db:
            return db.query(DjangoSite).filter(DjangoSite.id == site_id).first()
