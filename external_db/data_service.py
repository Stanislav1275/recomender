from typing import Dict, List, Any
from datetime import datetime

from sqlalchemy import select

from external_db.db_connecter import get_external_session
from external_db.models import DjangoSite, TitleStatus, TitleType, Genres


class ExternalDataService:
    """Сервис для работы с данными из внешней БД"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_sites(self) -> List[Dict[str, Any]]:
        """Получить список всех сайтов из таблицы DjangoSite"""
        with get_external_session() as session:
            stmt = select(DjangoSite)
            result = session.execute(stmt).scalars().all()
            return [
                {"value": site.id, "name": site.name}
                for site in result
            ]
    
    async def get_reference_data(self, model_class) -> List[Dict[str, Any]]:
        """Получить справочные данные в формате id-name"""
        with get_external_session() as session:
            stmt = select(model_class)
            result = session.execute(stmt).scalars().all()
            return [
                {"value": item.id, "name": item.name}
                for item in result
            ]

    async def get_field_metadata(self, field_name: str) -> List[Dict[str, Any]]:
        """Получить метаданные для поля"""
        if field_name == "status_id":
            return await self.get_reference_data(TitleStatus)
        elif field_name == "type_id":
            return await self.get_reference_data(TitleType)
        elif field_name == "genre_id":
            return await self.get_reference_data(Genres)
        return None