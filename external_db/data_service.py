from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import select
from external_db.db_connecter import get_external_session
from external_db.models import DjangoSite, Titles, TitleStatus, Genres, Categories, TitleType


class ExternalDataService:
    """Сервис для работы с данными из внешней БД"""

    @staticmethod
    def get_sites() -> List[Dict[str, Any]]:
        """Получить список всех сайтов из таблицы DjangoSite"""
        with get_external_session() as session:
            stmt = select(DjangoSite)
            result = session.execute(stmt).scalars().all()
            return [
                {"id": site.id, "name": site.name, "domain": site.domain}
                for site in result
            ]

    @staticmethod
    def get_title_statuses() -> List[Dict[str, Any]]:
        """Получить все статусы произведений"""
        with get_external_session() as session:
            stmt = select(TitleStatus)
            result = session.execute(stmt).scalars().all()
            return [
                {"id": status.id, "name": status.name}
                for status in result
            ]

    @staticmethod
    def get_title_types() -> List[Dict[str, Any]]:
        """Получить все типы произведений"""
        with get_external_session() as session:
            stmt = select(TitleType)
            result = session.execute(stmt).scalars().all()
            return [
                {"id": type_.id, "name": type_.name, "description": type_.description}
                for type_ in result
            ]

    @staticmethod
    def get_genres() -> List[Dict[str, Any]]:
        """Получить все жанры"""
        with get_external_session() as session:
            stmt = select(Genres)
            result = session.execute(stmt).scalars().all()
            return [
                {"id": genre.id, "name": genre.name, "description": genre.description}
                for genre in result
            ]

    @staticmethod
    def get_categories() -> List[Dict[str, Any]]:
        """Получить все категории"""
        with get_external_session() as session:
            stmt = select(Categories)
            result = session.execute(stmt).scalars().all()
            return [
                {"id": category.id, "name": category.name, "description": category.description}
                for category in result
            ]

    @staticmethod
    def get_field_metadata() -> Dict[str, Dict[str, Any]]:
        """Получить метаданные о полях таблицы Titles"""
        return {
            "status_id": {
                "name": "Статус произведения",
                "type": "reference",
                "reference_table": "title_status",
                "values": ExternalDataService.get_title_statuses()
            },
            "type_id": {
                "name": "Тип произведения",
                "type": "reference",
                "reference_table": "title_type",
                "values": ExternalDataService.get_title_types()
            },
            "is_yaoi": {
                "name": "Яой",
                "type": "boolean",
                "values": [
                    {"id": 0, "name": "Нет"},
                    {"id": 1, "name": "Да"}
                ]
            },
            "is_erotic": {
                "name": "Эротика",
                "type": "boolean",
                "values": [
                    {"id": 0, "name": "Нет"},
                    {"id": 1, "name": "Да"}
                ]
            },
            "is_legal": {
                "name": "Легальный",
                "type": "boolean",
                "values": [
                    {"id": 0, "name": "Нет"},
                    {"id": 1, "name": "Да"}
                ]
            },
            "age_limit": {
                "name": "Возрастное ограничение",
                "type": "integer",
                "values": [
                    {"id": 0, "name": "0+"},
                    {"id": 1, "name": "12+"},
                    {"id": 2, "name": "18+"},
                ]
            }
        }

    @staticmethod
    def get_related_tables_metadata() -> Dict[str, Dict[str, Any]]:
        """Получить метаданные о связанных таблицах"""
        return {
            "titles_sites": {
                "name": "Сайты произведения",
                "fields": {
                    "site_id": {
                        "name": "Сайт",
                        "type": "reference",
                        "reference_table": "django_site",
                        "values": ExternalDataService.get_sites()
                    }
                }
            },
            "titles_genres": {
                "name": "Жанры произведения",
                "fields": {
                    "genre_id": {
                        "name": "Жанр",
                        "type": "reference",
                        "reference_table": "genres",
                        "values": ExternalDataService.get_genres()
                    }
                }
            },
            "titles_categories": {
                "name": "Категории произведения",
                "fields": {
                    "category_id": {
                        "name": "Категория",
                        "type": "reference",
                        "reference_table": "categories",
                        "values": ExternalDataService.get_categories()
                    }
                }
            }
        }