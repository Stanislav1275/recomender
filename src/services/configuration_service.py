from typing import Dict, List, Any, Optional, Tuple
from external_db.data_service import ExternalDataService
from internal.repositories import ConfigRepository
from internal.models import FilterOperator

class AdminPanelService:
    """Сервис для админ-панели"""

    @staticmethod
    def get_field_options() -> Dict[str, Any]:
        """Получить опции полей для формы в админке"""
        external_data = ExternalDataService()

        # Получаем метаданные полей из внешней БД
        title_fields = external_data.get_field_metadata()
        related_tables = external_data.get_related_tables_metadata()
        sites = external_data.get_sites()

        formatted_sites = {
            "values": [site["id"] for site in sites],
            "mapping": [site["name"] for site in sites]
        }

        operators = [
            {'value': op.value, 'display': op.value}
            for op in FilterOperator
        ]

        schedule_types = [
            {'value': 'once_day', 'display': 'Ежедневно(00:00)'},
            {'value': 'once_year', 'display': 'Ежегодно (14.02:04:00), приоритетнее ежедневных'}
        ]

        return {
            'title_fields': title_fields,
            'related_tables': related_tables,
            'sites': formatted_sites,
            'operators': operators,
            'schedule_types': schedule_types
        }

    @staticmethod
    def get_config_template() -> Dict[str, Any]:
        """Получить шаблон конфигурации для создания"""
        return {
            'name': '',
            'description': '',
            'is_active': True,
            'title_field_filters': [],
            'related_table_filters': [],
            'schedules_dates': [
                {
                    'type': 'once_day',
                    'date_like': '04:00',
                    'is_active': True
                }
            ]
        }

    @staticmethod
    def get_config_with_metadata(config_id: str) -> Optional[Dict[str, Any]]:
        """Получить конфигурацию с метаданными"""
        config = ConfigRepository.get_by_id(config_id)
        if not config:
            return None

        # Получаем данные из внешней БД для отображения
        external_data = ExternalDataService()

        # Получаем словарь конфигурации и преобразуем ObjectId в строку
        config_dict = config.to_mongo().to_dict()
        if '_id' in config_dict:
            config_dict['_id'] = str(config_dict['_id'])

        # Собираем информацию о полях для отображения в админке
        result = {
            'config': config_dict,
            'metadata': {
                'title_fields': external_data.get_field_metadata(),
                'related_tables': external_data.get_related_tables_metadata(),
                'sites': {
                    "values": [site["id"] for site in external_data.get_sites()],
                    "mapping": [site["name"] for site in external_data.get_sites()]
                },
                'operators': [op.value for op in FilterOperator]
            }
        }

        return result
