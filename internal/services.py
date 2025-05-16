from datetime import datetime, timezone
from typing import List, Optional, Tuple

from external_db.data_service import ExternalDataService
from internal.models import RecommendationConfig as DBRecommendationConfig, FieldFilter as DBFieldFilter, \
    RelatedTableFilter as DBRelatedTableFilter, FilterOperator, ScheduleConfig as DBScheduleConfig
from internal.repositories import ConfigRepository
from internal.types import (
    RecommendationConfig,
    FieldOptions,
    ConfigWithMetadata
)


class ConfigService:
    """Сервис для работы с конфигурациями"""
    
    @staticmethod
    def create_config(config_data: RecommendationConfig) -> DBRecommendationConfig:
        """Создать новую конфигурацию"""
        # Предобработка данных
        title_filters = []
        for filter_data in config_data.get('title_field_filters', []):
            field_filter = DBFieldFilter(
                field_name=filter_data['field_name'],
                operator=filter_data['operator'],
                values=filter_data['values']
            )
            title_filters.append(field_filter)
        
        related_filters = []
        for filter_data in config_data.get('related_table_filters', []):
            related_filter = DBRelatedTableFilter(
                table_name=filter_data['table_name'],
                field_name=filter_data['field_name'],
                operator=filter_data['operator'],
                values=filter_data['values']
            )
            related_filters.append(related_filter)
        
        schedules = []
        for schedule_data in config_data.get('schedules_dates', []):
            schedule = DBScheduleConfig(
                type=schedule_data['type'],
                date_like=schedule_data['date_like'],
                is_active=schedule_data.get('is_active', True),
                next_run=schedule_data.get('next_run')
            )
            schedules.append(schedule)
        
        # Создаем конфигурацию
        config = DBRecommendationConfig(
            name=config_data['name'],
            description=config_data.get('description', ''),
            is_active=config_data.get('is_active', True),
            title_field_filters=title_filters,
            related_table_filters=related_filters,
            schedules_dates=schedules,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        config.save()
        return config
    
    @staticmethod
    def update_config(config_id: str, config_data: RecommendationConfig) -> Optional[DBRecommendationConfig]:
        """Обновить существующую конфигурацию"""
        config = ConfigRepository.get_by_id(config_id)
        if not config:
            return None
        
        # Обновляем только те поля, которые предоставлены
        if 'name' in config_data:
            config.name = config_data['name']
        
        if 'description' in config_data:
            config.description = config_data['description']
        
        if 'is_active' in config_data:
            config.is_active = config_data['is_active']
        
        if 'title_field_filters' in config_data:
            title_filters = []
            for filter_data in config_data['title_field_filters']:
                field_filter = DBFieldFilter(
                    field_name=filter_data['field_name'],
                    operator=filter_data['operator'],
                    values=filter_data['values']
                )
                title_filters.append(field_filter)
            config.title_field_filters = title_filters
        
        if 'related_table_filters' in config_data:
            related_filters = []
            for filter_data in config_data['related_table_filters']:
                related_filter = DBRelatedTableFilter(
                    table_name=filter_data['table_name'],
                    field_name=filter_data['field_name'],
                    operator=filter_data['operator'],
                    values=filter_data['values']
                )
                related_filters.append(related_filter)
            config.related_table_filters = related_filters
        
        if 'schedules_dates' in config_data:
            schedules = []
            for schedule_data in config_data['schedules_dates']:
                schedule = DBScheduleConfig(
                    type=schedule_data['type'],
                    date_like=schedule_data['date_like'],
                    is_active=schedule_data.get('is_active', True),
                    next_run=schedule_data.get('next_run')
                )
                schedules.append(schedule)
            config.schedules_dates = schedules
        
        config.updated_at = datetime.now(timezone.utc)
        config.save()
        return config

    @staticmethod
    def get_config_with_metadata(config_id: str) -> Optional[ConfigWithMetadata]:
        """Получить конфигурацию с метаданными"""
        config = ConfigRepository.get_by_id(config_id)
        if not config:
            return None

        external_data = ExternalDataService()

        config_dict = config.to_mongo().to_dict()
        if '_id' in config_dict:
            config_dict['_id'] = str(config_dict['_id'])

        # --- Добавляем description в title_fields ---
        title_fields = external_data.get_field_metadata()
        for field_key, field in title_fields.items():
            if 'description' not in field:
                field['description'] = field['name']
            # Если есть values, добавим description для каждого значения (если нужно)
            if 'values' in field and isinstance(field['values'], list):
                for v in field['values']:
                    if 'description' not in v:
                        v['description'] = v.get('name', '')

        # --- Добавляем description в related_tables и их поля ---
        related_tables = external_data.get_related_tables_metadata()
        for table_key, table in related_tables.items():
            if 'description' not in table:
                table['description'] = table['name']
            if 'fields' in table:
                for field_key, field in table['fields'].items():
                    if 'description' not in field:
                        field['description'] = field['name']
                    if 'values' in field and isinstance(field['values'], list):
                        for v in field['values']:
                            if 'description' not in v:
                                v['description'] = v.get('name', '')

        operators = [
            {'value': op.value, 'display': op.value}
            for op in FilterOperator
        ] if 'FilterOperator' in globals() else []

        schedule_types = [
            {'value': 'once_day', 'display': 'Ежедневно(00:00)'},
            {'value': 'once_year', 'display': 'Ежегодно (14.02:04:00), приоритетнее ежедневных'}
        ]

        # --- Формируем sites ---
        sites = external_data.get_sites()
        formatted_sites = {
            "values": [site["id"] for site in sites],
            "mapping": [site["name"] for site in sites]
        }

        result: ConfigWithMetadata = {
            'config': config_dict,
            'metadata': {
                'title_fields': title_fields,
                'related_tables': related_tables,
                'sites': formatted_sites,
                'operators': [],
                'schedule_types': schedule_types
            }
        }

        return result


class AdminPanelService:
    """Сервис для админ-панели"""
    
    @staticmethod
    def get_field_options() -> FieldOptions:
        external_data = ExternalDataService()

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
            'operators': [],
            'schedule_types': schedule_types
        }
    
    @staticmethod
    def validate_config(config_data: RecommendationConfig) -> Tuple[bool, List[str]]:
        """Валидация конфигурации перед сохранением"""
        errors = []
        external_data = ExternalDataService()
        
        # Базовая валидация
        if not config_data.get('name'):
            errors.append('Имя конфигурации обязательно')
        
        # Валидация фильтров полей
        title_fields = external_data.get_field_metadata()
        for field_filter in config_data.get('title_field_filters', []):
            field_name = field_filter.get('field_name')
            if field_name not in title_fields:
                errors.append(f'Поле {field_name} не существует в таблице Titles')
            
            # Проверка значений
            if field_name in title_fields:
                field_info = title_fields[field_name]
                allowed_values = [v["id"] for v in field_info.get('values', [])] if field_info.get('values') else None
                if allowed_values:
                    for value in field_filter.get('values', []):
                        if value not in allowed_values:
                            errors.append(f'Значение {value} не допустимо для поля {field_name}')
        
        # Валидация связанных таблиц
        related_tables = external_data.get_related_tables_metadata()
        for related_filter in config_data.get('related_table_filters', []):
            table_name = related_filter.get('table_name')
            if table_name not in related_tables:
                errors.append(f'Таблица {table_name} не существует или не поддерживается')
            
            # Проверка полей
            if table_name and table_name in related_tables:
                fields = related_tables[table_name].get('fields', {})
                field_name = related_filter.get('field_name')
                if field_name not in fields:
                    errors.append(f'Поле {field_name} не существует в таблице {table_name}')
        
        # Валидация расписания
        for schedule in config_data.get('schedules_dates', []):
            schedule_type = schedule.get('type')
            date_like = schedule.get('date_like')
            
            if schedule_type == 'once_year':
                import re
                if not re.match(r'^\d{2}\.\d{2}:\d{2}:\d{2}$', date_like):
                    errors.append('Для типа once_year используйте формат DD.MM:HH:MM (e.g., "12.04:04:00")')
            elif schedule_type == 'once_day':
                import re
                if not re.match(r'^\d{2}:\d{2}$', date_like):
                    errors.append('Для типа once_day используйте формат HH:MM (e.g., "04:00")')
        
        return len(errors) == 0, errors 