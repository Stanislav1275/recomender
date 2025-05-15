from mongoengine import Document, StringField, IntField, FloatField, DateTimeField, DictField, ListField, \
    ReferenceField, EmbeddedDocument, fields, ValidationError
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum


class Configuration(Document):
    site_id = StringField(required=True)
    site_name = StringField(required=True)
    name = StringField(required=True)
    filter_params = DictField(required=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)


class FilterOperator(str, Enum):
    """Операторы фильтрации"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class FieldFilter(EmbeddedDocument):
    """Фильтр для поля таблицы Titles"""
    field_name = fields.StringField(required=True)
    operator = fields.EnumField(FilterOperator, required=True)
    values = fields.ListField()

    def clean(self):
        # Валидация что поле существует в таблице Titles
        allowed_fields = [
            'status_id', 'age_limit', 'count_chapters', 'type_id',
            'issue_year', 'is_yaoi', 'is_erotic', 'total_views',
            'total_votes', 'avg_rating', 'is_legal', 'is_licensed',
            'uploaded', 'last_chapter_uploaded'
        ]
        if self.field_name not in allowed_fields:
            raise ValidationError(f'Field {self.field_name} not allowed in Titles table')


class RelatedTableFilter(EmbeddedDocument):
    """Фильтр для связанных таблиц"""
    table_name = fields.StringField(required=True)
    field_name = fields.StringField(required=True)
    operator = fields.EnumField(FilterOperator, required=True)
    values = fields.ListField()

    def clean(self):
        allowed_tables = [
            'titles_genres', 'titles_categories', 'titles_sites',
            'titles_collections', 'bookmarks', 'rating'
        ]
        if self.table_name not in allowed_tables:
            raise ValidationError(f'Table {self.table_name} not allowed for filtering')


class ScheduleConfig(EmbeddedDocument):
    """Конфигурация расписания запуска"""
    type = fields.StringField(required=True, choices=['once_year', 'once_day'])
    date_like = fields.StringField(required=True)  # 'DD.MM:HH:MM' для once_year, 'HH:MM' для once_day
    is_active = fields.BooleanField(default=True)
    next_run = fields.DateTimeField(null=True)

    def clean(self):
        if self.type == 'once_year':
            # Формат: DD.MM:HH:MM (например, "12.04:04:00")
            import re
            if not re.match(r'^\d{2}\.\d{2}:\d{2}:\d{2}$', self.date_like):
                raise ValidationError('For once_year use format DD.MM:HH:MM (e.g., "12.04:04:00")')
        elif self.type == 'once_day':
            # Формат: HH:MM (например, "04:00")
            import re
            if not re.match(r'^\d{2}:\d{2}$', self.date_like):
                raise ValidationError('For once_day use format HH:MM (e.g., "04:00")')


class RecommendationConfig(Document):
    """Основная конфигурация рекомендательной системы"""
    name = fields.StringField(required=True, unique=True)
    description = fields.StringField()
    title_field_filters = fields.ListField(fields.EmbeddedDocumentField(FieldFilter))
    related_table_filters = fields.ListField(fields.EmbeddedDocumentField(RelatedTableFilter))
    # Расписание
    schedules_dates = fields.ListField(fields.EmbeddedDocumentField(ScheduleConfig))
    # Метаданные
    created_at = fields.DateTimeField(default=datetime.now(timezone.utc))
    updated_at = fields.DateTimeField(default=datetime.now(timezone.utc))
    is_active = fields.BooleanField(default=True)

    meta = {
        'collection': 'recommendation_configs',
        'indexes': [
            'name',
            'is_active',
            'created_at',
            'schedules_dates.next_run'
        ]
    }

    def clean(self):
        """Валидация конфигурации"""
        # Убеждаемся что нет дублирующихся полей в фильтрах
        field_names = [f.field_name for f in self.title_field_filters]
        if len(field_names) != len(set(field_names)):
            raise fields.ValidationError('Duplicate field names in title_field_filters')

    def get_active_config(self) -> dict:
        config = {
            'title_field_filters': self.title_field_filters,
            'related_table_filters': self.related_table_filters,
        }
        return config

    def save(self, *args, **kwargs):
        self.updated_at = datetime.now(timezone.utc)
        return super().save(*args, **kwargs)


#
# 1,remanga.org,comics_ReManga
# 2,renovels.org,book_ReNovels
# 3,recomics.org,comics_ReComics
# 4,reanime.org,movie_ReAnime
# 5,rehentai.org,comics_ReHentai
# 6,neremanga.org,comics_NeReManga
class ConfigExecutionLog(Document):
    """Лог выполнения конфигураций"""
    config_id = fields.ObjectIdField(required=True)
    executed_at = fields.DateTimeField(default=datetime.now(timezone.utc))
    status = fields.StringField(choices=['success', 'error', 'partial'], required=True)
    error_message = fields.StringField()
    titles_processed = fields.IntField()
    meta = {
        'collection': 'config_execution_logs',
        'indexes': [
            'config_id',
            'executed_at',
            'status'
        ]
    }


