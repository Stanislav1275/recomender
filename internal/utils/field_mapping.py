from enum import Enum
from typing import List, Dict, Any, Union, Optional, TypedDict
from internal.types import FieldMetadata


class FieldType(str, Enum):
    """
    Enumeration of possible field types in the system.
    Used to define the data type of fields in the recommendation system.
    """
    INTEGER = "integer"  # Whole numbers
    BOOLEAN = "boolean"  # True/False values
    STRING = "string"    # Text values
    FLOAT = "float"      # Decimal numbers
    DATETIME = "datetime"  # Date and time values
    DATE = "date"        # Date values only
    ENUM = "enum"        # Predefined set of values
    REFERENCE = "reference"  # Reference to another table


class FieldMapping(TypedDict):
    """Структура маппинга поля"""
    name: str                  # Имя поля
    description: str           # Описание поля
    type: FieldType            # Тип поля
    table: str                 # Таблица, в которой находится поле
    allowed_values: list       # Разрешенные значения (для enum)
    reference_table: str       # Таблица-ссылка (для reference)
    reference_display: str     # Поле для отображения (для reference)
    default_value: Any         # Значение по умолчанию


class TableRelation(TypedDict):
    """
    Defines a relationship between two tables in the system.
    Used to establish connections between different data entities.
    """
    table_name: str      # Name of the relationship table
    source_table: str    # Name of the source table
    target_table: str    # Name of the target table
    source_field: str    # Field in the source table
    target_field: str    # Field in the target table


# Mapping of fields from the Titles table
TITLES_FIELD_MAPPINGS: Dict[str, FieldMetadata] = {
    "status_id": {
        "name": "status_id",
        "description": "Status of the title (e.g., ongoing, completed)",
        "type": FieldType.REFERENCE,
        "values": None
    },
    "age_limit": {
        "name": "age_limit",
        "description": "Age restriction for the content",
        "type": FieldType.INTEGER,
        "values": [{"id": v, "name": str(v)} for v in [0, 1, 2]]
    },
    "is_yaoi": {
        "name": "is_yaoi",
        "description": "Indicates if the content is yaoi",
        "type": FieldType.INTEGER,
        "values": [{"id": v, "name": str(v)} for v in [0, 1]]
    },
    "is_erotic": {
        "name": "is_erotic",
        "description": "Indicates if the content is 18+",
        "type": FieldType.INTEGER,
        "values": [{"id": v, "name": str(v)} for v in [0, 1]]
    },
    "is_legal": {
        "name": "is_legal",
        "description": "Indicates if the content is legally available",
        "type": FieldType.INTEGER,
        "values": [{"id": v, "name": str(v)} for v in [0, 1]]
    },
    "is_uploaded": {
        "name": "is_uploaded",
        "description": "Indicates if the content has been uploaded",
        "type": FieldType.INTEGER,
        "values": [{"id": v, "name": str(v)} for v in [0, 1]]
    }
}


# Mapping of fields for related tables
RELATED_TABLE_MAPPINGS: Dict[str, Dict[str, FieldMetadata]] = {
    "titles_sites": {
        "site_id": {
            "name": "site_id",
            "description": "Platform where the content is hosted (e.g., remanga, renovels, recomics, reanime, rehentai, neremanga)",
            "type": FieldType.REFERENCE,
            "values": None
        }
    },
    "titles_genres": {
        "genre_id": {
            "name": "genre_id",
            "description": "Genre identifier for the content",
            "type": FieldType.REFERENCE,
            "values": None
        }
    }
}


# List of table relationships in the system
TABLE_RELATIONS: List[TableRelation] = [
    {
        "table_name": "titles_sites",
        "source_table": "titles",
        "target_table": "django_site",
        "source_field": "id",
        "target_field": "id"
    },
    {
        "table_name": "titles_genres",
        "source_table": "titles",
        "target_table": "genres",
        "source_field": "id",
        "target_field": "id"
    }
]