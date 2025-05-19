from enum import Enum
from typing import List, Dict, Any, TypedDict

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
