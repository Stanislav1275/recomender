from datetime import datetime

# TODO: Возможно стоит вынести в БД и добавить возможность
# настройки через админку. Сейчас хардкод, так как значения
# меняются редко и требуют перезапуска сервиса
# Используется для фильтрации контента по возрастным ограничениям в рекомендациях
# и для формирования black/white листов при обучении модели
AGE_LIMITS = {
    0: "Без ограничений",
    1: "16+",
    2: "18+"
}

# TODO: Добавить поддержку сложных операторов (AND, OR)
# и возможность создания пользовательских операторов
# Операторы для формирования условий фильтрации при обучении модели
# и создания правил для рекомендаций. Используются в SQL-подобных запросах
# для формирования условий WHERE
FILTER_OPERATORS = {
    "equals": "Равно",
    "not_equals": "Не равно",
    "in": "В списке",
    "not_in": "Не в списке",
    "greater_than": "Больше",
    "less_than": "Меньше"
}

# TODO: Добавить валидацию входных данных и обработку ошибок
# Функции для приведения типов данных при обработке входных параметров
# и валидации данных из внешних источников (API, БД, файлы)
TRANSFORM_FUNCTIONS = {
    'to_int': lambda x: int(x) if x is not None else None,
    'to_float': lambda x: float(x) if x is not None else None,
    'to_bool': lambda x: bool(x) if x is not None else None,
    'to_list': lambda x: x.split(',') if isinstance(x, str) else x,
    'to_datetime': lambda x: datetime.fromisoformat(x) if x is not None else None
}

FIELD_OPTIONS = {
    'is_erotic': {
        'description': 'Эротический контент',
        'type': 'boolean',
        'operators': ['equals', 'not_equals']
    },
    'is_legal': {
        'description': 'Легальный контент',
        'type': 'boolean',
        'operators': ['equals', 'not_equals']
    },
    'is_licensed': {
        'description': 'Лицензированный контент',
        'type': 'boolean',
        'operators': ['equals', 'not_equals']
    },
    'is_yaoi': {
        'description': 'Яой контент',
        'type': 'boolean',
        'operators': ['equals', 'not_equals']
    },
    'uploaded': {
        'description': 'Залито',
        'type': 'boolean',
        'operators': ['equals', 'not_equals']
    },
    'age_limit': {
        'description': 'Возрастное ограничение',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'greater_than', 'less_than', 'in', 'not_in']
    },
    'issue_year': {
        'description': 'Год выпуска',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'greater_than', 'less_than', 'in', 'not_in']
    },
    'total_views': {
        'description': 'Общее количество просмотров',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'greater_than', 'less_than', 'in', 'not_in']
    },
    'total_votes': {
        'description': 'Общее количество голосов',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'greater_than', 'less_than', 'in', 'not_in']
    },
    'avg_rating': {
        'description': 'Средний рейтинг',
        'type': 'float',
        'operators': ['equals', 'not_equals', 'greater_than', 'less_than', 'in', 'not_in']
    },
    'site_id': {
        'description': 'ID сайта',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'in', 'not_in']
    },
    'status_id': {
        'description': 'Статус тайтла',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'in', 'not_in']
    },
    'type_id': {
        'description': 'Тип тайтла',
        'type': 'integer',
        'operators': ['equals', 'not_equals', 'in', 'not_in']
    }
}

VALID_OPERATORS = {
    'equals': 'Равно',
    'not_equals': 'Не равно',
    'greater_than': 'Больше',
    'less_than': 'Меньше',
    'in': 'В списке',
    'not_in': 'Не в списке'
} 