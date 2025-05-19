from typing import Dict, Any, Optional

from internal.types import ExternalDataMapping, FieldMetadata


class ExternalDataMapper:
    """Маппер для преобразования внешних данных в внутренний формат"""
    
    def __init__(self):
        self._mappings: Dict[str, ExternalDataMapping] = {}
        self._transform_functions: Dict[str, callable] = {
            'to_int': lambda x: int(x) if x is not None else None,
            'to_float': lambda x: float(x) if x is not None else None,
            'to_bool': lambda x: bool(x) if x is not None else None,
            'to_list': lambda x: x.split(',') if isinstance(x, str) else x,
            'to_datetime': lambda x: datetime.fromisoformat(x) if x is not None else None
        }

    def add_mapping(self, mapping: ExternalDataMapping) -> None:
        """Добавить маппинг"""
        self._mappings[mapping.field_name] = mapping

    def add_transform_function(self, name: str, func: callable) -> None:
        """Добавить функцию преобразования"""
        self._transform_functions[name] = func

    def transform_value(self, value: Any, transform_func: Optional[str]) -> Any:
        """Применить функцию преобразования к значению"""
        if not transform_func or transform_func not in self._transform_functions:
            return value
        return self._transform_functions[transform_func](value)

    def map_data(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразовать внешние данные во внутренний формат"""
        result = {}
        
        for field_name, mapping in self._mappings.items():
            external_value = external_data.get(mapping.external_field)
            
            if external_value is None:
                result[field_name] = mapping.default_value
                continue
                
            result[field_name] = self.transform_value(
                external_value,
                mapping.transform_function
            )
            
        return result

    def get_field_metadata(self) -> Dict[str, FieldMetadata]:
        """Получить метаданные полей на основе маппингов"""
        return {
            field_name: FieldMetadata(
                name=field_name,
                description=f"Mapped from {mapping.external_field}",
                type="string",  # Базовый тип, может быть переопределен
                default_value=mapping.default_value
            )
            for field_name, mapping in self._mappings.items()
        } 