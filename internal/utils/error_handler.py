from typing import Optional, Dict, Any

from fastapi import HTTPException

from internal.types import ErrorResponse


class BaseError(Exception):
    """Базовый класс для всех ошибок приложения"""
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ValidationError(BaseError):
    """Ошибка валидации данных"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class NotFoundError(BaseError):
    """Ошибка - ресурс не найден"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details=details
        )


class ExternalServiceError(BaseError):
    """Ошибка внешнего сервиса"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details=details
        )


def handle_error(error: BaseError) -> ErrorResponse:
    """Преобразование ошибки в стандартизированный ответ"""
    return ErrorResponse(
        error_code=error.error_code,
        message=error.message,
        details=error.details
    )


def to_http_exception(error: BaseError) -> HTTPException:
    """Преобразование ошибки в HTTP исключение"""
    return HTTPException(
        status_code=error.status_code,
        detail=handle_error(error).dict()
    ) 