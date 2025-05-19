import traceback
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from internal.utils.error_handler import BaseError, handle_error

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def error_handler_middleware(request: Request, call_next):
    """Middleware для обработки ошибок"""
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    try:
        return await call_next(request)
    except BaseError as e:
        # Логируем известные ошибки
        logger.error(
            f"Known error occurred [ID: {request_id}]:\n"
            f"Method: {request.method}\n"
            f"Path: {request.url.path}\n"
            f"Error: {str(e)}\n"
            f"Details: {e.details}\n"
            f"Status: {e.status_code}"
        )
        return JSONResponse(
            status_code=e.status_code,
            content=handle_error(e).dict()
        )
    except Exception as e:
        # Подробное логирование неожиданных ошибок
        error_traceback = traceback.format_exc()
        logger.error(
            f"Unexpected error occurred [ID: {request_id}]:\n"
            f"Method: {request.method}\n"
            f"Path: {request.url.path}\n"
            f"Error: {str(e)}\n"
            f"Traceback:\n{error_traceback}"
        )
        
        # Обработка неожиданных ошибок
        error = BaseError(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            status_code=500,
            details={
                "error": str(e),
                "traceback": error_traceback,
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path
            }
        )
        return JSONResponse(
            status_code=error.status_code,
            content=handle_error(error).dict()
        ) 