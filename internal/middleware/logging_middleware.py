import logging
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Логируем начало запроса
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        logger.info(f"Request started: {request.method} {request.url.path} [ID: {request_id}]")
        logger.debug(f"Request headers: {dict(request.headers)}")
        
        try:
            # Получаем тело запроса для POST/PUT запросов
            if request.method in ["POST", "PUT"]:
                body = await request.body()
                if body:
                    logger.debug(f"Request body: {body.decode()}")
            
            # Выполняем запрос
            response = await call_next(request)
            
            # Логируем ответ
            process_time = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"[ID: {request_id}] - Status: {response.status_code} "
                f"- Time: {process_time:.2f}s"
            )
            
            return response
            
        except Exception as e:
            # Логируем ошибку
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"[ID: {request_id}] - Error: {str(e)} "
                f"- Time: {process_time:.2f}s"
            )
            raise 
 