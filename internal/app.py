from fastapi import FastAPI

from external_db.data_service import ExternalDataService
from src.routes import admin_routes
from .config.mongo_adapter import get_database
from internal.middleware.error_handler import error_handler_middleware
from internal.middleware.logging_middleware import LoggingMiddleware
from .repositories import RecommendationConfigRepository


def create_app() -> FastAPI:
    """Создание FastAPI приложения"""
    app = FastAPI(
        title="Recommendation System API",
        description="API для системы рекомендаций",
        version="1.0.0"
    )

    # Добавляем middleware для логирования
    app.add_middleware(LoggingMiddleware)
    
    # Добавляем middleware для обработки ошибок
    app.middleware("http")(error_handler_middleware)

    # Инициализация MongoDB
    @app.on_event("startup")
    async def startup_db_client():
        """Инициализация подключения к базе данных при старте приложения"""
        app.mongodb_client = await get_database()
        app.config_repository = RecommendationConfigRepository(app.mongodb_client)
        await app.config_repository.create_indexes()
        
        # Инициализация сервиса внешних данных
        app.external_data_service = ExternalDataService()

    @app.on_event("shutdown")
    async def shutdown_db_client():
        """Закрытие подключения к базе данных при остановке приложения"""
        if hasattr(app, 'mongodb_client'):
            app.mongodb_client.close()

    # Подключаем роуты
    app.include_router(admin_routes.router)

    return app 