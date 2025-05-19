from typing import List

from fastapi import APIRouter, Depends, HTTPException

from internal.config.mongo_adapter import get_database
from internal.models import RecommendationConfig, ConfigExecutionLog
from internal.repositories import RecommendationConfigRepository
from internal.services import RecommendationConfigService
from internal.types import ConfigResponse, ErrorResponse, FieldOptions
from external_db.data_service import ExternalDataService

router = APIRouter(prefix="/api/admin/configs", tags=["admin"])


async def get_config_service() -> RecommendationConfigService:
    """Получить сервис конфигураций"""
    db = await get_database()
    repository = RecommendationConfigRepository(db)
    external_service = ExternalDataService()
    return RecommendationConfigService(repository, external_service)


@router.get(
    "/field-options",
    response_model=FieldOptions,
    responses={
        200: {"description": "Опции полей для админ-панели"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def get_field_options(
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Получить опции полей для админ-панели"""
    try:
        return await service.get_field_options()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        )


@router.get(
    "/",
    response_model=List[ConfigResponse],
    responses={
        200: {"description": "Список конфигураций"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def get_configs(
    skip: int = 0,
    limit: int = 100,
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Получить список конфигураций"""
    try:
        return await service.get_configs(skip, limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        )


@router.get(
    "/{config_id}",
    response_model=ConfigResponse,
    responses={
        200: {"description": "Конфигурация найдена"},
        404: {"model": ErrorResponse, "description": "Конфигурация не найдена"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def get_config(
    config_id: str,
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Получить конфигурацию по ID"""
    try:
        config = await service.get_config(config_id)
        if not config:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error_code="NOT_FOUND",
                    message="Configuration not found",
                    details={"config_id": config_id}
                ).dict()
            )
        return config
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        )


@router.post(
    "/",
    response_model=ConfigResponse,
    responses={
        201: {"description": "Конфигурация создана"},
        400: {"model": ErrorResponse, "description": "Неверные данные"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def create_config(
    config: RecommendationConfig,
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Создать новую конфигурацию"""
    try:
        return await service.create_config(config)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message=str(e)
            ).dict()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        )


@router.put(
    "/{config_id}",
    response_model=ConfigResponse,
    responses={
        200: {"description": "Конфигурация обновлена"},
        404: {"model": ErrorResponse, "description": "Конфигурация не найдена"},
        400: {"model": ErrorResponse, "description": "Неверные данные"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def update_config(
    config_id: str,
    config: RecommendationConfig,
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Обновить конфигурацию"""
    try:
        # Проверяем существование конфигурации
        existing_config = await service.get_config(config_id)
        if not existing_config:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error_code="NOT_FOUND",
                    message="Configuration not found",
                    details={"config_id": config_id}
                ).dict()
            )
        return await service.update_config(config_id, config)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="VALIDATION_ERROR",
                message=str(e)
            ).dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        )


@router.delete(
    "/{config_id}",
    response_model=bool,
    responses={
        200: {"description": "Конфигурация удалена"},
        404: {"model": ErrorResponse, "description": "Конфигурация не найдена"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def delete_config(
    config_id: str,
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Удалить конфигурацию"""
    try:
        # Проверяем существование конфигурации
        existing_config = await service.get_config(config_id)
        if not existing_config:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error_code="NOT_FOUND",
                    message="Configuration not found",
                    details={"config_id": config_id}
                ).dict()
            )
        return await service.delete_config(config_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        )


@router.get(
    "/{config_id}/logs",
    response_model=List[ConfigExecutionLog],
    responses={
        200: {"description": "Логи конфигурации"},
        404: {"model": ErrorResponse, "description": "Конфигурация не найдена"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"}
    }
)
async def get_config_logs(
    config_id: str,
    skip: int = 0,
    limit: int = 100,
    service: RecommendationConfigService = Depends(get_config_service)
):
    """Получить логи выполнения конфигурации"""
    try:
        return await service.get_config_logs(config_id, skip, limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message=str(e)
            ).dict()
        ) 