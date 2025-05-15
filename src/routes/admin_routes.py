from fastapi import APIRouter, HTTPException, Depends
from typing import List
from internal.repositories import ConfigRepository
from src.services.configuration_service import AdminPanelService
from internal.types import (
    FieldOptions,
    RecommendationConfig,
    ConfigWithMetadata
)

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/field-options")
async def get_field_options() -> FieldOptions:
    """
    Get all available field options for the admin panel.
    Returns metadata about fields, related tables, sites, and available operators.
    """
    return AdminPanelService.get_field_options()

@router.get("/configs", response_model=List[RecommendationConfig])
async def get_configs() -> List[RecommendationConfig]:
    """
    Get a list of all recommendation configurations.
    Returns basic information about each configuration without detailed metadata.
    """
    configs = ConfigRepository.get_all()
    result = []
    for config in configs:
        config_dict = config.to_mongo().to_dict()
        # Convert ObjectId to string for JSON serialization
        if '_id' in config_dict:
            config_dict['_id'] = str(config_dict['_id'])
        result.append(config_dict)
    return result

@router.get("/configs/{config_id}", response_model=ConfigWithMetadata)
async def get_config(config_id: str) -> ConfigWithMetadata:
    """
    Get a specific recommendation configuration with all its metadata.
    Returns detailed information about the configuration including all available options.
    """
    config_data = AdminPanelService.get_config_with_metadata(config_id)
    if not config_data:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return config_data

@router.post("/configs", response_model=RecommendationConfig)
async def create_config(config_data: RecommendationConfig) -> RecommendationConfig:
    """
    Create a new recommendation configuration.
    Validates the input data and creates a new configuration in the system.
    """
    config = ConfigRepository.create(config_data)
    config_dict = config.to_mongo().to_dict()
    if '_id' in config_dict:
        config_dict['_id'] = str(config_dict['_id'])
    return config_dict

@router.put("/configs/{config_id}", response_model=RecommendationConfig)
async def update_config(config_id: str, config_data: RecommendationConfig) -> RecommendationConfig:
    """
    Update an existing recommendation configuration.
    Validates the input data and updates the specified configuration.
    """
    config = ConfigRepository.update(config_id, config_data)
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    config_dict = config.to_mongo().to_dict()
    if '_id' in config_dict:
        config_dict['_id'] = str(config_dict['_id'])
    return config_dict

@router.delete("/configs/{config_id}")
async def delete_config(config_id: str) -> dict[str, bool]:
    """
    Delete a recommendation configuration.
    Removes the specified configuration from the system.
    """
    success = ConfigRepository.delete(config_id)
    if not success:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return {"success": True} 