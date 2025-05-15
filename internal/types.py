from typing import TypedDict, List, Optional, Union
from datetime import datetime
from enum import Enum


class FieldFilter(TypedDict):
    """
    Represents a filter configuration for a single field.
    Used to define filtering conditions in the recommendation system.
    """
    field_name: str  # Name of the field to filter on
    operator: str    # Comparison operator (e.g., 'equals', 'greater_than')
    values: List[Union[str, int, float, bool]]  # List of values to compare against


class RelatedTableFilter(TypedDict):
    """
    Represents a filter configuration for a related table.
    Used to filter recommendations based on related table data.
    """
    table_name: str  # Name of the related table
    field_name: str  # Name of the field in the related table
    operator: str    # Comparison operator
    values: List[Union[str, int, float, bool]]  # List of values to compare against


class ScheduleConfig(TypedDict):
    """
    Configuration for scheduling recommendation updates.
    Defines when and how often recommendations should be updated.
    """
    type: str  # Type of schedule (e.g., 'once_day', 'once_year')
    date_like: str  # Time specification (e.g., '04:00' for daily at 4 AM)
    is_active: bool  # Whether this schedule is currently active
    next_run: Optional[datetime]  # When this schedule will next run


class RecommendationConfig(TypedDict):
    """
    Main configuration for the recommendation system.
    Contains all settings and filters that define how recommendations are generated.
    """
    name: str  # Unique name for this configuration
    description: str  # Human-readable description of what this configuration does
    is_active: bool  # Whether this configuration is currently active
    title_field_filters: List[FieldFilter]  # Filters for title fields
    related_table_filters: List[RelatedTableFilter]  # Filters for related tables
    schedules_dates: List[ScheduleConfig]  # Schedule configurations
    created_at: datetime  # When this configuration was created
    updated_at: datetime  # When this configuration was last updated


class FieldMetadata(TypedDict):
    """
    Metadata about a field in the system.
    Used to describe fields and their possible values.
    """
    name: str  # Name of the field
    description: str  # Human-readable description of the field
    type: str  # Data type of the field
    values: Optional[List[dict]]  # Possible values for enum/reference fields


class RelatedTableMetadata(TypedDict):
    """
    Metadata about a related table in the system.
    Used to describe relationships between tables.
    """
    name: str  # Name of the related table
    description: str  # Human-readable description of the relationship
    fields: dict[str, FieldMetadata]  # Fields available in the related table


class FieldOptions(TypedDict):
    """
    Available options for field configuration in the admin panel.
    Contains all possible values and settings for fields.
    """
    title_fields: dict[str, FieldMetadata]  # Available fields in the titles table
    related_tables: dict[str, RelatedTableMetadata]  # Available related tables
    sites: dict[str, List[Union[int, str]]]  # Available sites and their mappings
    operators: List[str]  # Available comparison operators
    schedule_types: List[dict[str, str]]  # Available schedule types and their descriptions


class ConfigWithMetadata(TypedDict):
    """
    Complete configuration with all associated metadata.
    Used when retrieving a configuration with all its related information.
    """
    config: RecommendationConfig  # The actual configuration
    metadata: FieldOptions  # All available options and metadata 