from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class InteractionMessage(BaseModel):
    """Модель сообщения о взаимодействии пользователя с элементом"""
    user_id: int
    item_id: int
    interaction_type: str = "watch"  # watch, like, etc.
    weight: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "item_id": 456,
                "interaction_type": "watch",
                "weight": 1.0,
                "timestamp": "2023-05-15T14:30:00Z",
                "metadata": {"session_id": "abc123", "device": "mobile"}
            }
        } 