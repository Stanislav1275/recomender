"""
Конфигурационный файл для рекомендательного сервиса.
Содержит параметры моделей и настройки сервиса.
"""

import os
from typing import Dict, Any

# Параметры модели LightFM
MODEL_CONFIG: Dict[str, Any] = {
    # Размерность эмбеддингов
    "no_components": 100,
    
    # Функция потерь (bpr, warp, logistic)
    "loss": "bpr",
    
    # Число потоков для обучения
    "num_threads": 3,
    
    # Количество эпох обучения
    "epochs": 30,
    
    # L2 регуляризация пользовательских факторов
    "user_alpha": 0.0001,
    
    # L2 регуляризация факторов элементов
    "item_alpha": 0.0001,
    
    # Фактор обучения
    "learning_rate": 0.05,
    
    # Начальное значение генератора случайных чисел
    "random_state": 60,
}

# Настройки обучения модели
TRAINING_CONFIG = {
    # Обновление модели каждые N часов
    "update_frequency_hours": 24,
    
    # Максимальное количество взаимодействий для обучения
    "max_interactions": 10000000,
    
    # Фактор временного затухания для старых взаимодействий
    "time_decay_factor": 0.01,
    
    # Максимальный коэффициент затухания
    "max_decay": 0.7,
}

# Настройки gRPC сервера
GRPC_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("GRPC_PORT", "50051")),
    "max_workers": int(os.getenv("GRPC_MAX_WORKERS", "10")),
}

# Настройки API
API_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("DEBUG", "False").lower() in ("true", "1", "t"),
}

class Config:
    MODEL_PARAMS = {
        'default': {
            'no_components': 64,
            'loss': 'warp',
            'learning_rate': 0.05
        },
        'fallback': {
            'no_components': 32,
            'loss': 'bpr'
        }
    }

    DATA_SETTINGS = {
        'train_days': 60,
        'test_days': 7,
        'time_decay_rate': 0.9
    }