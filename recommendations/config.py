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