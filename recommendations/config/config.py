from rectools.models.lightfm import LightFMWrapperModelConfig

config = {
    "model": {
        "cls": "lightfm.lightfm.LightFM",
        # "cls": "LightFM",
        "no_components": 32,
        "learning_rate": 0.05,
        "random_state": 42,
        "loss": "warp",
        "k": 40
    },
    "num_threads": 4,
    "epochs": 15,
}