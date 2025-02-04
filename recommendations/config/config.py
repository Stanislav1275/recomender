from rectools.models.lightfm import LightFMWrapperModelConfig

config = {
    "model": {
        "cls": "lightfm.lightfm.LightFM",
        # "cls": "LightFM",
        "no_components": 16,
        "learning_rate": 0.03,
        "random_state": 40,
        "loss": "bpr",
        "k":40
    },
    "epochs": 10,
}