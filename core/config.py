from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    RE_DB_HOST: str = ""
    RE_DB: str = ""
    RE_DB_USER: str = ""
    RE_DB_PASSWORD: str = ""
    RE_DB_PORT: str = "3306"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings():
    return Settings()