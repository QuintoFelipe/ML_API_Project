# app/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
