import logging
from typing import List, Optional

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    API_V1_STR: str = "/api/v1"
    APP_NAME: str = "Snap To Report Service"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    WORKER_BASE_URL: str = "http://host.docker.internal:8001/api"
    CALLBACK_BASE_URL: str = "http://host.docker.internal:8000/api/internal"

    # Logging configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/report_db"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5432,http://localhost:3000,http://localhost:9090"

    # Storage
    STORAGE_TYPE: str = "gcs"
    GCS_BUCKET_NAME: Optional[str] = None

    # Auth
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"

    # Celery
    # CELERY_BROKER_URL: str = "redis://redis:6379/0"
    # CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


configs = Config()