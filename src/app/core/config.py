import logging
from typing import List, Optional

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Config(BaseSettings):
    API_V1_STR: str = "/api/v1"
    APP_NAME: str = "Snap To Report Service"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    CLUSTER_SERVICE_URL: str = "http://host.docker.internal:8001"
    CALLBACK_BASE_URL: str = "http://localhost:8000/api"
    FRONTEND_ORIGIN: str = ""

    # Logging configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/report_db"
    FRONTEND_URL: str = "http://localhost:3000"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5432,http://localhost:3000,http://localhost:9090"

    # Storage
    STORAGE_TYPE: str = "gcs"  # Options: 'gcs', 'local'
    MEDIA_ROOT: str = "media"
    MEDIA_URL: str = "media"
    GCS_BUCKET_NAME: Optional[str] = None
    PDF_BASE_TEMPLATE_PATH: Optional[str] = None

    # Auth
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"

    # Clustering defaults
    MIN_SAMPLES: int = 3
    MAX_LOCATION_DIST_M: float = 10.0
    MAX_ALT_DIFF_M: float = 20.0

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Output settings
    THUMB_SIZE: tuple[int, int] = (256, 256)
    MONTAGE_COLS: int = 10

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


configs = Config()


# This class will be instantiated per-request, holding session-specific paths.
class JobConfig:
    def __init__(self, job_id: str, min_samples: int = 3, max_dist_m: float = 10.0, max_alt_diff_m: float = 20.0):
        self.job_id = job_id
        self.CLUSTERERS = ["gps", "image"]

        # Clustering thresholds
        self.MIN_SAMPLES: int = min_samples or configs.MIN_SAMPLES
        self.MAX_LOCATION_DIST_M: float = max_dist_m or configs.MAX_LOCATION_DIST_M
        self.MAX_ALT_DIFF_M: float = max_alt_diff_m or configs.MAX_ALT_DIFF_M
        self.MAX_TIME_GAP_SEC: int = 240
        self.FOCAL_TOLERANCE_35MM: float = 10.0
        self.DIRECTION_TOL_DEG: float = 20.0

        # Output settings
        self.THUMB_SIZE: tuple[int, int] = (256, 256)
        self.MONTAGE_COLS: int = 10
