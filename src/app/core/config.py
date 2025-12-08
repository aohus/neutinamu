import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Snap To Report Service"
    CLUSTER_SERVICE_URL: str = "http://host.docker.internal:8001"
    CALLBACK_BASE_URL: str = "http://localhost:8000/api"
    FRONTEND_ORIGIN: str = ""

    # Logging configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/report_db"
    FRONTEND_URL: str = "http://localhost:3000"
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5432,http://localhost:3000"
    
    # Storage
    STORAGE_TYPE: str = "local" # local, gcs, s3
    MEDIA_ROOT: str = "/app/assets" # For local storage
    MEDIA_URL: str = "/api/uploads"
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

    # Output settings
    THUMB_SIZE: tuple[int, int] = (256, 256)
    MONTAGE_COLS: int = 10

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# This class will be instantiated per-request, holding session-specific paths.
class JobConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    # JobConfig should rely on settings or storage service, not re-define MEDIA_ROOT generally
    # But if needed for internal logic:
    MEDIA_ROOT = Path(settings.MEDIA_ROOT)

    def __init__(self,    
                 job_id: str,
                 min_samples: int = 3, 
                 max_dist_m: float = 10.0, 
                 max_alt_diff_m: float = 20.0
                 ):
        self.job_id = job_id
        self.CLUSTERERS = ["gps", "image"]

        # Clustering thresholds
        self.MIN_SAMPLES: int = min_samples or settings.MIN_SAMPLES
        self.MAX_LOCATION_DIST_M: float = max_dist_m or settings.MAX_LOCATION_DIST_M
        self.MAX_ALT_DIFF_M: float = max_alt_diff_m or settings.MAX_ALT_DIFF_M
        self.MAX_TIME_GAP_SEC: int = 240
        self.FOCAL_TOLERANCE_35MM: float = 10.0
        self.DIRECTION_TOL_DEG: float = 20.0

        # Output settings
        self.THUMB_SIZE: tuple[int, int] = (256, 256)
        self.MONTAGE_COLS: int = 10