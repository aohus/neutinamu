import logging
import os
import shutil
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class GlobalSettings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Photo Clustering Service"
    
    # Logging configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/report_db"
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Base directory for storing all user data
    DATA_DIR: Path = Path("assets")
    ALLOWED_ORIGINS: str = "http://localhost:5432,http://localhost:3000"
    
    # Default clustering strategies
    CLUSTERERS: List[str] = ["gps"]
    # Clustering thresholds
    MIN_SAMPLES: int = 3
    MAX_LOCATION_DIST_M: float = 10.0
    MAX_ALT_DIFF_M: float = 20.0
    MAX_TIME_GAP_SEC: int = 240
    FOCAL_TOLERANCE_35MM: float = 10.0
    DIRECTION_TOL_DEG: float = 20.0
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SECRET_KEY: str
    ALGORITHM: str

    # Output settings
    THUMB_SIZE: tuple[int, int] = (256, 256)
    MONTAGE_COLS: int = 10

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = GlobalSettings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)


# This class will be instantiated per-request, holding session-specific paths.

class JobConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    MEDIA_ROOT = settings.DATA_DIR

    def __init__(self,    
                 job_id: str,
                 min_samples: int = 3, 
                 max_dist_m: float = 10.0, 
                 max_alt_diff_m: float = 20.0
                 ):
        self.job_id = job_id
        self.base_dir = self.MEDIA_ROOT / job_id
        self.IMAGE_DIR = self.base_dir / "set"
        self.META_OUTPUT_DIR = self.base_dir / "meta"
        self.IMG_OUTPUT_DIR = self.base_dir / "clustered"
        self.REPORT_DIR = self.base_dir / "report"
        
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

    def setup_output_dirs(self, clean: bool = False):
        """
        Prepares output directories.
        If clean is True, it removes the existing output directory.
        """
        OUTPUT_DIRS = [self.IMG_OUTPUT_DIR, self.META_OUTPUT_DIR, self.REPORT_DIR]
        if clean:
            for dir_path in OUTPUT_DIRS:
                self.remake_dir(dir_path)

        for dir_path in OUTPUT_DIRS:
            os.makedirs(dir_path, exist_ok=True)

    def remake_dir(self, dir_path: str):
        """Removes and recreates a directory."""
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Removed existing directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


class LocalConfig(JobConfig):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self,    
                 job_id: str,
                 min_samples: int = 3, 
                 max_dist_m: float = 10.0, 
                 max_alt_diff_m: float = 20.0
                 ):
        self.job_id = job_id
        self.base_dir = self.PROJECT_ROOT / job_id
        self.IMAGE_DIR = self.base_dir / "set"
        self.META_OUTPUT_DIR = self.base_dir / "meta"
        self.IMG_OUTPUT_DIR = self.base_dir / "clustered"
        self.REPORT_DIR = self.base_dir / "report"
        
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
    