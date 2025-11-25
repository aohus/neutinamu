import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    IMAGE_DIR: str = "/Users/aohus/Workspaces/github/neutinamu/assets/set2/work"
    META_OUTPUT_DIR: str = IMAGE_DIR + "_clustered_meta"
    IMG_OUTPUT_DIR: str = IMAGE_DIR + "_clustered_loc"
    REPORT_DIR: str = IMAGE_DIR + "_report"

    # Clustering strategies to apply in order
    CLUSTERERS: List[str] = field(default_factory=lambda: ["location"])

    # Clustering thresholds
    MAX_LOCATION_DIST_M: float = 30.0
    MAX_ALT_DIFF_M: float = 20.0
    MAX_TIME_GAP_SEC: int = 240
    FOCAL_TOLERANCE_35MM: float = 10.0
    DIRECTION_TOL_DEG: float = 20.0

    # Output settings
    THUMB_SIZE: tuple[int, int] = (256, 256)
    MONTAGE_COLS: int = 10

    def setup_output_dirs(self, clean: bool = False):
        """
        Prepares output directories.
        If clean is True, it removes the existing output directory.
        """
        OUTPUT_DIRS = [self.IMG_OUTPUT_DIR, self.META_OUTPUT_DIR]
        if clean:
            for dir_path in OUTPUT_DIRS:
                self.remake_dir(dir_path)

        os.makedirs(self.REPORT_DIR, exist_ok=True)
    
    def remake_dir(self, dir_path: str):
        """Removes and recreates a directory."""
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Removed existing directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
