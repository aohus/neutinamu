import logging
import os
import shutil
from pathlib import Path
from typing import BinaryIO

import aiofiles
import aioshutil
from app.core.config import settings

from .base import StorageService

logger = logging.getLogger(__name__)


class LocalStorageService(StorageService):
    """Implementation of StorageService for local filesystem."""

    def __init__(self):
        # LocalConfig나 settings에서 MEDIA_ROOT를 가져옴
        # settings.MEDIA_ROOT가 "/app/src/assets" 같은 형태라고 가정
        self.media_root = Path(settings.MEDIA_ROOT)
        self.media_root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"LocalStorageService initialized with base path {self.media_root}")

    def list_image_paths(self, job_id) -> list[str]:
        image_dir = self.media_root / job_id
        logger.debug(f"Listing image paths from: {image_dir}")
        if not image_dir.exists():
            logger.error(f"Image directory does not exist: {image_dir}")
            return []
        return [
            str(image_dir / fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(("png", "jpg", "jpeg"))
        ]
    
    async def save_file(self, file: BinaryIO, path: str, content_type: str = None) -> str:
        full_path = self.media_root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Saving file to local storage: {full_path}")
        async with aiofiles.open(full_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        logger.info(f"Successfully saved file: {full_path}")
        return path

    async def delete_file(self, path: str) -> bool:
        full_path = self.media_root / path
        logger.debug(f"Deleting file from local storage: {full_path}")
        if full_path.exists():
            os.remove(full_path)
            logger.info(f"Successfully deleted file: {full_path}")
            return True
        logger.warning(f"File not found for deletion: {full_path}")
        return False

    async def move_file(self, source_path: str, dest_path: str) -> str:
        src = self.media_root / source_path
        dst = self.media_root / dest_path
        
        logger.debug(f"Moving file from {src} to {dst}")
        if not src.exists():
            logger.error(f"Source file not found for move: {src}")
            raise FileNotFoundError(f"Source file not found: {source_path}")
            
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        await aioshutil.move(str(src), str(dst))
        logger.info(f"Successfully moved file to: {dst}")
        return dest_path

    def get_url(self, path: str) -> str:
        # 로컬 개발 환경에서는 보통 static file serving을 통해 접근
        # 예: http://localhost:8000/assets/path/to/file.jpg
        # settings.MEDIA_URL이 "/assets" 라고 가정
        url = f"{settings.MEDIA_URL}/{path}".replace("//", "/")
        logger.debug(f"Generating URL for local path: {path} -> {url}")
        return url
