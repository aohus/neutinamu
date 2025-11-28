import logging
import os
from pathlib import Path
from typing import BinaryIO

import aiofiles
import aioshutil

from .base import StorageService

logger = logging.getLogger(__name__)


class LocalStorageService(StorageService):
    """Implementation of StorageService for local filesystem."""

    def __init__(self, config):
        self.config = config
        logger.debug(f"LocalStorageService initialized for job {config.job_id} with base path {config.IMAGE_DIR}")

    def list_image_paths(self) -> list[str]:
        image_dir = self.config.IMAGE_DIR
        logger.debug(f"Listing image paths from: {image_dir}")
        if not image_dir.exists():
            logger.error(f"Image directory does not exist: {image_dir}")
            return []
        return [
            str(image_dir / fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(("png", "jpg", "jpeg"))
        ]

    async def save_file(self, file: BinaryIO) -> str:
        img_path = self.config.IMAGE_DIR / file.filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Saving file to local storage: {img_path}")
        async with aiofiles.open(img_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        logger.info(f"Successfully saved file: {img_path}")
        return img_path

    async def delete_file(self, path: str) -> bool:
        full_path = self.base_path / path
        logger.debug(f"Deleting file from local storage: {full_path}")
        if full_path.exists():
            os.remove(full_path)
            logger.info(f"Successfully deleted file: {full_path}")
            return True
        logger.warning(f"File not found for deletion: {full_path}")
        return False

    async def move_file(self, source_path: str, dest_path: str) -> str:
        src = self.base_path / source_path
        dst = self.base_path / dest_path
        
        logger.debug(f"Moving file from {src} to {dst}")
        if not src.exists():
            logger.error(f"Source file not found for move: {src}")
            raise FileNotFoundError(f"Source file not found: {source_path}")
            
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        await aioshutil.move(str(src), str(dst))
        logger.info(f"Successfully moved file to: {dst}")
        return str(dest_path)
        
    def get_url(self, path: str) -> str:
        logger.debug(f"Generating URL for local path: {path}")
        return path