import os
from pathlib import Path
from typing import BinaryIO

import aiofiles
import aioshutil

from .base import StorageService


class LocalStorageService(StorageService):
    """Implementation of StorageService for local filesystem."""

    def __init__(self, config):
        self.config = config

    async def save_file(self, file: BinaryIO) -> str:
        img_path = self.config.IMAGE_DIR / file.filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Note: If file is large, we might want to stream copy. 
        # For now, we assume file.read() is acceptable or file is SpooledTemporaryFile.
        # If file is from FastAPI UploadFile, we might need to handle it differently in the caller
        # or accept UploadFile type. Here we stick to BinaryIO interface.
        
        async with aiofiles.open(img_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        return img_path

    async def delete_file(self, path: str) -> bool:
        full_path = self.base_path / path
        if full_path.exists():
            # os.remove is sync, but usually fast. 
            # For strict async, run in executor or use aiofiles.os.remove if available (it's not standard)
            # or just os.remove is fine for now.
            os.remove(full_path)
            return True
        return False

    async def move_file(self, source_path: str, dest_path: str) -> str:
        src = self.base_path / source_path
        dst = self.base_path / dest_path
        
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
            
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Use aioshutil for async move
        await aioshutil.move(str(src), str(dst))
        
        return str(dest_path)
        
    def get_url(self, path: str) -> str:
        # In a real scenario, this might return a static file URL served by nginx or FastAPI
        # For now, we return the relative path which the frontend can use with the base API URL
        return path