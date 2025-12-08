import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO, Optional

from app.core.config import settings
from google.cloud import storage  # 동기 라이브러리

from .base import StorageService

logger = logging.getLogger(__name__)


class GCSStorageService(StorageService):
    def __init__(self):
        self.client = storage.Client()
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(f"GCSStorageService initialized for bucket '{self.bucket_name}'")

    async def save_file(self, file: BinaryIO, path: str, content_type: str = None) -> str:
        blob = self.bucket.blob(path)
        
        # file to memory (file 객체가 비동기 read()를 지원해야 함)
        content = await file.read()
        
        def upload_sync():
            blob.upload_from_string(content, content_type=content_type)
            logger.info(f"[GCS] Uploaded to {path}")
            return path
        return await asyncio.to_thread(upload_sync)

    async def delete_file(self, path: str) -> bool:
        blob = self.bucket.blob(path)
        
        def delete_sync():
            if blob.exists():
                blob.delete()
                logger.info(f"[GCS] Deleted {path}")
                return True
            return False

        return await asyncio.to_thread(delete_sync)

    async def delete_directory(self, prefix: str) -> bool:
        if not prefix.endswith('/'):
            prefix += '/'

        def delete_directory_sync():
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                logger.info(f"[GCS] Directory prefix '{prefix}' not found or is empty.")
                return False

            self.bucket.delete_blobs(blobs)
            deleted_names = [blob.name for blob in blobs]
            logger.info(f"[GCS] Deleted directory content ({len(deleted_names)} files) for prefix: {prefix}")
            return True

        return await asyncio.to_thread(delete_directory_sync)

    async def move_file(self, source_path: str, dest_path: str) -> str:
        source_blob = self.bucket.blob(source_path)

        def move_sync():
            if not source_blob.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            self.bucket.rename_blob(source_blob, dest_path)
            logger.info(f"[GCS] Moved {source_path} to {dest_path}")
            return dest_path

        return await asyncio.to_thread(move_sync)
        
    def get_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"https://storage.googleapis.com/{self.bucket_name}/{path}"

    def generate_upload_url(self, path: str, content_type: str = None) -> Optional[str]:
        blob = self.bucket.blob(path)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="PUT",
            content_type=content_type,
        )
        return url

    def generate_resumable_session_url(self, target_path: str, content_type: str, origin: str = None) -> str:
        """
        GCS Resumable Upload Session을 시작하고, 
        클라이언트가 업로드를 수행할 수 있는 Session URL을 반환합니다.
        """
        blob = self.bucket.blob(target_path)
        
        # 세션 시작 (백엔드에서 GCS로 요청을 보냄)
        # origin 인자는 CORS Preflight를 위해 클라이언트의 도메인을 넣어주는 것이 좋습니다.
        upload_url = blob.create_resumable_upload_session(
            content_type=content_type,
            origin=origin 
        )
        
        return upload_url
    
    async def download_file(self, path: str, destination_local_path: Path):
        item = path.replace(f"https://storage.googleapis.com/{self.bucket_name}/", "")
        blob = self.bucket.blob(item)
        
        def download_sync():
            blob.download_to_filename(str(destination_local_path))
            logger.info(f"[GCS] Downloaded {path} to {destination_local_path}")

        await asyncio.to_thread(download_sync)