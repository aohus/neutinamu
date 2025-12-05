import logging
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO, Optional

from app.core.config import settings
from google.cloud import storage

from .base import StorageService

logger = logging.getLogger(__name__)


class GCSStorageService(StorageService):
    def __init__(self):
        # GCS 클라이언트 초기화 (환경 변수 GOOGLE_APPLICATION_CREDENTIALS 사용)
        self.client = storage.Client()
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)
        logger.info(f"GCSStorageService initialized for bucket '{self.bucket_name}'")

    async def save_file(self, file: BinaryIO, path: str, content_type: str = None) -> str:
        blob = self.bucket.blob(path)
        # file은 SpooledTemporaryFile일 수 있으므로 read() 후 업로드
        content = await file.read()
        
        # 썸네일 등 작은 파일은 바로 업로드, 큰 파일은 upload_from_file 사용 고려
        # 여기서는 간단하게 upload_from_string 사용 (Blocking I/O 주의, MVP라 허용)
        blob.upload_from_string(content, content_type=content_type)
        
        logger.info(f"[GCS] Uploaded to {path}")
        return path

    async def delete_file(self, path: str) -> bool:
        blob = self.bucket.blob(path)
        if blob.exists():
            blob.delete()
            logger.info(f"[GCS] Deleted {path}")
            return True
        return False

    async def move_file(self, source_path: str, dest_path: str) -> str:
        source_blob = self.bucket.blob(source_path)
        if not source_blob.exists():
            # GCS는 폴더 개념이 없으므로, source_path가 폴더인 경우(prefix 검색) 고려해야 할 수도 있음.
            # 하지만 여기서는 단일 파일 이동으로 가정
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        self.bucket.rename_blob(source_blob, dest_path)
        logger.info(f"[GCS] Moved {source_path} to {dest_path}")
        return dest_path
        
    def get_url(self, path: str) -> str:
        # If the path is already a full URL, return it as is to prevent duplication.
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"https://storage.googleapis.com/{self.bucket_name}/{path}"

    def generate_upload_url(self, path: str, content_type: str = None) -> Optional[str]:
        """Generates a v4 signed URL for uploading a blob using HTTP PUT."""
        blob = self.bucket.blob(path)
        
        url = blob.generate_signed_url(
            version="v4",
            # This URL is valid for 15 minutes
            expiration=timedelta(minutes=15),
            # Allow PUT requests using this URL.
            method="PUT",
            content_type=content_type,
        )
        return url

    async def download_file(self, path: str, destination_local_path: Path):
        """Downloads a file from GCS to a local path."""
        blob = self.bucket.blob(path)
        blob.download_to_filename(str(destination_local_path))
        logger.info(f"[GCS] Downloaded {path} to {destination_local_path}")