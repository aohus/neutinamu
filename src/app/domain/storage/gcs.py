# import logging
# from typing import BinaryIO
# from google.cloud import storage
# from .base import StorageService
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class GCSStorageService(StorageService):
#     def __init__(self):
#         # GCS 클라이언트 초기화 (환경 변수 GOOGLE_APPLICATION_CREDENTIALS 사용)
#         self.client = storage.Client()
#         self.bucket_name = settings.GCS_BUCKET_NAME
#         self.bucket = self.client.bucket(self.bucket_name)
#         logger.info(f"GCSStorageService initialized for bucket '{self.bucket_name}'")

#     async def save_file(self, file: BinaryIO, path: str) -> str:
#         blob = self.bucket.blob(path)
#         # file은 SpooledTemporaryFile일 수 있으므로 read() 후 업로드
#         content = await file.read()
        
#         # 썸네일 등 작은 파일은 바로 업로드, 큰 파일은 upload_from_file 사용 고려
#         # 여기서는 간단하게 upload_from_string 사용 (Blocking I/O 주의, MVP라 허용)
#         blob.upload_from_string(content, content_type=file.content_type)
        
#         logger.info(f"[GCS] Uploaded to {path}")
#         return path

#     async def delete_file(self, path: str) -> bool:
#         blob = self.bucket.blob(path)
#         if blob.exists():
#             blob.delete()
#             logger.info(f"[GCS] Deleted {path}")
#             return True
#         return False

#     async def move_file(self, source_path: str, dest_path: str) -> str:
#         source_blob = self.bucket.blob(source_path)
#         if not source_blob.exists():
#             raise FileNotFoundError(f"Source file not found: {source_path}")
        
#         self.bucket.rename_blob(source_blob, dest_path)
#         logger.info(f"[GCS] Moved {source_path} to {dest_path}")
#         return dest_path
        
#     def get_url(self, path: str) -> str:
#         # 공개 버킷인 경우 URL 반환, 비공개인 경우 Signed URL 생성 필요
#         # 여기서는 단순히 media link 반환 (프론트에서 접근 가능해야 함)
#         return f"https://storage.googleapis.com/{self.bucket_name}/{path}"
