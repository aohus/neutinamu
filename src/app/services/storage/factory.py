from functools import lru_cache
from app.core.config import settings
from .base import StorageService
from .local import LocalStorageService
from .s3 import S3StorageService

class StorageFactory:
    @staticmethod
    def get_storage_service(service_type: str = "local") -> StorageService:
        if service_type == "local":
            return LocalStorageService()
        elif service_type == "s3":
            return S3StorageService()
        else:
            raise ValueError(f"Unknown storage service type: {service_type}")

@lru_cache()
def get_storage_client() -> StorageService:
    # In the future, this could read from settings.STORAGE_TYPE
    return StorageFactory.get_storage_service("local")