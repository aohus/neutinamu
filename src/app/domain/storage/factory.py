import logging
from functools import lru_cache

from app.core.config import settings

from .base import StorageService
from .gcs import GCSStorageService
from .local import LocalStorageService
from .s3 import S3StorageService

logger = logging.getLogger(__name__)


class StorageFactory:
    @staticmethod
    def get_storage_service(service_type: str = "local") -> StorageService:
        logger.info(f"Creating storage service of type: {service_type}")
        if service_type == "local":
            return LocalStorageService()
        elif service_type == "s3":
            return S3StorageService()
        elif service_type == "gcs":
            return GCSStorageService()
        else:
            logger.error(f"Unknown storage service type requested: {service_type}")
            raise ValueError(f"Unknown storage service type: {service_type}")

@lru_cache()
def get_storage_client() -> StorageService:
    # settings.STORAGE_TYPE 환경 변수 사용
    storage_type = getattr(settings, "STORAGE_TYPE", "local")
    logger.debug(f"Getting storage client (cached). Type: {storage_type}")
    return StorageFactory.get_storage_service(storage_type)