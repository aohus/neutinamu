from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class PhotoMove(BaseModel):
    target_cluster_id: str
    order_index: Optional[int] = None


class PhotoUpdate(BaseModel):
    labels: Optional[dict] = None


class PhotoResponse(BaseModel):
    id: str
    job_id: str
    order_index: Optional[int] = 0
    cluster_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    original_filename: str
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    storage_path: str
    thumbnail_path: Optional[str] = None
    labels: Optional[dict] = {}


class PhotoUploadRequest(BaseModel):
    filename: str
    content_type: str


class PhotoCompleteRequest(BaseModel):
    filename: str
    storage_path: str


class PresignedUrlResponse(BaseModel):
    filename: str
    upload_url: Optional[str]
    storage_path: str


class BatchPresignedUrlResponse(BaseModel):
    strategy: str  # "presigned" or "proxy"
    urls: List[PresignedUrlResponse]
