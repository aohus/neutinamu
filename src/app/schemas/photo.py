from typing import List, Optional

from pydantic import BaseModel


class PhotoMove(BaseModel):
    target_cluster_id: str


class PhotoResponse(BaseModel):
    id: str
    job_id: str
    order_index: Optional[int] = 0
    cluster_id: Optional[str] = None
    storage_path: str
    original_filename: str


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
    strategy: str  # "direct" or "proxy"
    urls: List[PresignedUrlResponse]
