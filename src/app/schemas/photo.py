from typing import Optional
from pydantic import BaseModel


class PhotoMove(BaseModel):
    photo_id: str
    target_cluster_id: str


class PhotoResponse(BaseModel):
    id: str
    job_id: str
    cluster_id: Optional[str] = None
    storage_path: str
    original_filename: str
