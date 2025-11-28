from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class PhotoBase(BaseModel):
    original_filename: str

class PhotoResponse(PhotoBase):
    id: str
    job_id: str
    cluster_id: Optional[str] = None
    storage_path: str
    thumbnail_path: Optional[str] = None
    
    class Config:
        from_attributes = True

class ClusterBase(BaseModel):
    name: Optional[str] = None
    order_index: Optional[int] = 0

class ClusterCreate(ClusterBase):
    name: str

class ClusterUpdate(ClusterBase):
    pass

class ClusterDetail(ClusterBase):
    id: str
    job_id: str
    name: str
    order_index: int
    photos: List[PhotoResponse] = []
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class JobCreate(BaseModel):
    pass

class JobResponse(BaseModel):
    id: str
    title: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class ClusterRename(BaseModel):
    new_name: str

class PhotoMove(BaseModel):
    photo_id: str
    target_cluster_id: str

class JobRequest(BaseModel):
    title: str