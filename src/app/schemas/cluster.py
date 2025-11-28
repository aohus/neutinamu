from typing import List, Optional

from pydantic import BaseModel

from app.schemas.photo import PhotoResponse



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

    class Config:
        from_attributes = True

class ClusterRename(BaseModel):
    new_name: str