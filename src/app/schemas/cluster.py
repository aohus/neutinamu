from typing import List, Optional

from app.schemas.photo import PhotoResponse
from pydantic import BaseModel


class ClusterBase(BaseModel):
    name: Optional[str] = None
    order_index: Optional[int] = 0


class ClusterCreateRequest(ClusterBase):
    name: Optional[str] = None
    order_index: Optional[int] = 0
    photo_ids: Optional[list] = []


class ClusterUpdateRequest(ClusterBase):
    new_name: Optional[str] = None


class ClusterResponse(ClusterBase):
    id: str
    job_id: str
    name: str
    order_index: int
    photos: List[PhotoResponse] = []

    class Config:
        from_attributes = True