from typing import Optional

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
