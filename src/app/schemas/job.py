from datetime import datetime
from typing import Optional

from app.models.job import Status
from pydantic import BaseModel


class JobRequest(BaseModel):
    title: str


class JobClusterRequest(BaseModel):
    min_samples: Optional[int] = 3
    max_dist_m: Optional[float] = 10.0
    max_alt_diff_m: Optional[float] = 20.0


class JobResponse(BaseModel):
    id: str
    title: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PhotoUploadResponse(BaseModel):
    job_id: str
    file_count: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ExportStatusOut(BaseModel):
    status: Status
    pdf_url: Optional[str] = None
    error_message: Optional[str] = None
