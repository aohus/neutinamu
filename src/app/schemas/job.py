from datetime import datetime
from typing import List, Optional

from app.schemas.enum import ExportStatus, JobStatus
from pydantic import BaseModel


class JobRequest(BaseModel):
    title: str
    contractor_name: Optional[str] = None
    work_date: Optional[datetime] = None


class JobClusterRequest(BaseModel):
    min_samples: Optional[int] = 3
    max_dist_m: Optional[float] = 12.0
    max_alt_diff_m: Optional[float] = 20.0


class JobResponse(BaseModel):
    id: str
    title: Optional[str] = None
    status: JobStatus
    contractor_name: Optional[str] = None
    work_date: Optional[datetime] = None
    export_status: Optional[ExportStatus] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PhotoResponse(BaseModel):
    id: str
    original_filename: str
    storage_path: str
    url: Optional[str] = None
    thumbnail_path: Optional[str] = None
    cluster_id: Optional[str] = None
    order_index: Optional[int] = None
    
    class Config:
        from_attributes = True


class ClusterResponse(BaseModel):
    id: str
    name: str
    order_index: int
    photos: List[PhotoResponse] = []

    class Config:
        from_attributes = True


class JobDetailsResponse(JobResponse):
    photos: List[PhotoResponse] = []
    clusters: List[ClusterResponse] = []


class PhotoUploadResponse(BaseModel):
    job_id: str
    file_count: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ExportStatusResponse(BaseModel):
    status: ExportStatus
    pdf_url: Optional[str] = None
    error_message: Optional[str] = None
