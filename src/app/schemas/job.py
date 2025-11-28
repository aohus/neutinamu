from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class JobCreate(BaseModel):
    pass


class JobRequest(BaseModel):
    title: str


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
