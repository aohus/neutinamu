import logging
from typing import List

from app.api.endpoints.auth import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.job import (
    JobRequest,
    JobResponse,
    JobStatusResponse,
    PhotoUploadResponse,
)
from app.services.job import JobService
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs", response_model=list[JobResponse], status_code=status.HTTP_200_OK)
async def get_jobs(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    service = JobService(db)
    jobs = await service.get_jobs(current_user)
    return [
        JobResponse(
            id=job.id, 
            title=job.title, 
            status=job.status, 
            created_at=job.created_at
        )
        for job in jobs
    ]


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: JobRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = JobService(db)
    logger.info(f"User {current_user.user_id} creating job with title: '{payload.title}'")
    job = await service.create_job(user=current_user, title=payload.title)
    return JobResponse(
        id=job.id, title=job.title, status=job.status, created_at=job.created_at
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get job details."""
    service = JobService(db)
    job = await service.get_job(job_id=job_id)
    return JobResponse(
        id=job.id, title=job.title, status=job.status, created_at=job.created_at
    )


@router.post(
    "/jobs/{job_id}/photos",
    summary="Upload photos",
    response_model=PhotoUploadResponse,
)
async def upload_photos(
    job_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    service = JobService(db)
    photos = await service.upload_photos(job_id=job_id, files=files)
    return PhotoUploadResponse(job_id=job_id, file_count=len(photos))


@router.post(
    "/jobs/{job_id}/cluster",
    summary="Start clustering",
    response_model=JobStatusResponse,
)
async def cluster_photos(
    job_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    service = JobService(db)
    await service.cluster_photos(job_id=job_id, background_tasks=background_tasks)
    return JobStatusResponse(
        job_id=job_id,
        status="PROCESSING",
        message="Clustering started",
    )