import logging
from typing import List

from app.api.endpoints.auth import get_current_user
from app.db.database import get_db
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob, Job, JobStatus
from app.models.photo import Photo
from app.models.user import User
from app.schemas.enum import ExportStatus, JobStatus
from app.schemas.job import (
    ExportStatusResponse,
    JobClusterRequest,
    JobDetailsResponse,
    JobRequest,
    JobResponse,
    JobStatusResponse,
    PhotoUploadResponse,
)
from app.schemas.photo import (
    BatchPresignedUrlResponse,
    PhotoCompleteRequest,
    PhotoUploadRequest,
)
from app.services.job import JobService
from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, with_loader_criteria

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
            export_status=job.export_job.status if job.export_job else ExportStatus.PENDING, 
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
        id=job.id, 
        title=job.title, 
        status=job.status, 
        export_status=ExportStatus.PENDING, 
        created_at=job.created_at
    )


@router.delete("/jobs/{job_id}", response_model=JobStatusResponse, status_code=status.HTTP_200_OK)
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    service = JobService(db)
    logger.info(f"User {current_user.user_id} deleting job '{job_id}'")
    await service.delete_job(job_id=job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus.DELETED,
        message="Job Deleted",
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get job details."""
    service = JobService(db)
    job = await service.get_job(job_id=job_id)
    return JobResponse(
            id=job.id, 
            title=job.title, 
            status=job.status, 
            export_status=job.export_job.status if job.export_job else ExportStatus.PENDING, 
            created_at=job.created_at)


@router.get("/jobs/{job_id}/details", response_model=JobDetailsResponse)
async def get_job_details(
    job_id: str, 
    db: AsyncSession = Depends(get_db)
):
    """Get full job details including photos and clusters."""
    # Efficiently fetch everything in one query or minimal queries
    query = (
        select(Job)
        .where(Job.id == job_id)
        .options(
            selectinload(Job.photos),
            selectinload(Job.clusters).selectinload(Cluster.photos),
            selectinload(Job.export_job),
            with_loader_criteria(Photo, Photo.deleted_at.is_(None)) 
        )
    )
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job not found")

    # Populate URLs for photos
    storage = get_storage_client()
    for photo in job.photos:
        photo.url = storage.get_url(photo.storage_path)
        photo.thumbnail_path = storage.get_url(photo.thumbnail_path) if photo.thumbnail_path else None
    
    for cluster in job.clusters:
        for photo in cluster.photos:
            photo.url = storage.get_url(photo.storage_path)
            photo.thumbnail_path = storage.get_url(photo.thumbnail_path) if photo.thumbnail_path else None

    return job


@router.post(
    "/jobs/{job_id}/photos",
    summary="Upload photos (Legacy/Proxy)",
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
    "/jobs/{job_id}/photos/presigned",
    summary="Get presigned URLs for direct upload",
    response_model=BatchPresignedUrlResponse,
)
async def generate_upload_urls(
    job_id: str,
    files: List[PhotoUploadRequest],
    db: AsyncSession = Depends(get_db),
):
    service = JobService(db)
    return await service.generate_presigned_urls(job_id=job_id, files=files)


@router.post(
    "/jobs/{job_id}/photos/complete",
    summary="Notify upload completion for direct uploads",
    response_model=PhotoUploadResponse,
)
async def complete_upload(
    job_id: str,
    # files: List[PhotoUploadRequest], # Reusing this to pass filename/content_type, but we might need storage_path.
    uploaded_files: List[dict],  # {filename: str, storage_path: str}
    db: AsyncSession = Depends(get_db),
):
    # Note: uploaded_files should ideally be a Pydantic model list. 
    # Using dict for flexibility now, but should be typed.
    service = JobService(db)
    photos = await service.process_uploaded_files(job_id=job_id, file_info_list=uploaded_files)
    return PhotoUploadResponse(job_id=job_id, file_count=len(photos))


@router.post(
    "/jobs/{job_id}/cluster",
    summary="Start clustering",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_cluster(
    job_id: str, 
    payload: JobClusterRequest, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    service = JobService(db)
    job, data = await service.start_cluster(job_id=job_id, 
                                background_tasks=background_tasks,
                                min_samples=payload.min_samples, 
                                max_dist_m=payload.max_dist_m, 
                                max_alt_diff_m=payload.max_alt_diff_m)
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        message=str(data),
    )


@router.post("/jobs/{job_id}/export", response_model=ExportStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_export(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    service = JobService(db)
    export_job = await service.start_export(job_id=job_id, background_tasks=background_tasks)
    return ExportStatusResponse(status=export_job.status)


@router.get("/jobs/{job_id}/export/status", response_model=ExportStatusResponse)
async def get_export_status(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    service = JobService(db)
    status, pdf_url, err = await service.get_export_job(job_id=job_id)
    return ExportStatusResponse(
        status=status,
        pdf_url=pdf_url,
        error_message=err,
    )

# @router.get("/jobs/{job_id}/export/download")
# def download_export_pdf(
#     job_id: str,
#     db: AsyncSession = Depends(get_db)
# ):
#     job = (
#         db.query(ExportJob)
#         .filter(ExportJob.session_id == session_id)
#         .order_by(ExportJob.created_at.desc())
#         .first()
#     )
#     if not job or job.status != ExportStatus.DONE or not job.pdf_path:
#         raise HTTPException(status_code=404, detail="No finished export for this session")

#     pdf_path = Path(job.pdf_path)
#     if not pdf_path.exists():
#         raise HTTPException(status_code=404, detail="PDF file not found")

#     return FileResponse(
#         path=str(pdf_path),
#         filename=pdf_path.name,
#         media_type="application/pdf",
#     )