import logging
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, with_loader_criteria

from app.api.deps import get_uow
from app.api.endpoints.auth import get_current_user
from app.common.uow import UnitOfWork
from app.db.database import get_db
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob, Job, JobStatus
from app.models.photo import Photo
from app.models.user import User
from app.schemas.enum import ExportStatus, JobStatus
from app.schemas.job import (
    ExportStatusResponse,
    FileResponse,
    JobClusterRequest,
    JobDetailsResponse,
    JobExportRequest,
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

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs", response_model=list[JobResponse], status_code=status.HTTP_200_OK)
async def get_jobs(current_user: User = Depends(get_current_user), uow: UnitOfWork = Depends(get_uow)):
    service = JobService(uow)
    jobs = await service.get_jobs(current_user)
    return [
        JobResponse(
            id=job.id,
            title=job.title,
            status=job.status,
            export_status=job.export_job.status if job.export_job else ExportStatus.PENDING,
            created_at=job.created_at,
        )
        for job in jobs
    ]


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: JobRequest,
    current_user: User = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    service = JobService(uow)
    logger.info(f"User {current_user.user_id} creating job with title: '{payload.title}'")
    job = await service.create_job(
        user=current_user,
        title=payload.title,
        construction_type=payload.construction_type,
        company_name=payload.company_name,
    )

    return JobResponse(
        id=job.id,
        title=job.title,
        status=job.status,
        export_status=ExportStatus.PENDING,
        created_at=job.created_at,
        construction_type=job.construction_type,
        company_name=job.company_name,
    )


@router.delete("/jobs/{job_id}", response_model=JobStatusResponse, status_code=status.HTTP_200_OK)
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    service = JobService(uow)
    logger.info(f"User {current_user.user_id} deleting job '{job_id}'")
    await service.delete_job(job_id=job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus.DELETED,
        message="Job Deleted",
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, uow: UnitOfWork = Depends(get_uow)):
    """Get job details."""
    service = JobService(uow)
    job = await service.get_job(job_id=job_id)
    return JobResponse(
        id=job.id,
        title=job.title,
        status=job.status,
        export_status=job.export_job.status if job.export_job else ExportStatus.PENDING,
        created_at=job.created_at,
    )


@router.get("/jobs/{job_id}/details", response_model=JobDetailsResponse)
async def get_job_details(job_id: str, uow: UnitOfWork = Depends(get_uow)):
    """Get full job details including photos and clusters."""
    service = JobService(uow)
    job = await service.get_job_details(job_id=job_id)
    return job


@router.post(
    "/jobs/{job_id}/photos",
    summary="Upload photos (Legacy/Proxy)",
    response_model=PhotoUploadResponse,
)
async def upload_photos(
    job_id: str,
    files: List[UploadFile] = File(...),
    uow: UnitOfWork = Depends(get_uow),
):
    service = JobService(uow)
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
    origin: str | None = Header(None),
    uow: UnitOfWork = Depends(get_uow),
):
    service = JobService(uow)
    return await service.generate_upload_urls(job_id=job_id, files=files, origin=origin)


@router.post(
    "/jobs/{job_id}/photos/complete",
    summary="Notify upload completion for direct uploads (Sync)",
    status_code=status.HTTP_200_OK,
    response_model=JobDetailsResponse,
)
async def complete_upload(
    job_id: str,
    uploaded_files: List[dict],  # {filename: str, storage_path: str}
    uow: UnitOfWork = Depends(get_uow),
):
    service = JobService(uow)
    
    # Mark as UPLOADING
    trigger_ts = await service.set_job_uploading(job_id)

    # Process files synchronously
    await service.process_uploaded_files(
        job_id=job_id, 
        file_info_list=uploaded_files, 
        trigger_timestamp=trigger_ts
    )

    # Return updated job details
    job = await service.get_job_details(job_id=job_id)
    return job


@router.post(
    "/jobs/{job_id}/cluster",
    summary="Start clustering",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_cluster(
    job_id: str, payload: JobClusterRequest, uow: UnitOfWork = Depends(get_uow)
):
    service = JobService(uow)
    job, data = await service.start_cluster_server(
        job_id=job_id,
        min_samples=payload.min_samples,
        max_dist_m=payload.max_dist_m,
        max_alt_diff_m=payload.max_alt_diff_m,
    )
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        message=str(data),
    )


@router.post("/jobs/{job_id}/export", response_model=ExportStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_export(
    job_id: str, payload: JobExportRequest, uow: UnitOfWork = Depends(get_uow)
):
    service = JobService(uow)
    export_job = await service.start_export(
        job_id=job_id,
        cover_title=payload.cover_title,
        cover_company_name=payload.cover_company_name,
        labels=payload.labels,
    )
    return ExportStatusResponse(status=export_job.status)


@router.get("/jobs/{job_id}/export/status", response_model=ExportStatusResponse)
async def get_export_status(job_id: str, uow: UnitOfWork = Depends(get_uow)):
    service = JobService(uow)
    status, pdf_url, err = await service.get_export_job(job_id=job_id)
    return ExportStatusResponse(
        status=status,
        pdf_url=pdf_url,
        error_message=err,
    )


@router.get("/jobs/{job_id}/export/download")
async def download_export_pdf(job_id: str, uow: UnitOfWork = Depends(get_uow)):
    service = JobService(uow)
    pdf_path, name = await service.download_export_pdf(job_id=job_id)

    return FileResponse(
        path=str(pdf_path),
        filename=name,
        media_type="application/pdf",
    )
