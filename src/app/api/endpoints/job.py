import asyncio
import uuid
from typing import List

from app.core.config import LocalConfig
from app.api.deps import get_process_executor, get_thread_executor
from app.api.endpoints.auth import get_current_user
from app.core.config import settings
from app.core.logger import get_logger
from app.db.database import AsyncSessionLocal, get_db
from app.models.job import Job
from app.models.photo import Photo
from app.models.user import User
from app.schemas.cluster import JobRequest, JobResponse
from app.services.pipeline import PhotoClusteringPipeline
from app.services.storage.local import LocalStorageService as StorageService
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
from sqlalchemy.future import select

router = APIRouter()
logger = get_logger(__name__)


@router.get("/jobs", response_model=list[JobResponse], status_code=status.HTTP_200_OK)
async def get_jobs(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Job).where(Job.user_id == current_user.user_id))
    jobs = result.scalars().all()
    if not jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return [
        JobResponse(
            id=job.id, title=job.title, status=job.status, created_at=job.created_at
        )
        for job in jobs
    ]


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: JobRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"title: {payload.title}")
    if not payload.title:
        raise
    job = Job(user_id=current_user.user_id, title=payload.title)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    return JobResponse(
        id=job.id, title=job.title, status=job.status, created_at=job.created_at
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get job details."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        id=job.id, title=job.title, status=job.status, created_at=job.created_at
    )


@router.post("/jobs/{job_id}/photos", summary="Upload photos")
async def upload_photos(
    job_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload photos to the job workspace."""
    # Check if job exists
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    config = LocalConfig(job_id=job.id)
    storage = StorageService(config)

    async def process_file(file: UploadFile):
        file_path = await storage.save_file(file)
        # Create Photo record
        photo = Photo(
            job_id=job_id,
            original_filename=file.filename,
            storage_path=str(file_path.relative_to(config.IMAGE_DIR)),
        )
        return photo

    # Process files concurrently
    photos = await asyncio.gather(*[process_file(file) for file in files])

    db.add_all(photos)
    await db.commit()

    return {"job_id": job_id, "file_count": len(files)}


# async def run_pipeline_task(job_id: str, db: AsyncSession):
#     """Background task to run the pipeline."""
#     # Note: Passing db session to background task requires care with async sessions and scoping.
#     # Usually better to create a new session within the task.
#     # For this MVP, we'll just instantiate the pipeline.
#     config = StorageService(job_id)
#     pipeline = PhotoClusteringPipeline(config)
#     await pipeline.run()

#     # Update job status (pseudo-code, would need a new DB session)
#     async with db() as session:
#         job = await session.get(Job, job_id)
#         job.status = "COMPLETED"
#         await session.commit()


async def run_pipeline_task(job_id: str):
    """Background task to run the pipeline."""
    logger.info(f"Starting pipeline for job {job_id}")
    async with AsyncSessionLocal() as session:
        try:
            config = LocalConfig(job_id=job_id)
            storage = StorageService(config)
            pipeline = PhotoClusteringPipeline(storage)
            await pipeline.run()

            job = await session.get(Job, job_id)
            if job:
                job.status = "COMPLETED"
                await session.commit()
            logger.info(f"Pipeline for job {job_id} completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline for job {job_id} failed: {e}")
            job = await session.get(Job, job_id)
            if job:
                job.status = "FAILED"
                await session.commit()


# @router.post("/jobs/{job_id}/cluster", summary="Start clustering")
# async def cluster_photos(
#     job_id: str,
#     background_tasks: BackgroundTasks,
#     db: AsyncSession = Depends(get_db)
# ):
#     """Trigger the clustering pipeline."""
#     result = await db.execute(select(Job).where(Job.id == job_id))
#     job = result.scalars().first()
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")

#     # Update status
#     job.status = "PROCESSING"
#     await db.commit()

#     # Trigger pipeline in background
#     background_tasks.add_task(run_pipeline_task, job_id, db)

#     return {"job_id": job_id, "status": "PROCESSING", "message": "Clustering started"}


@router.post("/jobs/{job_id}/cluster", summary="Start clustering")
async def cluster_photos(
    job_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    """Trigger the clustering pipeline."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Update status
    job.status = "PROCESSING"
    await db.commit()

    # Trigger pipeline in background
    background_tasks.add_task(run_pipeline_task, job_id)

    return {
        "job_id": job_id,
        "status": "PROCESSING",
        "message": "Clustering started",
    }