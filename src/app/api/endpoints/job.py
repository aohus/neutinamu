import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, List, Sequence

from app.api.deps import get_process_executor, get_thread_executor
from app.api.endpoints.auth import get_current_user
from app.core.config import LocalConfig, settings
from app.db.database import AsyncSessionLocal, get_db
from app.models.cluster import Cluster
from app.models.job import Job
from app.models.photo import Photo
from app.models.user import User
from app.schemas.job import (
    JobRequest,
    JobResponse,
    JobStatusResponse,
    PhotoUploadResponse,
)
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
logger = logging.getLogger(__name__)


@router.get("/jobs", response_model=list[JobResponse], status_code=status.HTTP_200_OK)
async def get_jobs(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    logger.info(f"Fetching jobs for user: {current_user.user_id}")
    result = await db.execute(select(Job).where(Job.user_id == current_user.user_id))
    jobs = result.scalars().all()
    if not jobs:
        logger.warning(f"No jobs found for user: {current_user.user_id}")
        raise HTTPException(status_code=404, detail="Job not found")

    logger.info(f"Found {len(jobs)} jobs for user: {current_user.user_id}")
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
    logger.info(f"User {current_user.user_id} creating job with title: '{payload.title}'")
    if not payload.title:
        logger.error("Job creation failed: Title is required.")
        raise HTTPException(status_code=400, detail="Title is required")

    job = Job(user_id=current_user.user_id, title=payload.title)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    logger.info(f"Job created successfully with ID: {job.id}")

    return JobResponse(
        id=job.id, title=job.title, status=job.status, created_at=job.created_at
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get job details."""
    logger.debug(f"Fetching job with ID: {job_id}")
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()
    if not job:
        logger.warning(f"Job with ID {job_id} not found.")
        raise HTTPException(status_code=404, detail="Job not found")
    
    logger.debug(f"Job {job_id} found with status {job.status}")
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
    """Upload photos to the job workspace."""
    logger.info(f"Uploading {len(files)} files for job ID: {job_id}")
    # Check if job exists
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()
    if not job:
        logger.error(f"Upload failed: Job with ID {job_id} not found.")
        raise HTTPException(status_code=404, detail="Job not found")

    config = LocalConfig(job_id=job.id)
    storage = StorageService(config)

    async def process_file(file: UploadFile):
        logger.debug(f"Processing file: {file.filename}")
        file_path = await storage.save_file(file)
        photo = Photo(
            job_id=job_id,
            original_filename=file.filename,
            storage_path=str(file_path.relative_to(config.IMAGE_DIR)),
        )
        logger.debug(f"Saved {file.filename} to {file_path}")
        return photo

    # Process files concurrently
    photos = await asyncio.gather(*[process_file(file) for file in files])

    db.add_all(photos)
    await db.commit()

    logger.info(
        f"Successfully uploaded and saved {len(photos)} photos for job {job_id}."
    )
    return PhotoUploadResponse(job_id=job_id, file_count=len(files))


# async def run_pipeline_task(job_id: str):
#     """Background task to run the pipeline."""
#     logger.info(f"Starting pipeline for job {job_id}")
#     async with AsyncSessionLocal() as session:
#         try:
#             config = LocalConfig(job_id=job_id)
#             storage = StorageService(config)
#             pipeline = PhotoClusteringPipeline(storage)
#             clusters = await pipeline.run()

#             update_clusters(session, job_id, clusters)
#             logger.info(f"Pipeline for job {job_id} completed successfully.")
#             await session.commit()
#         except Exception as e:
#             logger.error(f"Pipeline for job {job_id} failed: {e}", exc_info=True)
#             job = await session.get(Job, job_id)
#             if job:
#                 job.status = "FAILED"
#                 await session.commit()
#                 logger.warning(f"Job {job_id} status updated to FAILED.")


async def run_pipeline_task(
    job_id: int,
    # session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Background task to run the clustering pipeline for a job.

    - job을 PROCESSING 으로 변경
    - 파이프라인 실행
    - 클러스터/포토 업데이트
    - 성공 시 COMPLETED, 실패 시 FAILED
    """
    logger.info("Starting pipeline for job %s", job_id)

    async with AsyncSessionLocal() as session:
        try:
            await _mark_job_status(session, job_id, "PROCESSING")
            cluster_groups = await _run_pipeline(job_id)
            await _create_clusters_from_result(session, job_id, cluster_groups)
            await _mark_job_status(session, job_id, "COMPLETED")
            await session.commit()
            logger.info("Pipeline for job %s completed successfully.", job_id)
        except Exception as exc:
            logger.exception("Pipeline for job %s failed: %s", job_id, exc)
            await session.rollback()

            # 실패 시 상태를 FAILED 로 기록
            async with AsyncSessionLocal() as session2:
                await _mark_job_status(session2, job_id, "FAILED")
                await session2.commit()
                logger.warning("Job %s status updated to FAILED.", job_id)


async def _run_pipeline(job_id: int) -> Sequence[Sequence[Any]]:
    """
    실제 파이프라인 실행만 담당.
    cluster_groups: [[photo_obj, ...], ...] 형태를 기대.
    photo_obj 는 최소한 .timestamp, .path 를 가진 객체라고 가정.
    """
    config = LocalConfig(job_id=job_id)
    storage = StorageService(config)
    pipeline = PhotoClusteringPipeline(storage)

    clusters = await pipeline.run()
    return clusters


async def _mark_job_status(
    session: AsyncSession,
    job_id: int,
    status: str,
) -> None:
    job = await session.get(Job, job_id)
    if not job:
        logger.warning("Job %s not found while setting status to %s", job_id, status)
        return
    job.status = status
    job.updated_at = datetime.utcnow()  # updated_at 필드가 있다면


async def _create_clusters_from_result(
    session: AsyncSession,
    job_id: int,
    cluster_groups: Sequence[Sequence[Any]],
) -> list[Photo]:
    """
    파이프라인 결과(클러스터별 사진 그룹)를 기반으로
    Cluster / Photo 테이블을 업데이트.

    cluster_groups: 각 요소는 사진 객체 리스트
    사진 객체는 최소한 .timestamp, .path 속성을 가진다고 가정.
    """
    job = await session.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    updated_photos: list[Photo] = []

    for cluster_index, photo_group in enumerate(cluster_groups, start=1):
        # 1) Cluster 레코드 생성
        db_cluster = Cluster(
            job_id=job_id,
            name=f"장소 {cluster_index}",
            order_index=cluster_index,
        )
        session.add(db_cluster)
        # cluster_id 를 Photo에 써야 하므로 flush 로 PK 확보
        await session.flush()

        # 2) 그룹 내 사진 정렬
        sorted_group = sorted(
            photo_group,
            key=lambda p: (
                p.timestamp is None,
                p.timestamp if p.timestamp is not None else 0.0,
                p.path,
            ),
        )

        # 3) Photo 레코드 업데이트
        for order_index, photo_obj in enumerate(sorted_group, start=1):
            result = await session.execute(
                select(Photo).where(Photo.storage_path == photo_obj.original_name)
            )
            photo = result.scalars().first()

            if not photo:
                logger.warning(
                    "Photo not found for path %s in job %s", photo_obj.path, job_id
                )
                continue

            photo.cluster_id = db_cluster.id
            photo.order_index = order_index
            updated_photos.append(photo)
    return updated_photos


@router.post(
    "/jobs/{job_id}/cluster",
    summary="Start clustering",
    response_model=JobStatusResponse,
)
async def cluster_photos(
    job_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    """Trigger the clustering pipeline."""
    logger.info(f"Received request to start clustering for job {job_id}")
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalars().first()
    if not job:
        logger.error(f"Clustering trigger failed: Job with ID {job_id} not found.")
        raise HTTPException(status_code=404, detail="Job not found")

    # Update status
    job.status = "PROCESSING"
    await db.commit()
    logger.info(f"Job {job_id} status updated to PROCESSING.")

    # Trigger pipeline in background
    background_tasks.add_task(run_pipeline_task, job_id)
    logger.info(f"Clustering task for job {job_id} added to background tasks.")

    return JobStatusResponse(
        job_id=job_id,
        status="PROCESSING",
        message="Clustering started",
    )