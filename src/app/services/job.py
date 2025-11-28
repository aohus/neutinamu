import asyncio
import logging
from datetime import datetime
from typing import Any, List, Sequence

from app.core.config import LocalConfig
from app.db.database import AsyncSessionLocal
from app.domain.pipeline import PhotoClusteringPipeline
from app.domain.storage.local import LocalStorageService as StorageService
from app.models.cluster import Cluster
from app.models.job import Job
from app.models.photo import Photo
from app.models.user import User
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_jobs(self, user: User):
        logger.info(f"Fetching jobs for user: {user.user_id}")
        result = await self.db.execute(select(Job).where(Job.user_id == user.user_id))
        jobs = result.scalars().all()
        
        if not jobs:
            logger.warning(f"No jobs found for user: {user.user_id}")
            raise HTTPException(status_code=404, detail="Job not found")
        logger.info(f"Found {len(jobs)} jobs for user: {user.user_id}")
        return jobs

    async def create_job(self, user: User, title: str):
        logger.info(f"User {user.user_id} creating job with title: '{title}'")
        job = Job(user_id=user.user_id, title=title)
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        logger.info(f"Job created successfully with ID: {job.id}")
        return job

    async def get_job(self, job_id: str):
        logger.debug(f"Fetching job with ID: {job_id}")
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            logger.warning(f"Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")
        
        logger.debug(f"Job {job_id} found with status {job.status}")
        return job

    async def upload_photos(
        self,
        job_id: str,
        files: List[UploadFile] = File(...),
    ):
        logger.info(f"Uploading {len(files)} files for job ID: {job_id}")
        # Check if job exists
        result = await self.db.execute(select(Job).where(Job.id == job_id))
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

        self.db.add_all(photos)
        await self.db.commit()

        logger.info(
            f"Successfully uploaded and saved {len(photos)} photos for job {job_id}."
        )
        return photos

    async def cluster_photos(self, job_id: str, background_tasks: BackgroundTasks):
        """Trigger the clustering pipeline."""
        logger.info(f"Received request to start clustering for job {job_id}")
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            logger.error(f"Clustering trigger failed: Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        # Update status
        job.status = "PROCESSING"
        await self.db.commit()
        logger.info(f"Job {job_id} status updated to PROCESSING.")

        # Trigger pipeline in background
        background_tasks.add_task(run_pipeline_task, job_id)
        logger.info(f"Clustering task for job {job_id} added to background tasks.")


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


