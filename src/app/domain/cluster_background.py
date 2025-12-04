import logging
from datetime import datetime
from typing import Any, List, Sequence

from app.core.config import JobConfig
from app.db.database import AsyncSessionLocal
from app.domain.pipeline import PhotoClusteringPipeline
from app.domain.storage.factory import (
    get_storage_client,  # Import storage client factory
)
from app.models.cluster import Cluster
from app.models.job import ClusterJob, Job, JobStatus
from app.models.photo import Photo
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

# ============================================
# Clustering 
# ============================================
async def run_pipeline_task(
    job_id: str,
    min_samples: int = 3, 
    max_dist_m: float = 10.0, 
    max_alt_diff_m: float = 20.0
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
            await _mark_job_status(session, job_id, JobStatus.PROCESSING)
            cluster_job = ClusterJob(job_id=job_id)                                      
            session.add(cluster_job)                                                     
            await session.commit()                                                       
            await session.refresh(cluster_job)    

            photos = await _get_photos_from_job(session, job_id)
            cluster_groups = await _run_pipeline(job_id, photos, min_samples, max_dist_m, max_alt_diff_m)
            await _create_clusters_from_result(session, job_id, cluster_job, cluster_groups)
            await _mark_job_status(session, job_id, JobStatus.COMPLETED)
            await session.commit()
            logger.info("Pipeline for job %s completed successfully.", job_id)
        except Exception as exc:
            logger.exception("Pipeline for job %s failed: %s", job_id, exc)
            await session.rollback()

            # 실패 시 상태를 FAILED 로 기록
            async with AsyncSessionLocal() as session2:
                await _mark_job_status(session2, job_id, JobStatus.FAILED)
                cluster_job.error_message = "No photos found"                            
                cluster_job.finished_at = datetime.now()
                await session2.commit()
                logger.warning("Job %s status updated to FAILED.", job_id)


async def _run_pipeline(
    job_id: str,
    photos: list[Photo],
    min_samples: int, 
    max_dist_m: float, 
    max_alt_diff_m: float,
) -> Sequence[Sequence[Any]]:
    """
    실제 파이프라인 실행만 담당.
    cluster_groups: [[photo_obj, ...], ...] 형태를 기대.
    photo_obj 는 최소한 .timestamp, .path 를 가진 객체라고 가정.
    """
    config = JobConfig(job_id=job_id, 
                         min_samples=min_samples, 
                         max_dist_m=max_dist_m, 
                         max_alt_diff_m=max_alt_diff_m)
    storage = get_storage_client()
    pipeline = PhotoClusteringPipeline(config, storage, photos)
    clusters = await pipeline.run()
    return clusters

    
async def _get_photos_from_job(session: AsyncSession, job_id: str) -> List[Photo]:
    result = await session.execute(select(Photo).where(Photo.job_id == job_id))
    return list(result.scalars().all())


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
    job.updated_at = datetime.now()  # updated_at 필드가 있다면  


async def _create_clusters_from_result(
    session: AsyncSession,
    job_id: int,
    cluster_job: ClusterJob,
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

    await remove_previous_clusters(session, job_id)

    updated_photos: list[Photo] = []
    db_cluster = Cluster(
        job_id=job_id,
        name="reserve",
        order_index=-1,
    )
    session.add(db_cluster)

    job_title = job.title
    for cluster_index, photo_group in enumerate(cluster_groups):
        # 1) Cluster 레코드 생성
        db_cluster = Cluster(
            job_id=job_id,
            name=f"{job_title}",
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
        for order_index, photo_obj in enumerate(sorted_group):
            result = await session.execute(
                select(Photo).where(
                    (Photo.original_filename == photo_obj.original_name) & (Photo.job_id == job_id)
                )
            )
            photo = result.scalars().first()

            if not photo:
                logger.warning(
                    "Photo not found for path %s in job %s", photo_obj.path, job_id
                )
                continue
            photo.cluster_id = db_cluster.id
            photo.order_index = order_index
            photo.meta_lat = photo_obj.lat
            photo.meta_lon = photo_obj.lon
            photo.meta_timestamp = datetime.fromtimestamp(photo_obj.timestamp)
            updated_photos.append(photo)
        
        cluster_job.finished_at = datetime.now()
        cluster_job.result = str(cluster_groups) # Save raw result
    return updated_photos


async def remove_previous_clusters(session: AsyncSession, job_id: str):
    logger.info(f"Clearing previous clusters for job {job_id}")
    cluster_ids_subq = (
        select(Cluster.id)
        .where(Cluster.job_id == job_id)
        .subquery()
    )

    # 2) 그 cluster 들을 참조하는 Photo.cluster_id -> NULL
    await session.execute(
        update(Photo)
        .where(Photo.cluster_id.in_(select(cluster_ids_subq.c.id)))
        .values(cluster_id=None)
    )

    # 3) Cluster 삭제
    await session.execute(
        delete(Cluster).where(Cluster.id.in_(select(cluster_ids_subq.c.id)))
    )