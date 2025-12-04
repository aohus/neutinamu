from __future__ import annotations

import logging
import re
from typing import Any, List

import httpx
from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.job import ClusterJob, Job, JobStatus
from app.models.photo import Photo
from app.domain.storage.factory import get_storage_client
from app.domain.storage.local import LocalStorageService
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)


async def run_deep_cluster_for_job(
    job_id: str,
    min_samples: int = 3,
    max_dist_m: float = 10.0,
    max_alt_diff_m: float = 20.0,
    similarity_threshold: float = 0.8
):
    """
    1) job_id 에 속한 Photo들의 storage_path 조회
    2) image_cluster_server 호출
    3) Cluster / Photo.cluster_id 갱신
    4) Job.status 업데이트
    """
    logger.info(f"[DeepClusterRunner] start job_id={job_id}")

    # 새 세션을 직접 열어 사용 (BackgroundTask/asyncio.create_task 에서 종종 이렇게 처리)
    async with AsyncSessionLocal() as session:
        # 1. Job & Photo 조회
        job = await _get_job(session, job_id)
        if job is None:
            logger.error(f"[DeepClusterRunner] job_id={job_id} not found")
            return

        # 상태 업데이트: PROCESSING
        job.status = JobStatus.PROCESSING

        # ClusterJob 생성
        cluster_job = ClusterJob(job_id=job_id)
        session.add(cluster_job)
        await session.commit()
        await session.refresh(cluster_job)

        photos = await _get_photos_for_job(session, job_id)
        if not photos:
            logger.warning(f"[DeepClusterRunner] {job} has no photos")
            job.status = JobStatus.FAILED
            cluster_job.error_message = "No photos found"
            cluster_job.finished_at = func.now()
            await session.commit()
            return
        
        # Resolve paths based on storage type
        storage = get_storage_client()
        photo_paths = []
        for p in photos:
            if isinstance(storage, LocalStorageService):
                # Local: Absolute file path
                full_path = storage.media_root / p.storage_path
                photo_paths.append(str(full_path))
            else:
                # Remote (GCS/S3): Public or Signed URL
                photo_paths.append(storage.get_url(p.storage_path))
        
        logger.info(f"get {len(photo_paths)} photos")
        
        try:
            # 2. 클러스터 서버 호출
            return await call_cluster_service(
                photo_paths=photo_paths, 
                request_id=cluster_job.id,
                min_samples=min_samples,
                max_dist_m=max_dist_m,
                max_alt_diff_m=max_alt_diff_m,
                similarity_threshold=similarity_threshold
            )
        except Exception as e:
            logger.exception(f"[DeepClusterRunner] {job} failed: {e}")
            job.status = JobStatus.FAILED
            cluster_job.error_message = str(e)
            cluster_job.finished_At = func.now()
            await session.commit()


async def _get_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def _get_photos_for_job(session: AsyncSession, job_id: str) -> List[Photo]:
    result = await session.execute(select(Photo).where(Photo.job_id == job_id))
    return list(result.scalars().all())


async def call_cluster_service(
    photo_paths: list[str],
    request_id: str,
    min_samples: int = 3,
    max_dist_m: float = 10.0,
    max_alt_diff_m: float = 20.0,
    similarity_threshold: float = 0.8,
    use_cache: bool = True,
    remove_people: bool = True,
) -> None:
    """
    image_cluster_server 의 /cluster 를 호출.
    결과는 webhook으로 수신.
    """
    # Use the configured callback base URL
    webhook_url = f"{settings.CALLBACK_BASE_URL}/cluster/callback"

    payload = {
        "photo_paths": photo_paths,
        "webhook_url": webhook_url,
        "request_id": request_id,
        "min_samples": min_samples,
        "max_dist_m": max_dist_m,
        "max_alt_diff_m": max_alt_diff_m,
        "similarity_threshold": similarity_threshold,
        "use_cache": use_cache,
        "remove_people": remove_people,
    }
    async with httpx.AsyncClient(
        base_url=str(settings.CLUSTER_SERVICE_URL),
        timeout=10.0,
    ) as client:
        resp = await client.post("/api/cluster", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Task submitted successfully: {data.get('task_id')}, request_id: {request_id}")
        return data