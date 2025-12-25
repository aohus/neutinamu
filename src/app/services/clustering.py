import logging
import os

from app.common.uow import UnitOfWork
from app.common.core_clients import call_cluster_service
from app.models.photo import Photo
from app.models.user import User
from app.models.job import ClusterJob
from app.schemas.enum import JobStatus
from fastapi import HTTPException
from sqlalchemy import func


logger = logging.getLogger(__name__)


class ClusteringService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def start_cluster(
        self,
        job_id: str,
        min_samples: int = 3,
        max_dist_m: float = 10.0,
        max_alt_diff_m: float = 20.0,
        similarity_threshold: float = 0.8,
    ):
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        async with self.uow:
            job.status = JobStatus.PENDING
            await self.uow.jobs.save(job)

            cluster_job = ClusterJob(job_id=job_id)
            cluster_job_obj = await self.uow.jobs.create_cluster_job(cluster_job)

        try:
            response = await self.request_cluster_job(
                job_id, 
                cluster_job_obj.id,
                min_samples, 
                max_dist_m, 
                max_alt_diff_m, 
                similarity_threshold
            )
            return job, response
        except Exception as e:
            logger.exception(f"{job} failed: {e}")
            async with self.uow:
                job.status = JobStatus.FAILED
                cluster_job.error_message = str(e)
                cluster_job.finished_at = func.now()

    async def request_cluster_job(
        self,
        job_id: str,
        cluster_job_id: str,
        min_samples: int = 3,
        max_dist_m: float = 10.0,
        max_alt_diff_m: float = 20.0,
        similarity_threshold: float = 0.8,
    ):
        logger.info(f"Request cluster job to image_cluster_server {job_id}")

        photos = await self.uow.photos.get_by_job_id(job_id)
        if not photos:
            logger.warning(f"{job_id} has no photos")
            raise HTTPException(status_code=404, detail="No photos found")

        bucket_path = None
        for p in photos:
            if p.thumbnail_path:
                bucket_path = os.path.dirname(p.thumbnail_path)
                break
        
        if not bucket_path:
            for p in photos:
                if p.storage_path:
                    bucket_path = os.path.dirname(p.storage_path)
                    break
        
        if not bucket_path:
            logger.warning(f"{job} could not determine bucket path")
            raise HTTPException(status_code=404, detail="Could not determine bucket path")

        logger.info(f"Target bucket path: {bucket_path}, count: {len(photos)}")
        return await call_cluster_service(
            bucket_path=bucket_path,
            photo_cnt=len(photos),
            request_id=cluster_job_id,
            min_samples=min_samples,
            max_dist_m=max_dist_m,
            max_alt_diff_m=max_alt_diff_m,
            similarity_threshold=similarity_threshold,
        )
