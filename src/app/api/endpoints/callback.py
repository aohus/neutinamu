
import logging
from typing import List

from app.db.database import get_db
from app.models.cluster import Cluster
from app.models.job import ClusterJob, Job, JobStatus
from app.models.photo import Photo
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cluster/callback")
async def handle_clustering_result(payload: dict = Body(...), db: AsyncSession = Depends(get_db)):
    task_id = payload.get("task_id")
    status = payload.get("status")
    request_id = payload.get("request_id")  # This is our ClusterJob.id
    
    logger.info(f"Received callback for task {task_id} (req={request_id}), status: {status}")
    
    # 1. Find ClusterJob
    result = await db.execute(select(ClusterJob).where(ClusterJob.id == request_id))
    cluster_job = result.scalar_one_or_none()
    
    if not cluster_job:
        logger.error(f"ClusterJob not found for request_id: {request_id}")
        # We can't do much if we don't know the job.
        return {"status": "error", "message": "ClusterJob not found"}

    job_id = cluster_job.job_id
    
    # 2. Handle Status
    if status == "completed":
        cluster_result = payload.get("result")
        # logger.info(f"Clustering success! Result keys: {cluster_result.keys()}")
        try:
            await create_clusters(db, cluster_job, cluster_result)
            logger.info(f"Job {job_id} updated with clustering results.")
        except Exception as e:
            logger.exception(f"Error applying clustering results for job {job_id}: {e}")
            # Mark as failed if application fails
            cluster_job.error_message = f"Error applying results: {str(e)}"
            await _mark_job_failed(db, job_id)
            await db.commit()

    elif status == "failed":
        error = payload.get("error")
        logger.error(f"Task {task_id} failed: {error}")
        cluster_job.error_message = str(error)
        await _mark_job_failed(db, job_id)
        await db.commit()
    
    return {"status": "ok"}


async def create_clusters(db: AsyncSession, cluster_job: ClusterJob, result: dict):
    job_id = cluster_job.job_id
    cluster_json = result.get("clusters", [])

    # 1. Fetch Photos to map paths
    photo_result = await db.execute(select(Photo).where(Photo.job_id == job_id))
    photos = photo_result.scalars().all()
    
    # 2. Clear existing clusters (Idempotency)
    await db.execute(delete(Cluster).where(Cluster.job_id == job_id))
    await db.flush()

    # Create "reserve" cluster for noise (-1) or unassigned
    reserve_cluster = Cluster(
        job_id=job_id,
        name="reserve",
        order_index=-1,
    )
    db.add(reserve_cluster)
    await db.flush()

    # 3. Create Clusters and assign Photos
    for c_data in cluster_json:
        cluster_id_num = int(c_data.get("id", 0))
        
        # If id is -1, it's noise -> assign to reserve
        if cluster_id_num == -1:
            target_cluster_id = reserve_cluster.id
        else:
            # Create Cluster
            new_cluster = Cluster(
                job_id=job_id,
                name=f"Group {cluster_id_num + 1}",
                order_index=cluster_id_num,
            )
            db.add(new_cluster)
            await db.flush()
            target_cluster_id = new_cluster.id
        
        # Assign Photos and Update Metadata
        # photo_details is a list of objects with path, timestamp, lat, lon
        photo_details = c_data.get("photo_details", [])
        
        # If photo_details is empty, fallback to "photos" (list of paths)
        if not photo_details:
            photo_paths = c_data.get("photos", [])
            photo_details = [{"path": p} for p in photo_paths]

        for idx, p_detail in enumerate(photo_details):
            p_path = p_detail.get("path")
            
            # Match photo by storage_path
            matched_photo = None
            for photo in photos:
                if p_path and p_path.endswith(photo.storage_path):
                    matched_photo = photo
                    break
            
            if matched_photo:
                matched_photo.cluster_id = target_cluster_id
                matched_photo.order_index = idx # Use index in the list as order
                
                # Update metadata if provided
                if p_detail.get("timestamp"):
                    # timestamp is float (unix epoch) -> convert to datetime
                    from datetime import datetime
                    matched_photo.meta_timestamp = datetime.fromtimestamp(p_detail.get("timestamp"))
                if p_detail.get("lat"):
                    matched_photo.meta_lat = p_detail.get("lat")
                if p_detail.get("lon"):
                    matched_photo.meta_lon = p_detail.get("lon")
            else:
                logger.warning(f"Photo path not matched: {p_path}")

    # 4. Update Job and ClusterJob
    cluster_job.finished_at = func.now()
    cluster_job.result = result # Save raw result
    
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one()
    job.status = JobStatus.COMPLETED
    
    await db.commit()


async def _mark_job_failed(db: AsyncSession, job_id: str):
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if job:
        job.status = JobStatus.FAILED
