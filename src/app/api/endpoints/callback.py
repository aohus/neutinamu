
import logging
from typing import List

from app.db.database import get_db
from app.models.cluster import Cluster
from app.models.job import ClusterJob, Job, JobStatus
from app.models.photo import Photo
from app.models.photo_detail import PhotoDetail
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import delete, select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cluster/callback")
async def handle_clustering_result(payload: dict = Body(...), db: AsyncSession = Depends(get_db)):
    task_id = payload.get("task_id")
    status = payload.get("status")
    request_id = payload.get("request_id")  # This is our ClusterJob.id
    
    logger.info(f"Received callback for task {task_id} (req={request_id}), status: {status}")
    
    result = await db.execute(select(ClusterJob).where(ClusterJob.id == request_id))
    cluster_job = result.scalar_one_or_none()
    
    if not cluster_job:
        logger.error(f"ClusterJob not found for request_id: {request_id}")
        return {"status": "error", "message": "ClusterJob not found"}

    job_id = cluster_job.job_id
    
    if status == "completed":
        cluster_result = payload.get("result")
        # logger.info(f"Clustering success! Result keys: {cluster_result.keys()}")
        try:
            await create_clusters(db, cluster_job, cluster_result)
            logger.info(f"Job {job_id} updated with clustering results.")
        except Exception as e:
            logger.exception(f"Error applying clustering results for job {job_id}: {e}")
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
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one()
    
    cluster_json = result.get("clusters", [])

    photo_result = await db.execute(select(Photo).where(Photo.job_id == job_id))
    photos = photo_result.scalars().all()
    
    await _remove_previous_clusters(db, job_id)

    reserve_cluster = Cluster(
        job_id=job_id,
        name="reserve",
        order_index=-1,
    )
    db.add(reserve_cluster)
    await db.flush()

    base_name = job.construction_type if job.construction_type else job.title
    for c_data in cluster_json:
        cluster_id_num = int(c_data.get("id", 0))
        
        if cluster_id_num == -1:
            target_cluster_id = reserve_cluster.id
        else:
            new_cluster = Cluster(
                job_id=job_id,
                name=base_name,
                order_index=cluster_id_num,
            )
            db.add(new_cluster)
            await db.flush()
            target_cluster_id = new_cluster.id
        
        photo_list = c_data.get("photos", [])
        if not photo_list:
            logger.warning(f"No photos found in cluster {cluster_id_num}")
            continue

        for idx, p in enumerate(photo_list):
            p_path = p.get("path")
            
            matched_photo = None
            for photo in photos:
                if p_path and (p_path.endswith(photo.thumbnail_path) if photo.thumbnail_path else p_path.endswith(photo.storage_path)):
                    matched_photo = photo
                    break
            
            if matched_photo:
                matched_photo.cluster_id = target_cluster_id
                matched_photo.order_index = idx
                
                if p.get("timestamp"):
                    from datetime import datetime
                    matched_photo.meta_timestamp = datetime.fromtimestamp(p.get("timestamp"))
                if p.get("lat"):
                    matched_photo.meta_lat = p.get("lat")
                if p.get("lon"):
                    matched_photo.meta_lon = p.get("lon")
                await db.execute(delete(PhotoDetail).where(PhotoDetail.photo_id == matched_photo.id))
                
                new_detail = PhotoDetail(
                    photo_id=matched_photo.id,
                    device=p.get("device"),
                    focal_length=p.get("focal_length"),
                    exposure_time=p.get("exposure_time"),
                    iso=p.get("iso_speed_rating"),
                    flash=p.get("flash"),
                    orientation=p.get("orientation"),
                    gps_img_direction=p.get("gps_img_direction")
                )
                db.add(new_detail)
            else:
                logger.warning(f"Photo path not matched: {p_path}")

    cluster_job.finished_at = func.now()
    cluster_job.result = result
    
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one()
    job.status = JobStatus.COMPLETED
    
    await db.commit()


async def _mark_job_failed(db: AsyncSession, job_id: str):
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job = job_result.scalar_one_or_none()
    if job:
        job.status = JobStatus.FAILED

async def _remove_previous_clusters(db: AsyncSession, job_id: str):
    logger.info(f"Clearing previous clusters for job {job_id}")
    cluster_ids_subq = select(Cluster.id).where(Cluster.job_id == job_id).subquery()

    await db.execute(
        update(Photo).where(Photo.cluster_id.in_(select(cluster_ids_subq.c.id))).values(cluster_id=None)
    )

    await db.execute(delete(Cluster).where(Cluster.id.in_(select(cluster_ids_subq.c.id))))
    await db.flush()


@router.post("/pdf/callback")
async def handle_pdf_result(payload: dict = Body(...), db: AsyncSession = Depends(get_db)):
    task_id = payload.get("task_id")
    status = payload.get("status")
    request_id = payload.get("request_id")  # This is our ClusterJob.id
    
    logger.info(f"Received callback for task {task_id} (req={request_id}), status: {status}")
    
    result = await db.execute(select(ClusterJob).where(ClusterJob.id == request_id))
    cluster_job = result.scalar_one_or_none()
    
    if not cluster_job:
        logger.error(f"ClusterJob not found for request_id: {request_id}")
        return {"status": "error", "message": "ClusterJob not found"}

    job_id = cluster_job.job_id
    
    if status == "completed":
        cluster_result = payload.get("result")
        # logger.info(f"Clustering success! Result keys: {cluster_result.keys()}")
        try:
            await create_clusters(db, cluster_job, cluster_result)
            logger.info(f"Job {job_id} updated with clustering results.")
        except Exception as e:
            logger.exception(f"Error applying clustering results for job {job_id}: {e}")
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