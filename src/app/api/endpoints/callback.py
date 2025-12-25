
import logging
from typing import List

from app.api.deps import get_uow
from app.common.uow import UnitOfWork
from app.db.database import get_db
from app.models.cluster import Cluster
from app.models.job import ClusterJob
from app.schemas.enum import JobStatus, ExportStatus
from app.models.photo import Photo
from app.models.photo_detail import PhotoDetail
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import delete, select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cluster/callback")
async def handle_clustering_result(payload: dict = Body(...), uow: UnitOfWork = Depends(get_uow)):
    task_id = payload.get("task_id")
    status = payload.get("status")
    request_id = payload.get("request_id")  # This is our ClusterJob.id
    
    logger.info(f"Received callback for task {task_id} (req={request_id}), status: {status}")
    
    # Use uow to get ClusterJob
    cluster_job = await uow.jobs.get_cluster_job_by_id(request_id)
    
    if not cluster_job:
        logger.error(f"ClusterJob not found for request_id: {request_id}")
        return {"status": "error", "message": "ClusterJob not found"}

    job_id = cluster_job.job_id
    
    async with uow: # Start a Unit of Work transaction
        if status == "completed":
            cluster_result = payload.get("result")
            try:
                # create_clusters now uses uow
                await create_clusters(uow, cluster_job, cluster_result)
                logger.info(f"Job {job_id} updated with clustering results.")
            except Exception as e:
                logger.exception(f"Error applying clustering results for job {job_id}: {e}")
                cluster_job.error_message = f"Error applying results: {str(e)}"
                # _mark_job_failed now uses uow
                await _mark_job_failed(uow, job_id)

        elif status == "failed":
            error = payload.get("error")
            logger.error(f"Task {task_id} failed: {error}")
            cluster_job.error_message = str(error)
            # _mark_job_failed now uses uow
            await _mark_job_failed(uow, job_id)
        
        # All changes within this async with uow block will be committed
    
    return {"status": "ok"}


async def create_clusters(uow: UnitOfWork, cluster_job: ClusterJob, result: dict):
    job_id = cluster_job.job_id
    job = await uow.jobs.get_by_id(job_id) # Use uow.jobs to get Job
    
    if not job:
        logger.error(f"Job not found for job_id: {job_id}")
        return {"status": "error", "message": "Job not found"}

    cluster_json = result.get("clusters", [])

    photos = await uow.photos.get_by_job_id(job_id) # Use uow.photos to get Photos
    
    await _remove_previous_clusters(uow, job_id) # Pass uow

    reserve_cluster = Cluster(
        job_id=job_id,
        name="reserve",
        order_index=-1,
    )
    await uow.clusters.create(reserve_cluster) # Use uow.clusters to create Cluster
    await uow.flush()

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
            await uow.clusters.create(new_cluster) # Use uow.clusters to create Cluster
            await uow.flush()
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
                
                # Delete existing PhotoDetail
                await uow.db.execute(delete(PhotoDetail).where(PhotoDetail.photo_id == matched_photo.id)) 
                
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
                uow.db.add(new_detail) 
            else:
                logger.warning(f"Photo path not matched: {p_path}")

    cluster_job.finished_at = func.now()
    cluster_job.result = result
    
    # Job status update, assuming job object is already being tracked by SQLAlchemy
    # and changes will be committed by the outer UOW block.
    if job: # Check if job is not None
        job.status = JobStatus.COMPLETED


async def _mark_job_failed(uow: UnitOfWork, job_id: str):
    job = await uow.jobs.get_by_id(job_id) # Use uow.jobs to get Job
    if job:
        job.status = JobStatus.FAILED

async def _remove_previous_clusters(uow: UnitOfWork, job_id: str):
    logger.info(f"Clearing previous clusters for job {job_id}")
    cluster_ids_subq = select(Cluster.id).where(Cluster.job_id == job_id).subquery()

    await uow.db.execute( # Use uow.db.execute
        update(Photo).where(Photo.cluster_id.in_(select(cluster_ids_subq.c.id))).values(cluster_id=None)
    )

    await uow.db.execute(delete(Cluster).where(Cluster.id.in_(select(cluster_ids_subq.c.id)))) # Use uow.db.execute
    await uow.db.flush() # Use uow.db.flush


@router.post("/pdf/callback")
async def handle_pdf_result(payload: dict = Body(...), uow: UnitOfWork = Depends(get_uow)):
    task_id = payload.get("task_id")
    request_id = payload.get("request_id")  # This is our ExportJob.id
    status = payload.get("status")
    
    logger.info(f"Received callback for task {task_id} (req={request_id}), status: {status}")
    
    # Use uow to get ExportJob
    export_job = await uow.jobs.get_export_job_by_id(request_id)
    
    if not export_job:
        logger.error(f"ExportJob not found for request_id: {request_id}")
        return {"status": "error", "message": "ExportJob not found"}
    
    async with uow: # Start a Unit of Work transaction
        if status == "completed":
            result_data = payload.get("result", {})
            pdf_url = result_data.get("pdf_url")
            
            export_job.status = ExportStatus.EXPORTED
            export_job.pdf_path = pdf_url
            export_job.finished_at = func.now()
            logger.info(f"PDF Export finished for job {export_job.job_id}")

        elif status == "failed":
            error = payload.get("error")
            export_job.status = ExportStatus.FAILED
            export_job.error_message = str(error)
            export_job.finished_at = func.now()
            logger.error(f"PDF Export failed for job {export_job.job_id}: {error}")
        await uow.jobs.save(export_job)

    return {"status": "ok"}
    