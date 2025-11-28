import logging
from typing import List

from app.db.database import get_db
from app.models.cluster import Cluster
from app.models.job import Job
from app.schemas.cluster import ClusterCreate, ClusterDetail, ClusterRename
from app.services.storage import StorageService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/clusters", response_model=List[ClusterDetail])
async def list_clusters(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all clusters for a job."""
    logger.info(f"Listing clusters for job_id: {job_id}")
    result = await db.execute(
        select(Cluster)
        .where(Cluster.job_id == job_id)
        .options(selectinload(Cluster.photos))
    )
    clusters = result.scalars().all()
    logger.info(f"Found {len(clusters)} clusters for job_id: {job_id}")
    return clusters

@router.post("/jobs/{job_id}/clusters", response_model=ClusterDetail, status_code=status.HTTP_201_CREATED)
async def create_cluster(
    job_id: str,
    payload: ClusterCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new empty cluster."""
    logger.info(f"Creating cluster for job_id: {job_id} with name: {payload.name}")
    result = await db.execute(select(Job).where(Job.id == job_id))
    if not result.scalars().first():
        logger.error(f"Failed to create cluster: Job not found for id {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")

    storage = StorageService(job_id)
    
    if payload.name:
        try:
            storage.create_cluster(payload.name)
            logger.debug(f"Created directory for cluster: {payload.name}")
        except Exception as e:
            logger.error(f"Failed to create cluster directory: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    cluster = Cluster(
        job_id=job_id,
        name=payload.name or "New Cluster",
        order_index=payload.order_index or 0
    )
    db.add(cluster)
    await db.commit()
    await db.refresh(cluster)
    logger.info(f"Cluster '{cluster.name}' created successfully with id: {cluster.id}")
    return cluster

@router.patch("/clusters/{cluster_id}", response_model=ClusterDetail)
async def update_cluster(
    cluster_id: str,
    payload: ClusterRename,
    db: AsyncSession = Depends(get_db)
):
    """Rename a cluster."""
    logger.info(f"Updating cluster_id: {cluster_id} with new name: {payload.new_name}")
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    cluster = result.scalars().first()
    if not cluster:
        logger.error(f"Failed to update cluster: Cluster not found for id {cluster_id}")
        raise HTTPException(status_code=404, detail="Cluster not found")

    storage = StorageService(cluster.job_id)
    
    original_name = cluster.name
    if payload.new_name and payload.new_name != original_name:
        try:
            logger.debug(f"Renaming cluster directory from '{original_name}' to '{payload.new_name}'")
            storage.rename_cluster(original_name, payload.new_name)
            cluster.name = payload.new_name
        except Exception as e:
            logger.error(f"Failed to rename cluster directory: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))
            
    await db.commit()
    await db.refresh(cluster)
    logger.info(f"Cluster {cluster_id} renamed successfully to '{cluster.name}'")
    return cluster