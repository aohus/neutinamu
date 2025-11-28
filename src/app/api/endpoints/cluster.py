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


@router.get("/jobs/{job_id}/clusters", response_model=List[ClusterDetail])
async def list_clusters(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all clusters for a job."""
    result = await db.execute(
        select(Cluster)
        .where(Cluster.job_id == job_id)
        .options(selectinload(Cluster.photos))
        .order_by(Cluster.order_index)
    )
    clusters = result.scalars().all()
    return clusters

@router.post("/jobs/{job_id}/clusters", response_model=ClusterDetail, status_code=status.HTTP_201_CREATED)
async def create_cluster(
    job_id: str,
    payload: ClusterCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new empty cluster."""
    result = await db.execute(select(Job).where(Job.id == job_id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Job not found")

    storage = StorageService(job_id)
    
    # Check if cluster name exists in DB (optional, but good practice)
    # Create directory
    if payload.name:
        try:
            storage.create_cluster(payload.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    cluster = Cluster(
        job_id=job_id,
        name=payload.name or "New Cluster",
        order_index=payload.order_index or 0
    )
    db.add(cluster)
    await db.commit()
    await db.refresh(cluster)
    return cluster

@router.patch("/clusters/{cluster_id}", response_model=ClusterDetail)
async def update_cluster(
    cluster_id: str,
    payload: ClusterRename,
    db: AsyncSession = Depends(get_db)
):
    """Rename a cluster."""
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    cluster = result.scalars().first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    storage = StorageService(cluster.job_id)
    
    if payload.new_name and payload.new_name != cluster.name:
        try:
            storage.rename_cluster(cluster.name, payload.new_name)
            cluster.name = payload.new_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    await db.commit()
    await db.refresh(cluster)
    return cluster