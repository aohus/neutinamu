import logging

from app.domain.storage import StorageService
from app.models.cluster import Cluster
from app.models.job import Job
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class ClusterService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_clusters(self, job_id: str):
        """List all clusters for a job."""
        logger.info(f"Listing clusters for job_id: {job_id}")
        result = await self.db.execute(
            select(Cluster)
            .where(Cluster.job_id == job_id)
            .options(selectinload(Cluster.photos))
        )
        clusters = result.scalars().all()
        logger.info(f"Found {len(clusters)} clusters for job_id: {job_id}")
        return clusters

    async def create_cluster(self, job_id: str, name: str = None, order_index: int = 0):
        """Create a new empty cluster."""
        logger.info(f"Creating cluster for job_id: {job_id} with name: {name}")
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        if not result.scalars().first():
            logger.error(f"Failed to create cluster: Job not found for id {job_id}")
            raise HTTPException(status_code=404, detail="Job not found")

        storage = StorageService(job_id)
        if name:
            try:
                storage.create_cluster(name)
                logger.debug(f"Created directory for cluster: {name}")
            except Exception as e:
                logger.error(f"Failed to create cluster directory: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=str(e))

        cluster = Cluster(
            job_id=job_id,
            name=name or "New Cluster",
            order_index=order_index or 0
        )
        self.db.add(cluster)
        await self.db.commit()
        await self.db.refresh(cluster)
        logger.info(f"Cluster '{cluster.name}' created successfully with id: {cluster.id}")
        return cluster

    async def update_cluster(self, job_id: str, cluster_id: str, new_name: str = None, order_index: int = 0):
        """Rename a cluster."""
        logger.info(f"Updating cluster_id: {cluster_id} with new name: {new_name}")
        result = await self.db.execute(select(Cluster).where(Cluster.id == cluster_id))
        cluster = result.scalars().first()
        if not cluster:
            logger.error(f"Failed to update cluster: Cluster not found for id {cluster_id}")
            raise HTTPException(status_code=404, detail="Cluster not found")

        storage = StorageService(cluster.job_id)
        
        original_name = cluster.name
        if new_name and new_name != original_name:
            try:
                logger.debug(f"Renaming cluster directory from '{original_name}' to '{new_name}'")
                storage.rename_cluster(original_name, new_name)
                cluster.name = new_name
            except Exception as e:
                logger.error(f"Failed to rename cluster directory: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=str(e))
                
        await self.db.commit()
        await self.db.refresh(cluster)
        logger.info(f"Cluster {cluster_id} renamed successfully to '{cluster.name}'")
        return cluster