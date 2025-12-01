import logging

from app.domain.storage import StorageService
from app.models.cluster import Cluster
from app.models.job import Job
from app.models.photo import Photo
from fastapi import HTTPException
from sqlalchemy import delete
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
            .options(
                selectinload(Cluster.photos.and_(Photo.deleted_at.is_(None)))
            )
        )
        clusters = result.scalars().all()
        logger.info(f"Found {len(clusters)} clusters for job_id: {job_id}")
        return clusters

    async def create_cluster(self, job_id: str, order_index: int, name: str = None, photo_ids: list[str] = None):
        """Create a new cluster."""
        logger.info(f"Creating cluster for job_id: {job_id} with name: {name}")
        result = await self.db.execute(
            select(Cluster)
            .where(Cluster.job_id == job_id)
            .order_by(Cluster.order_index.asc())
        )
        clusters = result.scalars().all()
        if not clusters:
            logger.error(f"Failed to create cluster: Job not found for id {job_id}")
            raise HTTPException(status_code=404, detail="Job not found")

        cluster_length = len(clusters)
        if order_index is None:
            order_index = cluster_length  # 맨 뒤
        if order_index < 0:
            order_index = 0
        if order_index > cluster_length:
            order_index = cluster_length

        for idx, c in enumerate(clusters):
            if idx >= order_index:
                # 새 클러스터가 들어갈 자리 이후 것들은 한 칸씩 뒤로
                c.order_index = idx + 1
            else:
                c.order_index = idx

        cluster = Cluster(
            job_id=job_id,
            name=name or "이름 없음",
            order_index=order_index or 0
        )
        self.db.add(cluster)
        await self.db.flush()

        if photo_ids:
            result = await self.db.execute(
                select(Photo)
                .where(Photo.id.in_(photo_ids))
                .order_by(Photo.order_index.asc())
            )
            photos = result.scalars().all()
            for idx, photo in enumerate(photos):
                photo.order_index = idx
                photo.cluster_id = cluster.id
        
        await self.db.commit()
        await self.db.refresh(cluster)
        logger.info(f"Cluster '{cluster.name}' created successfully with id: {cluster.id}")
        return cluster

    async def update_cluster(self, job_id: str, cluster_id: str, new_name: str = None, order_index: int = 0):
        """Rename a cluster."""
        logger.info(f"Updating cluster_id: {cluster_id} with new name: {new_name}")
        result = await self.db.execute(
            select(Cluster)
            .options(selectinload(Cluster.photos))
            .where(Cluster.id == cluster_id)
        )
        cluster = result.scalars().first()
        if not cluster:
            logger.error(f"Failed to update cluster: Cluster not found for id {cluster_id}")
            raise HTTPException(status_code=404, detail="Cluster not found")

        original_name = cluster.name
        if new_name and new_name != original_name:
            try:
                logger.debug(f"Renaming cluster directory from '{original_name}' to '{new_name}'")
                cluster.name = new_name
            except Exception as e:
                logger.error(f"Failed to rename cluster directory: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=str(e))
                
        await self.db.commit()
        await self.db.refresh(cluster)
        logger.info(f"Cluster {cluster_id} renamed successfully to '{cluster.name}'")
        return cluster
    
    async def delete_cluster(self, job_id: str, cluster_id: str):
        result = await self.db.execute(
            delete(Cluster)
            .where(Cluster.id == cluster_id)
            .returning(Cluster.order_index)
        )
        idx = result.scalars().first()
        if idx:
            result = await self.db.execute(
                select(Cluster)
                .where(Cluster.order_index > idx)
                .order_by(Cluster.order_index.asc())
            )
            clusters = result.scalars().all()
            for cluster in clusters:
                cluster.order_index -= 1
        await self.db.commit()