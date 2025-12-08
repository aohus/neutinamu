import logging
from datetime import datetime

from app.domain.storage import StorageService
from app.models.cluster import Cluster
from app.models.job import Job
from app.models.photo import Photo
from fastapi import HTTPException
from sqlalchemy import delete, update
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

        for c in clusters:
            idx = c.order_index
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

        photos = []
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
        return cluster, photos

    async def update_cluster(self, job_id: str, cluster_id: str, new_name: str = None, order_index: int = None):
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
        
        original_index = cluster.order_index
        if order_index and order_index != original_index and order_index > 0:
            try:
                logger.debug(f"order_index changed'{original_index}' to '{order_index}'")
                cluster.order_index = order_index
            except Exception as e:
                logger.error(f"Failed to rename: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=str(e))
            
        await self.db.commit()
        await self.db.refresh(cluster)
        logger.info(f"Cluster {cluster_id} renamed successfully to '{cluster.name or cluster.order_index}'")
        return cluster
    
    async def delete_cluster(self, job_id: str, cluster_id: str):
        # Unassign photos first to avoid Foreign Key violation
        await self.db.execute(
            update(Photo)
            .where(Photo.cluster_id == cluster_id)
            .values(cluster_id=None,
                    deleted_at=datetime.now())
        )

        result = await self.db.execute(
            delete(Cluster)
            .where(Cluster.id == cluster_id)
            .returning(Cluster.order_index)
        )
        idx = result.scalars().first()
        
        if idx is not None:
            result = await self.db.execute(
                select(Cluster)
                .where(Cluster.order_index > idx)
                .order_by(Cluster.order_index.asc())
            )
            clusters = result.scalars().all()
            for cluster in clusters:
                cluster.order_index -= 1
        await self.db.commit()

    async def add_photos(self, job_id: str, cluster_id: str, photo_ids: list[str]):
        """Add photos to a cluster."""
        logger.info(f"Adding {len(photo_ids)} photos to cluster {cluster_id}")
        
        # Check cluster exists
        result = await self.db.execute(select(Cluster).where(Cluster.id == cluster_id))
        cluster = result.scalars().first()
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        # Get current photos count to append
        # We need to know the max order_index currently in the cluster to append correctly
        result = await self.db.execute(
            select(Photo).where(Photo.cluster_id == cluster_id)
        )
        current_photos = result.scalars().all()
        start_index = len(current_photos)

        # Update photos
        # Fetch photos to ensure they exist and we can update them
        result = await self.db.execute(
            select(Photo).where(Photo.id.in_(photo_ids))
        )
        photos_to_move = result.scalars().all()
        
        for idx, photo in enumerate(photos_to_move):
            photo.cluster_id = cluster_id
            photo.order_index = start_index + idx
        
        await self.db.commit()
        return

    async def sync_clusters(self, job_id: str, cluster_data: list[dict]):
        """Sync cluster order and photo assignments."""
        logger.info(f"Syncing clusters for job {job_id}")
        
        # To optimize, we can fetch all clusters and photos for this job first
        # But for simplicity and safety with SQLAlchemy async session tracking, 
        # we'll iterate. Given the likely size (< 100 clusters, < 1000 photos), it should be acceptable.
        # Optimization: Check if data actually changed? 
        # Frontend sends everything, so we overwrite.
        
        for c_data in cluster_data:
            c_id = c_data.get('id')
            c_name = c_data.get('name')
            c_order = c_data.get('order_index')
            c_photo_ids = c_data.get('photo_ids', [])
            
            # Update Cluster
            cluster = await self.db.get(Cluster, c_id)
            if cluster and cluster.job_id == job_id:
                if cluster.name != c_name:
                    cluster.name = c_name
                    
                if cluster.order_index != c_order:
                    cluster.order_index = c_order
                
                # Update Photos
                if c_photo_ids:
                    # Fetch photos that need updating
                    # We only need to update if they are different.
                    # But simply overwriting order_index and cluster_id is fast enough in memory
                    result = await self.db.execute(select(Photo).where(Photo.id.in_(c_photo_ids)))
                    photos = result.scalars().all()
                    photo_map = {p.id: p for p in photos}
                    
                    for p_idx, p_id in enumerate(c_photo_ids):
                        if p_id in photo_map:
                            photo = photo_map[p_id]
                            # Only update if changed to avoid unnecessary db writes?
                            # SQLAlchemy tracks changes, so assignment is cheap if value is same.
                            photo.cluster_id = c_id
                            photo.order_index = p_idx
            
        await self.db.commit()