import logging
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cluster import Cluster
from app.models.job import Job
from app.models.photo import Photo
from app.repository.cluster import ClusterRepository
from app.repository.job import JobRepository
from app.repository.photo import PhotoRepository

logger = logging.getLogger(__name__)


class ClusterService:
    def __init__(self, db: AsyncSession):
        self.cluster_repo = ClusterRepository(db)
        self.job_repo = JobRepository(db)
        self.photo_repo = PhotoRepository(db)

    async def list_clusters(self, job_id: str):
        """List all clusters for a job."""
        logger.info(f"Listing clusters for job_id: {job_id}")

        # Fetch the job to check its status
        job = await self.job_repo.get_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Only allow listing clusters for jobs with specific statuses
        allowed_statuses = ["CREATED", "FAILED", "COMPLETED"]
        if job.status not in allowed_statuses:
            logger.info(
                f"Job {job_id} has status {job.status}, which is not in allowed statuses {allowed_statuses}. Returning empty cluster list."
            )
            return []

        clusters = await self.cluster_repo.get_by_job_id(job_id)
        logger.info(f"Found {len(clusters)} clusters for job_id: {job_id}")
        return clusters

    async def create_cluster(self, job_id: str, order_index: int, name: str = None, photo_ids: list[str] = None):
        """Create a new cluster."""
        logger.info(f"Creating cluster for job_id: {job_id} with name: {name}")
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.construction_type:
            name = job.construction_type
        else:
            name = job.title

        clusters = await self.cluster_repo.get_ordered_by_job_id(job_id)
        
        # Original logic checked for empty clusters and raised error if not found? 
        # "if not clusters: raise ... Job not found". This logic seems flawed if user deleted all clusters?
        # But maybe logical assumption is at least one cluster exists or job creation creates one?
        # Let's preserve logic but adapt if necessary.
        # Actually, if job exists, clusters might be empty initially?
        # If the intention was to check job existence again, we already did `get_by_id`.
        # I will remove the "Job not found" check based on empty clusters as it's redundant/incorrect if job found.
        
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

        cluster = Cluster(job_id=job_id, name=name or "이름 없음", order_index=order_index or 0)
        await self.cluster_repo.create(cluster)
        
        photos = []
        if photo_ids:
            photos = await self.photo_repo.get_by_ids(photo_ids)
            for idx, photo in enumerate(photos):
                photo.order_index = idx
                photo.cluster_id = cluster.id

        await self.cluster_repo.commit()
        await self.cluster_repo.db.refresh(cluster) # Refreshing via db session in repo? repo.save does refresh.
        # But here we committed after modifying photos too. 
        # Let's trust `repo.save` style or manual.
        
        logger.info(f"Cluster '{cluster.name}' created successfully with id: {cluster.id}")
        return cluster, photos

    async def update_cluster(self, job_id: str, cluster_id: str, new_name: str = None, order_index: int = None):
        """Rename a cluster."""
        logger.info(f"Updating cluster_id: {cluster_id} with new name: {new_name}")
        cluster = await self.cluster_repo.get_by_id_with_photos(cluster_id)
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

        await self.cluster_repo.save(cluster)
        logger.info(f"Cluster {cluster_id} renamed successfully to '{cluster.name or cluster.order_index}'")
        return cluster

    async def delete_cluster(self, job_id: str, cluster_id: str):
        # Unassign photos first to avoid Foreign Key violation
        await self.photo_repo.unassign_cluster(cluster_id)

        idx = await self.cluster_repo.delete_by_id_returning_order_index(cluster_id)

        if idx is not None:
            # FIX: Adding job_id filter which was missing in original code, to avoid reordering other jobs' clusters
            clusters = await self.cluster_repo.get_clusters_after_order_for_job(job_id, idx)
            for cluster in clusters:
                cluster.order_index -= 1
        
        await self.cluster_repo.commit()

    async def add_photos(self, job_id: str, cluster_id: str, photo_ids: list[str]):
        """Add photos to a cluster."""
        logger.info(f"Adding {len(photo_ids)} photos to cluster {cluster_id}")

        # Check cluster exists
        cluster = await self.cluster_repo.get_by_id(cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        # Get current photos count to append
        current_photos = await self.photo_repo.get_by_cluster_id(cluster_id)
        start_index = len(current_photos)

        # Update photos
        photos_to_move = await self.photo_repo.get_by_ids(photo_ids)

        for idx, photo in enumerate(photos_to_move):
            photo.cluster_id = cluster_id
            photo.order_index = start_index + idx

        await self.photo_repo.commit()
        return

    async def sync_clusters(self, job_id: str, cluster_data: list[dict]):
        """Sync cluster order and photo assignments."""
        logger.info(f"Syncing clusters for job {job_id}")

        for c_data in cluster_data:
            c_id = c_data.get("id")
            c_name = c_data.get("name")
            c_order = c_data.get("order_index")
            c_photo_ids = c_data.get("photo_ids", [])

            # Update Cluster
            cluster = await self.cluster_repo.get_by_id(c_id)
            if cluster and cluster.job_id == job_id:
                if cluster.name != c_name:
                    cluster.name = c_name

                if cluster.order_index != c_order:
                    cluster.order_index = c_order

                # Update Photos
                if c_photo_ids:
                    photos = await self.photo_repo.get_by_ids(c_photo_ids)
                    photo_map = {p.id: p for p in photos}

                    for p_idx, p_id in enumerate(c_photo_ids):
                        if p_id in photo_map:
                            photo = photo_map[p_id]
                            photo.cluster_id = c_id
                            photo.order_index = p_idx

        await self.cluster_repo.commit()