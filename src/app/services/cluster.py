import logging
from datetime import datetime

from fastapi import HTTPException

from app.common.uow import UnitOfWork
from app.models.cluster import Cluster

logger = logging.getLogger(__name__)


class ClusterService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def list_clusters(self, job_id: str):
        """List all clusters for a job."""
        logger.info(f"Listing clusters for job_id: {job_id}")

        # Fetch the job to check its status
        job = await self.uow.jobs.get_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Only allow listing clusters for jobs with specific statuses
        allowed_statuses = ["CREATED", "FAILED", "COMPLETED"]
        if job.status not in allowed_statuses:
            logger.info(
                f"Job {job_id} has status {job.status}, which is not in allowed statuses {allowed_statuses}. Returning empty cluster list."
            )
            return []

        clusters = await self.uow.clusters.get_by_job_id(job_id)
        logger.info(f"Found {len(clusters)} clusters for job_id: {job_id}")
        return clusters

    async def create_cluster(self, job_id: str, order_index: int, name: str = None, photo_ids: list[str] = None):
        """Create a new cluster."""
        logger.info(f"Creating cluster for job_id: {job_id} with name: {name}")
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.construction_type:
            name = job.construction_type
        else:
            name = job.title

        clusters = await self.uow.clusters.get_ordered_by_job_id(job_id)
        
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

        async with self.uow:
            cluster = Cluster(job_id=job_id, name=name or "이름 없음", order_index=order_index or 0)
            await self.uow.clusters.create(cluster)
            
            photos = []
            if photo_ids:
                photos = await self.uow.photos.get_by_ids(photo_ids)
                for idx, photo in enumerate(photos):
                    photo.order_index = idx
                    photo.cluster_id = cluster.id

            await self.uow.refresh(cluster)
        
        logger.info(f"Cluster '{cluster.name}' created successfully with id: {cluster.id}")
        return cluster, photos

    async def update_cluster(self, job_id: str, cluster_id: str, new_name: str = None, order_index: int = None):
        """Rename a cluster."""
        logger.info(f"Updating cluster_id: {cluster_id} with new name: {new_name}")
        
        async with self.uow:
            cluster = await self.uow.clusters.get_by_id_with_photos(cluster_id)
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

            await self.uow.clusters.save(cluster)
        
        logger.info(f"Cluster {cluster_id} renamed successfully to '{cluster.name or cluster.order_index}'")
        return cluster

    async def delete_cluster(self, job_id: str, cluster_id: str):
        async with self.uow:
            # Unassign photos first to avoid Foreign Key violation
            await self.uow.photos.unassign_cluster(cluster_id)

            idx = await self.uow.clusters.delete_by_id_returning_order_index(cluster_id)

            if idx is not None:
                clusters = await self.uow.clusters.get_clusters_after_order_for_job(job_id, idx)
                for cluster in clusters:
                    cluster.order_index -= 1

    async def add_photos(self, job_id: str, cluster_id: str, photo_ids: list[str]):
        """Add photos to a cluster."""
        logger.info(f"Adding {len(photo_ids)} photos to cluster {cluster_id}")

        async with self.uow:
            # Check cluster exists
            cluster = await self.uow.clusters.get_by_id(cluster_id)
            if not cluster:
                raise HTTPException(status_code=404, detail="Cluster not found")

            # Get current photos count to append
            current_photos = await self.uow.photos.get_by_cluster_id(cluster_id)
            start_index = len(current_photos)

            # Update photos
            photos_to_move = await self.uow.photos.get_by_ids(photo_ids)

            for idx, photo in enumerate(photos_to_move):
                photo.cluster_id = cluster_id
                photo.order_index = start_index + idx

        return

    async def sync_clusters(self, job_id: str, cluster_data: list[dict]):
        """Sync cluster order and photo assignments."""
        logger.info(f"Syncing clusters for job {job_id}")

        async with self.uow:
            for c_data in cluster_data:
                c_id = c_data.get("id")
                c_name = c_data.get("name")
                c_order = c_data.get("order_index")
                c_photo_ids = c_data.get("photo_ids", [])

                # Update Cluster
                cluster = await self.uow.clusters.get_by_id(c_id)
                if cluster and cluster.job_id == job_id:
                    if cluster.name != c_name:
                        cluster.name = c_name

                    if cluster.order_index != c_order:
                        cluster.order_index = c_order

                    # Update Photos
                    if c_photo_ids:
                        photos = await self.uow.photos.get_by_ids(c_photo_ids)
                        photo_map = {p.id: p for p in photos}

                        for p_idx, p_id in enumerate(c_photo_ids):
                            if p_id in photo_map:
                                photo = photo_map[p_id]
                                photo.cluster_id = c_id
                                photo.order_index = p_idx