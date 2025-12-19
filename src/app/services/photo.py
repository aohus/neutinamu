import logging
from datetime import datetime

from fastapi import HTTPException

from app.common.uow import UnitOfWork
from app.models.cluster import Cluster
from app.schemas.photo import PhotoMove, PhotoUpdate

logger = logging.getLogger(__name__)


class PhotoService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def list_photos(
        self,
        job_id: str,
    ):
        """List all photos for a job."""
        logger.info(f"Listing photos for job_id: {job_id}")
        photos = await self.uow.photos.get_by_job_id(job_id)
        logger.info(f"Found {len(photos)} photos for job_id: {job_id}")
        return photos

    async def update_photo(
        self,
        photo_id: str,
        payload: PhotoUpdate,
    ):
        """Update photo metadata (e.g. labels)."""
        photo = await self.uow.photos.get_by_id(photo_id)

        if not photo:
            logger.error(f"Update failed: Photo {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")

        try:
            if payload.labels is not None:
                photo.labels = payload.labels

            await self.uow.photos.save(photo)
            logger.info(f"Successfully updated photo {photo_id}")
            return photo

        except Exception as e:
            logger.error(f"Error updating photo {photo_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def move_photo(self, photo_id: str, payload: PhotoMove):
        """Move a photo from one cluster to another with reordering."""
        target_cluster_id = payload.target_cluster_id
        new_order_index = payload.order_index

        # Find photo first to get job_id
        photo = await self.uow.photos.get_by_id(photo_id)
        if not photo:
            logger.error(f"Move failed: Photo with id {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")

        source_cluster_id = photo.cluster_id
        logger.info(
            f"Moving photo {photo_id} from cluster {source_cluster_id} to {target_cluster_id} at index {new_order_index}"
        )

        # Handle 'reserve' creation if needed
        if target_cluster_id == "reserve":
            # Check if reserve exists
            cluster = await self.uow.clusters.get_by_name(photo.job_id, "reserve")
            if not cluster:
                # Create reserve cluster directly using UoW
                cluster = Cluster(job_id=photo.job_id, name="reserve", order_index=-1)
                await self.uow.clusters.create(cluster)
                await self.uow.commit()
                await self.uow.refresh(cluster)
            target_cluster_id = cluster.id

        try:
            # Case 1: Intra-cluster move (Reordering within same cluster)
            if source_cluster_id == target_cluster_id:
                if new_order_index is not None:
                    photos = await self.uow.photos.get_by_cluster_id_ordered(source_cluster_id)
                    self._insert_and_reindex(photos, photo, new_order_index)

            # Case 2: Inter-cluster move
            else:
                # Check if target cluster exists
                target_cluster = await self.uow.clusters.get_by_id(target_cluster_id)
                if not target_cluster:
                    logger.error(f"Move failed: Target cluster {target_cluster_id} not found.")
                    raise HTTPException(status_code=404, detail="Target cluster not found")

                target_photos = await self.uow.photos.get_by_cluster_id_ordered(target_cluster_id)
                self._insert_and_reindex(target_photos, photo, new_order_index)

                # Update photo cluster
                photo.cluster_id = target_cluster_id

            await self.uow.commit()
            # await self.uow.flush() # Commit implies flush
            logger.info(f"Successfully moved photo {photo.id} to cluster {target_cluster_id}")

            # Check if source cluster became empty (only if it wasn't reserve and different from target)
            if source_cluster_id != target_cluster_id:
                src_cluster_obj = await self.uow.clusters.get_by_id(source_cluster_id)

                if src_cluster_obj and src_cluster_obj.name != "reserve":
                    active_photos = await self.uow.photos.get_active_by_cluster_id(source_cluster_id)
                    if not active_photos:
                        # Replicate delete logic using UoW to avoid circular dependency on ClusterService
                        await self.uow.photos.unassign_cluster(source_cluster_id)
                        idx = await self.uow.clusters.delete_by_id_returning_order_index(source_cluster_id)

                        if idx is not None:
                            clusters = await self.uow.clusters.get_clusters_after_order_for_job(photo.job_id, idx)
                            for cluster in clusters:
                                cluster.order_index -= 1
                        await self.uow.commit()

        except Exception as e:
            logger.error(f"Error moving photo {photo.id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def _insert_and_reindex(self, photos: list[Photo], photo: Photo, new_index: int | None):
        """
        Helper to insert a photo into a list at a specific index and re-index the list.
        Removes the photo from the list if it's already there (for intra-cluster moves).
        """
        # Remove photo if present to avoid duplication/confusion
        photos = [p for p in photos if p.id != photo.id]

        if new_index is None:
            new_index = len(photos)

        # Clamp index
        if new_index < 0:
            new_index = 0
        if new_index > len(photos):
            new_index = len(photos)

        photos.insert(new_index, photo)

        for idx, p in enumerate(photos):
            p.order_index = idx

    async def delete_photo(
        self,
        photo_id: str,
    ):
        """Delete a photo from a cluster."""
        photo = await self.uow.photos.get_by_id(photo_id)

        logger.info(f"Deleting photo {photo_id}")
        if not photo:
            logger.error(f"Delete failed: Photo {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")

        try:
            logger.debug(f"Deleting photo file '{photo.original_filename}'")
            photo.deleted_at = datetime.now()
            # await self.db.delete(photo) # Logic was commented out in original too
            await self.uow.photos.save(photo)
            logger.info(f"Successfully deleted photo {photo_id} from job {photo.job_id}")

        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}", exc_info=True)
            # TODO: Consider data consistency if file deletion fails but DB transaction proceeds
            raise HTTPException(status_code=500, detail=str(e))
