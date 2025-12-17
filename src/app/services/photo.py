import logging
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cluster import Cluster
from app.models.photo import Photo
from app.schemas.photo import PhotoMove, PhotoUpdate
from app.services.cluster import ClusterService
from app.repository.photo import PhotoRepository
from app.repository.cluster import ClusterRepository

logger = logging.getLogger(__name__)


class PhotoService:
    def __init__(self, db: AsyncSession):
        self.photo_repo = PhotoRepository(db)
        self.cluster_repo = ClusterRepository(db)

    async def list_photos(
        self,
        job_id: str,
    ):
        """List all photos for a job."""
        logger.info(f"Listing photos for job_id: {job_id}")
        photos = await self.photo_repo.get_by_job_id(job_id)
        logger.info(f"Found {len(photos)} photos for job_id: {job_id}")
        return photos

    async def update_photo(
        self,
        photo_id: str,
        payload: PhotoUpdate,
    ):
        """Update photo metadata (e.g. labels)."""
        photo = await self.photo_repo.get_by_id(photo_id)

        if not photo:
            logger.error(f"Update failed: Photo {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")

        try:
            if payload.labels is not None:
                photo.labels = payload.labels

            await self.photo_repo.save(photo)
            logger.info(f"Successfully updated photo {photo_id}")
            return photo

        except Exception as e:
            logger.error(f"Error updating photo {photo_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def move_photo(self, photo_id: str, payload: PhotoMove, cluster_service: "ClusterService"):
        """Move a photo from one cluster to another with reordering."""
        target_cluster_id = payload.target_cluster_id
        new_order_index = payload.order_index

        # Find photo first to get job_id
        photo = await self.photo_repo.get_by_id(photo_id)
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
            cluster = await self.cluster_repo.get_by_name(photo.job_id, "reserve")
            if not cluster:
                # Assuming ClusterService can be used here or we should duplicate logic?
                # Using cluster_service as passed in argument is fine, but it creates circular dependency risk if types are imported.
                # Here cluster_service is passed as arg, so it's runtime dependency.
                # However, cleaner to use repo directly if simple. 
                # ClusterService.create_cluster handles reordering etc. Better to use it if possible.
                cluster, _ = await cluster_service.create_cluster(job_id=photo.job_id, order_index=-1, name="reserve")
            target_cluster_id = cluster.id

        try:
            # Case 1: Intra-cluster move (Reordering within same cluster)
            if source_cluster_id == target_cluster_id:
                if new_order_index is not None:
                    # Fetch all photos in this cluster
                    photos = await self.photo_repo.get_by_cluster_id_ordered(source_cluster_id)
                    
                    # Remove from current position
                    current_list = [p for p in photos if p.id != photo_id]

                    # Insert at new position
                    # Clamp index
                    if new_order_index < 0:
                        new_order_index = 0
                    if new_order_index > len(current_list):
                        new_order_index = len(current_list)

                    current_list.insert(new_order_index, photo)

                    # Update indices
                    for idx, p in enumerate(current_list):
                        p.order_index = idx

            # Case 2: Inter-cluster move
            else:
                # Check if target cluster exists
                target_cluster = await self.cluster_repo.get_by_id(target_cluster_id)
                if not target_cluster:
                    logger.error(f"Move failed: Target cluster {target_cluster_id} not found.")
                    raise HTTPException(status_code=404, detail="Target cluster not found")

                # 1. Remove from source (implicitly done by changing cluster_id)
                # Ideally, we should reorder source cluster to fill gaps, but it's optional for correctness
                # as long as order is relative. We can skip source reordering for performance.

                # 2. Add to target
                target_photos = await self.photo_repo.get_by_cluster_id_ordered(target_cluster_id)
                
                # Insert
                if new_order_index is None:
                    new_order_index = len(target_photos)

                if new_order_index < 0:
                    new_order_index = 0
                if new_order_index > len(target_photos):
                    new_order_index = len(target_photos)

                target_photos.insert(new_order_index, photo)

                # Update photo properties and target indices
                photo.cluster_id = target_cluster_id
                for idx, p in enumerate(target_photos):
                    p.order_index = idx

            await self.photo_repo.commit()
            await self.photo_repo.flush() # Is flush needed after commit? Usually commit implies flush.
            logger.info(f"Successfully moved photo {photo.id} to cluster {target_cluster_id}")

            # Check if source cluster became empty (only if it wasn't reserve and different from target)
            if source_cluster_id != target_cluster_id:
                # Check source cluster type or name. If it's 'reserve', don't delete.
                src_cluster_obj = await self.cluster_repo.get_by_id(source_cluster_id)

                if src_cluster_obj and src_cluster_obj.name != "reserve":
                    active_photos = await self.photo_repo.get_active_by_cluster_id(source_cluster_id)
                    if not active_photos:
                        await cluster_service.delete_cluster(job_id=photo.job_id, cluster_id=source_cluster_id)

        except Exception as e:
            logger.error(f"Error moving photo {photo.id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_photo(
        self,
        photo_id: str,
    ):
        """Delete a photo from a cluster."""
        photo = await self.photo_repo.get_by_id(photo_id)

        logger.info(f"Deleting photo {photo_id}")
        if not photo:
            logger.error(f"Delete failed: Photo {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")

        try:
            logger.debug(f"Deleting photo file '{photo.original_filename}'")
            photo.deleted_at = datetime.now()
            # await self.db.delete(photo) # Logic was commented out in original too
            await self.photo_repo.save(photo)
            logger.info(f"Successfully deleted photo {photo_id} from job {photo.job_id}")

        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}", exc_info=True)
            # TODO: Consider data consistency if file deletion fails but DB transaction proceeds
            raise HTTPException(status_code=500, detail=str(e))