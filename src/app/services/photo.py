import logging
from datetime import datetime

from app.models.photo import Photo
from app.schemas.photo import PhotoMove, PhotoResponse
from app.services.cluster import ClusterService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = logging.getLogger(__name__)


class PhotoService:
    def __init__(self, db: AsyncSession):
        self.db = db
        
    async def list_photos(
        self,
        job_id: str,
    ):
        """List all photos for a job."""
        logger.info(f"Listing photos for job_id: {job_id}")
        result = await self.db.execute(
            select(Photo)
            .where(Photo.job_id == job_id)
        )
        photos = result.scalars().all()
        logger.info(f"Found {len(photos)} photos for job_id: {job_id}")
        return photos

    async def move_photo(
        self,
        photo_id: str,
        payload: PhotoMove,
        cluster_service: "ClusterService"
    ):
        """Move a photo from one cluster to another."""
        target_cluster_id = payload.target_cluster_id

        # Find photo first to get job_id
        result = await self.db.execute(select(Photo).where(Photo.id == photo_id))
        photo = result.scalars().first()
        cluster_id = photo.cluster_id
        logger.info(f"Moving photo {photo_id} from cluster {cluster_id} to {target_cluster_id}")
        if not photo:
            logger.error(f"Move failed: Photo with id {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")

        if photo.cluster_id != cluster_id:
            logger.error(f"Move failed: Photo {photo_id} does not belong to source cluster {cluster_id}.")
            raise HTTPException(status_code=400, detail="Photo does not belong to the source cluster")
        
        try:
            logger.debug(f"Moving photo file '{photo.original_filename}' from '{photo.cluster_id}' to '{target_cluster_id}'")
            # Update DB
            photo.cluster_id = target_cluster_id
            await self.db.commit()
            await self.db.flush()
            logger.info(f"Successfully moved photo {photo.id} to cluster {target_cluster_id}")
        except Exception as e:
            logger.error(f"Error moving photo {photo.id}: {e}", exc_info=True)
            # TODO: Consider rolling back file move if DB update fails
            raise HTTPException(status_code=500, detail=str(e))

        result = await self.db.execute(select(Photo).where(Photo.cluster_id == cluster_id))
        if not result.scalars().first():
            await cluster_service.delete_cluster(job_id='', cluster_id=cluster_id)

    # async def move_photo(
    #     self,
    #     cluster_id: str,
    #     payload: PhotoMove,
    # ):
    #     """Move a photo from one cluster to another."""
    #     logger.info(f"Moving photo {payload.photo_id} from cluster {cluster_id} to {payload.target_cluster_id}")

    #     # Find photo first to get job_id
    #     result = await self.db.execute(select(Photo).where(Photo.id == payload.photo_id))
    #     photo = result.scalars().first()
    #     if not photo:
    #         logger.error(f"Move failed: Photo with id {payload.photo_id} not found.")
    #         raise HTTPException(status_code=404, detail="Photo not found")

    #     if photo.cluster_id != cluster_id:
    #         logger.error(f"Move failed: Photo {payload.photo_id} does not belong to source cluster {cluster_id}.")
    #         raise HTTPException(status_code=400, detail="Photo does not belong to the source cluster")

    #     # Verify target cluster
    #     result = await self.db.execute(select(Cluster).where(Cluster.id == payload.target_cluster_id))
    #     target_cluster = result.scalars().first()
    #     if not target_cluster:
    #         logger.error(f"Move failed: Target cluster {payload.target_cluster_id} not found.")
    #         raise HTTPException(status_code=404, detail="Target cluster not found")
            
    #     # Get source cluster for its name
    #     result = await self.db.execute(select(Cluster).where(Cluster.id == cluster_id))
    #     source_cluster = result.scalars().first()
    #     if not source_cluster:
    #         logger.error(f"Move failed: Source cluster {cluster_id} not found.")
    #         raise HTTPException(status_code=404, detail="Source cluster not found")
    #     storage = StorageService(photo.job_id)
        
    #     try:
    #         logger.debug(f"Moving photo file '{photo.original_filename}' from '{source_cluster.name}' to '{target_cluster.name}'")
    #         storage.move_photo(
    #             filename=photo.original_filename,
    #             from_cluster=source_cluster.name,
    #             to_cluster=target_cluster.name
    #         )
            
    #         # Update DB
    #         photo.cluster_id = target_cluster.id
    #         await self.db.commit()
    #         logger.info(f"Successfully moved photo {photo.id} to cluster {target_cluster.id}")
            
    #     except Exception as e:
    #         logger.error(f"Error moving photo {photo.id}: {e}", exc_info=True)
    #         # TODO: Consider rolling back file move if DB update fails
    #         raise HTTPException(status_code=500, detail=str(e))

    async def delete_photo(
        self,
        photo_id: str,
    ):
        """Delete a photo from a cluster."""
        result = await self.db.execute(select(Photo).where(Photo.id == photo_id))
        photo = result.scalars().first()
        
        logger.info(f"Deleting photo {photo_id} from cluster {photo.cluster_id}")
        if not photo:
            logger.error(f"Delete failed: Photo {photo_id} not found.")
            raise HTTPException(status_code=404, detail="Photo not found")
            
        # if photo.cluster_id != cluster_id:
        #     logger.error(f"Delete failed: Photo {photo_id} does not belong to cluster {cluster_id}.")
        #     raise HTTPException(status_code=400, detail="Photo does not belong to this cluster")
            
        # Get cluster for name
        # result = await self.db.execute(select(Cluster).where(Cluster.id == cluster_id))
        # cluster = result.scalars().first()
        # if not cluster:
        #     # This should ideally not happen if the above checks passed
        #     logger.error(f"Delete failed: Cluster {cluster_id} not found unexpectedly.")
        #     raise HTTPException(status_code=404, detail="Cluster not found")
        
        try:
            logger.debug(f"Deleting photo file '{photo.original_filename}'")
            photo.deleted_at = datetime.now()
            # await self.db.delete(photo)
            await self.db.commit()
            logger.info(f"Successfully deleted photo {photo_id} from job {photo.job_id}")
            
        except Exception as e:
            logger.error(f"Error deleting photo {photo_id}: {e}", exc_info=True)
            # TODO: Consider data consistency if file deletion fails but DB transaction proceeds
            raise HTTPException(status_code=500, detail=str(e))