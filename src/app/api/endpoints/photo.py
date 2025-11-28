import logging
from app.db.database import get_db
from app.models.cluster import Cluster
from app.models.photo import Photo
from app.schemas.photo import PhotoMove, PhotoResponse
from app.services.storage import StorageService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/photos", response_model=list[PhotoResponse])
async def list_photos(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all photos for a job."""
    logger.info(f"Listing photos for job_id: {job_id}")
    result = await db.execute(
        select(Photo)
        .where(Photo.job_id == job_id)
    )
    photos = result.scalars().all()
    logger.info(f"Found {len(photos)} photos for job_id: {job_id}")
    return [PhotoResponse(id=photo.id, 
                  job_id=photo.job_id, 
                  cluster_id=photo.cluster_id, 
                  storage_path=photo.storage_path, 
                  original_filename=photo.original_filename
            ) for photo in photos]


@router.post("/clusters/{cluster_id}/move-photo", status_code=status.HTTP_204_NO_CONTENT)
async def move_photo(
    cluster_id: str,
    payload: PhotoMove,
    db: AsyncSession = Depends(get_db)
):
    """Move a photo from one cluster to another."""
    logger.info(f"Moving photo {payload.photo_id} from cluster {cluster_id} to {payload.target_cluster_id}")

    # Find photo first to get job_id
    result = await db.execute(select(Photo).where(Photo.id == payload.photo_id))
    photo = result.scalars().first()
    if not photo:
        logger.error(f"Move failed: Photo with id {payload.photo_id} not found.")
        raise HTTPException(status_code=404, detail="Photo not found")

    if photo.cluster_id != cluster_id:
        logger.error(f"Move failed: Photo {payload.photo_id} does not belong to source cluster {cluster_id}.")
        raise HTTPException(status_code=400, detail="Photo does not belong to the source cluster")

    # Verify target cluster
    result = await db.execute(select(Cluster).where(Cluster.id == payload.target_cluster_id))
    target_cluster = result.scalars().first()
    if not target_cluster:
        logger.error(f"Move failed: Target cluster {payload.target_cluster_id} not found.")
        raise HTTPException(status_code=404, detail="Target cluster not found")
        
    # Get source cluster for its name
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    source_cluster = result.scalars().first()
    if not source_cluster:
         logger.error(f"Move failed: Source cluster {cluster_id} not found.")
         raise HTTPException(status_code=404, detail="Source cluster not found")

    storage = StorageService(photo.job_id)
    
    try:
        logger.debug(f"Moving photo file '{photo.original_filename}' from '{source_cluster.name}' to '{target_cluster.name}'")
        storage.move_photo(
            filename=photo.original_filename,
            from_cluster=source_cluster.name,
            to_cluster=target_cluster.name
        )
        
        # Update DB
        photo.cluster_id = target_cluster.id
        await db.commit()
        logger.info(f"Successfully moved photo {photo.id} to cluster {target_cluster.id}")
        
    except Exception as e:
        logger.error(f"Error moving photo {photo.id}: {e}", exc_info=True)
        # TODO: Consider rolling back file move if DB update fails
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clusters/{cluster_id}/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(
    cluster_id: str,
    photo_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a photo from a cluster."""
    logger.info(f"Deleting photo {photo_id} from cluster {cluster_id}")
    result = await db.execute(select(Photo).where(Photo.id == photo_id))
    photo = result.scalars().first()
    
    if not photo:
        logger.error(f"Delete failed: Photo {photo_id} not found.")
        raise HTTPException(status_code=404, detail="Photo not found")
        
    if photo.cluster_id != cluster_id:
        logger.error(f"Delete failed: Photo {photo_id} does not belong to cluster {cluster_id}.")
        raise HTTPException(status_code=400, detail="Photo does not belong to this cluster")
        
    # Get cluster for name
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    cluster = result.scalars().first()
    if not cluster:
        # This should ideally not happen if the above checks passed
        logger.error(f"Delete failed: Cluster {cluster_id} not found unexpectedly.")
        raise HTTPException(status_code=404, detail="Cluster not found")

    storage = StorageService(photo.job_id)
    
    try:
        logger.debug(f"Deleting photo file '{photo.original_filename}' from cluster '{cluster.name}'")
        storage.delete_photo(photo.original_filename, cluster.name)
        
        await db.delete(photo)
        await db.commit()
        logger.info(f"Successfully deleted photo {photo_id} from cluster {cluster_id}")
        
    except Exception as e:
        logger.error(f"Error deleting photo {photo_id}: {e}", exc_info=True)
        # TODO: Consider data consistency if file deletion fails but DB transaction proceeds
        raise HTTPException(status_code=500, detail=str(e))