from app.db.database import get_db
from app.models.cluster import Cluster
from app.models.photo import Photo
from app.schemas.cluster import PhotoMove, PhotoResponse
from app.services.storage import StorageService
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

router = APIRouter()


@router.get("/jobs/{job_id}/photos", response_model=list[PhotoResponse])
async def list_photos(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """List all clusters for a job."""
    result = await db.execute(
        select(Photo)
        .where(Photo.job_id == job_id)
    )
    photos = result.scalars().all()
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
    # Verify target cluster
    result = await db.execute(select(Cluster).where(Cluster.id == payload.to_cluster)) # Assuming payload.to_cluster is ID
    target_cluster = result.scalars().first()
    if not target_cluster:
        raise HTTPException(status_code=404, detail="Target cluster not found")
        
    # Verify source cluster (optional if we trust the path param, but good for consistency)
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    source_cluster = result.scalars().first()
    if not source_cluster:
         raise HTTPException(status_code=404, detail="Source cluster not found")

    # Find photo
    # Assuming payload.photo_name is actually the photo ID or we query by name
    # The schema defined PhotoMove with photo_name, but API spec says photo_id. 
    # Let's assume we find by name within the cluster for now based on the schema I wrote earlier, 
    # OR better, query by filename and cluster_id.
    
    result = await db.execute(
        select(Photo)
        .where(Photo.cluster_id == cluster_id)
        .where(Photo.original_filename == payload.photo_name)
    )
    photo = result.scalars().first()
    
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found in source cluster")

    storage = StorageService(photo.job_id)
    
    try:
        # Move file on disk
        # We need cluster names for filesystem operations
        storage.move_photo(
            filename=photo.original_filename,
            from_cluster=source_cluster.name,
            to_cluster=target_cluster.name
        )
        
        # Update DB
        photo.cluster_id = target_cluster.id
        await db.commit()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clusters/{cluster_id}/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(
    cluster_id: str,
    photo_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a photo from a cluster."""
    result = await db.execute(select(Photo).where(Photo.id == photo_id))
    photo = result.scalars().first()
    
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
        
    if photo.cluster_id != cluster_id:
        raise HTTPException(status_code=400, detail="Photo does not belong to this cluster")
        
    # Get cluster for name
    result = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    cluster = result.scalars().first()
    
    storage = StorageService(photo.job_id)
    
    try:
        # Delete from disk
        storage.delete_photo(photo.original_filename, cluster.name)
        
        # Delete from DB (or soft delete)
        await db.delete(photo)
        await db.commit()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))