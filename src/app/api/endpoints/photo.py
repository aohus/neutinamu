import logging

from app.db.database import get_db
from app.schemas.photo import PhotoMove, PhotoResponse
from app.services.cluster import ClusterService
from app.services.photo import PhotoService
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.storage.factory import get_storage_client # Import this

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/photos", response_model=list[PhotoResponse])
async def list_photos(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    service = PhotoService(db)
    photos = await service.list_photos(job_id=job_id)
    storage = get_storage_client() # Get storage client
    return [PhotoResponse(id=photo.id, 
                          job_id=photo.job_id, 
                          order_index=photo.order_index,
                          cluster_id=photo.cluster_id, 
                          storage_path=photo.storage_path, 
                          original_filename=photo.original_filename,
                          url=storage.get_url(photo.storage_path), # Populate URL
                          thumbnail_path=storage.get_url(photo.thumbnail_path) if photo.thumbnail_path else None
                          ) for photo in photos]

@router.post("/photos/{photo_id}/move", status_code=status.HTTP_204_NO_CONTENT)
async def move_photo(
    photo_id: str,
    payload: PhotoMove,
    db: AsyncSession = Depends(get_db)
):
    service = PhotoService(db)
    cluster_service = ClusterService(db)
    await service.move_photo(photo_id=photo_id, payload=payload, cluster_service=cluster_service)

# @router.post("/clusters/{cluster_id}/move-photo", status_code=status.HTTP_204_NO_CONTENT)
# async def move_photo(
#     cluster_id: str,
#     payload: PhotoMove,
#     db: AsyncSession = Depends(get_db)
# ):
#     service = PhotoService(db)
#     await service.move_photo(cluster_id=cluster_id, payload=payload)


@router.delete("/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(
    photo_id: str,
    db: AsyncSession = Depends(get_db)
):
    service = PhotoService(db)
    await service.delete_photo(photo_id=photo_id)

