import logging

from app.db.database import get_db
from app.schemas.photo import PhotoMove, PhotoResponse
from app.services.photo import PhotoService
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/photos", response_model=list[PhotoResponse])
async def list_photos(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    service = PhotoService(db)
    photos = await service.list_photos(job_id=job_id)
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
    service = PhotoService(db)
    await service.move_photo(cluster_id=cluster_id, payload=payload)


@router.delete("/clusters/{cluster_id}/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(
    cluster_id: str,
    photo_id: str,
    db: AsyncSession = Depends(get_db)
):
    service = PhotoService(db)
    await service.delete_photo(cluster_id=cluster_id, photo_id=photo_id)