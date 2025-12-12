import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.domain.storage.factory import get_storage_client  # Import this
from app.schemas.photo import PhotoMove, PhotoResponse, PhotoUpdate
from app.services.cluster import ClusterService
from app.services.photo import PhotoService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/photos", response_model=list[PhotoResponse])
async def list_photos(job_id: str, db: AsyncSession = Depends(get_db)):
    service = PhotoService(db)
    photos = await service.list_photos(job_id=job_id)
    return [
        PhotoResponse(
            id=photo.id,
            job_id=photo.job_id,
            order_index=photo.order_index,
            timestamp=photo.meta_timestamp,
            labels=photo.labels,
            cluster_id=photo.cluster_id,
            storage_path=photo.storage_path,
            original_filename=photo.original_filename,
            url=photo.url,
            thumbnail_path=photo.thumbnail_path,
            thumbnail_url=photo.thumbnail_url,
        )
        for photo in photos
    ]


@router.patch("/photos/{photo_id}", response_model=PhotoResponse)
async def update_photo(photo_id: str, payload: PhotoUpdate, db: AsyncSession = Depends(get_db)):
    service = PhotoService(db)
    photo = await service.update_photo(photo_id=photo_id, payload=payload)
    return PhotoResponse(
        id=photo.id,
        job_id=photo.job_id,
        order_index=photo.order_index,
        timestamp=photo.meta_timestamp,
        labels=photo.labels,
        cluster_id=photo.cluster_id,
        storage_path=photo.storage_path,
        original_filename=photo.original_filename,
        url=photo.url,
        thumbnail_path=photo.thumbnail_path,
        thumbnail_url=photo.thumbnail_url,
    )


@router.post("/photos/{photo_id}/move", status_code=status.HTTP_204_NO_CONTENT)
async def move_photo(photo_id: str, payload: PhotoMove, db: AsyncSession = Depends(get_db)):
    service = PhotoService(db)
    cluster_service = ClusterService(db)
    await service.move_photo(photo_id=photo_id, payload=payload, cluster_service=cluster_service)


@router.delete("/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(photo_id: str, db: AsyncSession = Depends(get_db)):
    service = PhotoService(db)
    await service.delete_photo(photo_id=photo_id)
