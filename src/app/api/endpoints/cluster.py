import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.cluster import (
    ClusterAddPhotosRequest,
    ClusterCreateRequest,
    ClusterResponse,
    ClusterSyncRequest,
    ClusterUpdateRequest,
)
from app.schemas.photo import PhotoResponse
from app.services.cluster import ClusterService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/jobs/{job_id}/clusters", response_model=list[ClusterResponse])
async def list_clusters(job_id: str, db: AsyncSession = Depends(get_db)):
    service = ClusterService(db)
    clusters = await service.list_clusters(job_id=job_id)
    return [
        ClusterResponse(
            id=cluster.id,
            job_id=cluster.job_id,
            name=cluster.name,
            order_index=cluster.order_index,
            photos=[
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
                for photo in cluster.photos
            ],
        )
        for cluster in clusters
    ]


@router.put("/jobs/{job_id}/clusters/sync", status_code=status.HTTP_204_NO_CONTENT)
async def sync_clusters(job_id: str, payload: ClusterSyncRequest, db: AsyncSession = Depends(get_db)):
    service = ClusterService(db)
    # Convert Pydantic models to list of dicts
    cluster_data = [c.model_dump() for c in payload.clusters]
    await service.sync_clusters(job_id=job_id, cluster_data=cluster_data)
    return


@router.post("/jobs/{job_id}/clusters", response_model=ClusterResponse, status_code=status.HTTP_201_CREATED)
async def create_cluster(job_id: str, payload: ClusterCreateRequest, db: AsyncSession = Depends(get_db)):
    service = ClusterService(db)
    order_index = payload.order_index if payload.order_index else 0
    name = payload.name if payload.name else None
    photo_ids = payload.photo_ids if payload.photo_ids else []
    cluster, photos = await service.create_cluster(
        job_id=job_id, order_index=order_index, name=name, photo_ids=photo_ids
    )
    return ClusterResponse(
        id=cluster.id,
        job_id=cluster.job_id,
        name=cluster.name,
        order_index=cluster.order_index,
        photos=[
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
                thumbnail_path=photo.thumbnail_url,
            )
            for photo in photos
        ],
    )


@router.post("/clusters/{cluster_id}/add_photos", status_code=status.HTTP_204_NO_CONTENT)
async def add_photos_to_cluster(cluster_id: str, payload: ClusterAddPhotosRequest, db: AsyncSession = Depends(get_db)):
    service = ClusterService(db)
    await service.add_photos(job_id="", cluster_id=cluster_id, photo_ids=payload.photo_ids)
    return


@router.patch("/clusters/{cluster_id}", response_model=ClusterResponse)
async def update_cluster(cluster_id: str, payload: ClusterUpdateRequest, db: AsyncSession = Depends(get_db)):
    service = ClusterService(db)
    new_name = payload.new_name if payload.new_name else None
    order_index = payload.order_index if payload.order_index else 0
    cluster = await service.update_cluster(job_id="", cluster_id=cluster_id, new_name=new_name, order_index=order_index)
    return ClusterResponse(
        id=cluster.id,
        job_id=cluster.job_id,
        name=cluster.name,
        order_index=cluster.order_index,
        photos=[
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
                thumbnail_path=photo.thumbnail_url,
            )
            for photo in cluster.photos
        ],
    )


@router.delete("/clusters/{cluster_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cluster(cluster_id: str, db: AsyncSession = Depends(get_db)):
    service = ClusterService(db)
    await service.delete_cluster(job_id="", cluster_id=cluster_id)
