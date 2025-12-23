from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.models.cluster import Cluster
from app.models.photo import Photo
from app.schemas.photo import PhotoMove, PhotoUpdate
from app.services.photo import PhotoService


@pytest.fixture
def mock_uow():
    uow = MagicMock()
    uow.photos = MagicMock()
    uow.clusters = MagicMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()
    uow.flush = AsyncMock()

    # Make the mock_uow behave like an async context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    
    # Custom side effect for __aexit__ to call commit or rollback on the mock_uow
    async def aexit_side_effect(exc_type, exc_val, exc_tb):
        if exc_type:
            await uow.rollback()
        else:
            await uow.commit()

    uow.__aexit__ = AsyncMock(side_effect=aexit_side_effect)

    return uow


@pytest.mark.asyncio
async def test_list_photos(mock_uow):
    service = PhotoService(mock_uow)
    mock_uow.photos.get_by_job_id = AsyncMock(return_value=[Photo(id="p1")])

    photos = await service.list_photos("j1")
    assert len(photos) == 1
    assert photos[0].id == "p1"


@pytest.mark.asyncio
async def test_update_photo(mock_uow):
    service = PhotoService(mock_uow)
    photo = Photo(id="p1", labels={})
    mock_uow.photos.get_by_id = AsyncMock(return_value=photo)
    mock_uow.photos.save = AsyncMock()

    payload = PhotoUpdate(labels={"tag": "true"})
    updated = await service.update_photo("p1", payload)

    assert updated.labels == {"tag": "true"}
    mock_uow.photos.save.assert_called_once()
    mock_uow.commit.assert_called_once() # Should now pass due to __aexit__ calling commit


@pytest.mark.asyncio
async def test_move_photo_intra_cluster(mock_uow):
    service = PhotoService(mock_uow)
    photo = Photo(id="p1", cluster_id="c1", order_index=0)
    mock_uow.photos.get_by_id = AsyncMock(return_value=photo)

    # Same cluster move
    payload = PhotoMove(target_cluster_id="c1", order_index=1)

    # Mock photos in cluster
    p2 = Photo(id="p2", cluster_id="c1", order_index=1)
    mock_uow.photos.get_by_cluster_id_ordered = AsyncMock(return_value=[photo, p2])

    await service.move_photo("p1", payload)

    mock_uow.commit.assert_called_once()
    # Logic is complex, but basic check is commit called


@pytest.mark.asyncio
async def test_move_photo_inter_cluster(mock_uow):
    service = PhotoService(mock_uow)
    photo = Photo(id="p1", cluster_id="c1", order_index=0)
    mock_uow.photos.get_by_id = AsyncMock(return_value=photo)

    # Different cluster
    payload = PhotoMove(target_cluster_id="c2", order_index=0)

    target_cluster = Cluster(id="c2")
    mock_uow.clusters.get_by_id = AsyncMock(return_value=target_cluster)
    mock_uow.photos.get_by_cluster_id_ordered = AsyncMock(return_value=[])

    # Source cluster cleanup check
    src_cluster = Cluster(id="c1", name="Group 1")
    # Need to handle get_by_id call order: first for target, second for source cleanup check
    mock_uow.clusters.get_by_id.side_effect = [target_cluster, src_cluster]

    mock_uow.photos.get_active_by_cluster_id = AsyncMock(return_value=[])  # Empty after move
    mock_uow.clusters.delete_by_id_returning_order_index = AsyncMock(return_value=None)
    mock_uow.photos.unassign_cluster = AsyncMock()

    await service.move_photo("p1", payload)

    assert photo.cluster_id == "c2"
    mock_uow.commit.assert_called_once() # Should now pass due to __aexit__ calling commit


@pytest.mark.asyncio
async def test_delete_photo(mock_uow):
    service = PhotoService(mock_uow)
    photo = Photo(id="p1")
    mock_uow.photos.get_by_id = AsyncMock(return_value=photo)
    mock_uow.photos.save = AsyncMock()

    await service.delete_photo("p1")

    assert photo.deleted_at is not None
    mock_uow.photos.save.assert_called_once()
    mock_uow.commit.assert_called_once() # Should now pass due to __aexit__ calling commit