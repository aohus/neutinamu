from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from app.models.photo import Photo


@pytest.mark.asyncio
async def test_list_photos(client):
    job_id = "job_123"

    # Mock Photo object
    photo = MagicMock(spec=Photo)
    photo.id = "p1"
    photo.job_id = job_id
    photo.order_index = 0
    photo.meta_timestamp = None
    photo.labels = {}
    photo.cluster_id = None
    photo.storage_path = "/path/to/photo.jpg"
    photo.original_filename = "photo.jpg"
    photo.url = "http://url"
    photo.thumbnail_path = "/path/thumb.jpg"
    photo.thumbnail_url = "http://thumb"

    with patch("app.api.endpoints.photo.PhotoService") as MockService:
        mock_service = MockService.return_value
        mock_service.list_photos = AsyncMock(return_value=[photo])

        response = await client.get(f"/api/jobs/{job_id}/photos")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "p1"


@pytest.mark.asyncio
async def test_update_photo(client):
    photo_id = "p1"
    payload = {"labels": {"tag1": "value"}}

    photo = MagicMock(spec=Photo)
    photo.id = photo_id
    photo.job_id = "job_123"
    photo.labels = {"tag1": "value"}
    # ... other fields need to be present for response model ...
    photo.order_index = 0
    photo.meta_timestamp = None
    photo.cluster_id = None
    photo.storage_path = "path"
    photo.original_filename = "orig"
    photo.url = "url"
    photo.thumbnail_path = "thumb_path"
    photo.thumbnail_url = "thumb_url"

    with patch("app.api.endpoints.photo.PhotoService") as MockService:
        mock_service = MockService.return_value
        mock_service.update_photo = AsyncMock(return_value=photo)

        response = await client.patch(f"/api/photos/{photo_id}", json=payload)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["labels"] == {"tag1": "value"}


@pytest.mark.asyncio
async def test_move_photo(client):
    photo_id = "p1"
    payload = {"target_cluster_id": "c2", "order_index": 5}

    with patch("app.api.endpoints.photo.PhotoService") as MockService:
        mock_service = MockService.return_value
        mock_service.move_photo = AsyncMock(return_value=None)

        response = await client.post(f"/api/photos/{photo_id}/move", json=payload)

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_service.move_photo.assert_called_once()


@pytest.mark.asyncio
async def test_delete_photo(client):
    photo_id = "p1"

    with patch("app.api.endpoints.photo.PhotoService") as MockService:
        mock_service = MockService.return_value
        mock_service.delete_photo = AsyncMock(return_value=None)

        response = await client.delete(f"/api/photos/{photo_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_service.delete_photo.assert_called_once()
