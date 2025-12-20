import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import status
from app.models.cluster import Cluster
from app.models.photo import Photo

@pytest.mark.asyncio
async def test_list_clusters(client):
    job_id = "job_123"
    
    # Create mock Cluster objects
    cluster1 = MagicMock(spec=Cluster)
    cluster1.id = "c1"
    cluster1.job_id = job_id
    cluster1.name = "Cluster 1"
    cluster1.order_index = 0
    cluster1.photos = []
    
    with patch("app.api.endpoints.cluster.ClusterService") as MockService:
        mock_service = MockService.return_value
        mock_service.list_clusters = AsyncMock(return_value=[cluster1])
        
        response = await client.get(f"/api/jobs/{job_id}/clusters")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "c1"

@pytest.mark.asyncio
async def test_sync_clusters(client):
    job_id = "job_123"
    payload = {
        "clusters": [
            {
                "id": "c1",
                "name": "Cluster 1", 
                "order_index": 0,
                "photos": []
            }
        ]
    }
    
    with patch("app.api.endpoints.cluster.ClusterService") as MockService:
        mock_service = MockService.return_value
        mock_service.sync_clusters = AsyncMock(return_value=None)
        
        response = await client.put(f"/api/jobs/{job_id}/clusters/sync", json=payload)
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_service.sync_clusters.assert_called_once()

@pytest.mark.asyncio
async def test_create_cluster(client):
    job_id = "job_123"
    payload = {
        "name": "New Cluster",
        "order_index": 1,
        "photo_ids": []
    }
    
    # Mock return values
    mock_cluster = MagicMock(spec=Cluster)
    mock_cluster.id = "new_c"
    mock_cluster.job_id = job_id
    mock_cluster.name = "New Cluster"
    mock_cluster.order_index = 1
    mock_cluster.photos = []
    
    with patch("app.api.endpoints.cluster.ClusterService") as MockService:
        mock_service = MockService.return_value
        mock_service.create_cluster = AsyncMock(return_value=(mock_cluster, []))
        
        response = await client.post(f"/api/jobs/{job_id}/clusters", json=payload)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == "new_c"
        assert data["name"] == "New Cluster"

@pytest.mark.asyncio
async def test_add_photos_to_cluster(client):
    cluster_id = "c1"
    payload = {"photo_ids": ["p1", "p2"]}
    
    with patch("app.api.endpoints.cluster.ClusterService") as MockService:
        mock_service = MockService.return_value
        mock_service.add_photos = AsyncMock(return_value=None)
        
        response = await client.post(f"/api/clusters/{cluster_id}/add_photos", json=payload)
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_service.add_photos.assert_called_once_with(
            job_id="", cluster_id=cluster_id, photo_ids=["p1", "p2"]
        )

@pytest.mark.asyncio
async def test_update_cluster(client):
    cluster_id = "c1"
    payload = {"new_name": "Updated Name", "order_index": 2}
    
    mock_cluster = MagicMock(spec=Cluster)
    mock_cluster.id = cluster_id
    mock_cluster.job_id = "job_123"
    mock_cluster.name = "Updated Name"
    mock_cluster.order_index = 2
    mock_cluster.photos = []

    with patch("app.api.endpoints.cluster.ClusterService") as MockService:
        mock_service = MockService.return_value
        mock_service.update_cluster = AsyncMock(return_value=mock_cluster)
        
        response = await client.patch(f"/api/clusters/{cluster_id}", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["order_index"] == 2

@pytest.mark.asyncio
async def test_delete_cluster(client):
    cluster_id = "c1"
    
    with patch("app.api.endpoints.cluster.ClusterService") as MockService:
        mock_service = MockService.return_value
        mock_service.delete_cluster = AsyncMock(return_value=None)
        
        response = await client.delete(f"/api/clusters/{cluster_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_service.delete_cluster.assert_called_once()
