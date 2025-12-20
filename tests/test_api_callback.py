import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.models.job import ClusterJob, JobStatus, Job
from app.models.photo import Photo

@pytest.mark.asyncio
async def test_handle_clustering_result_success(client):
    payload = {
        "task_id": "task_123",
        "status": "completed",
        "request_id": "req_123",
        "result": {
            "clusters": [
                {
                    "id": 0,
                    "photo_details": [{"path": "storage/path/photo1.jpg"}]
                }
            ]
        }
    }
    
    # Mock DB interaction
    # The endpoint does: 
    # 1. Select ClusterJob
    # 2. Select Photo
    # 3. Delete Cluster
    # 4. Insert Cluster
    # 5. Update Photo
    # 6. Update Job/ClusterJob
    
    # We can mock get_db to return a mock session
    
    mock_session = AsyncMock()
    
    # Mock ClusterJob query result
    mock_cluster_job = MagicMock(spec=ClusterJob)
    mock_cluster_job.job_id = "job_123"
    
    # Mock Photo query result
    mock_photo = MagicMock(spec=Photo)
    mock_photo.storage_path = "storage/path/photo1.jpg"
    
    # Mock Job query result
    mock_job = MagicMock(spec=Job)
    
    # Setup execute results
    # 1st execute: select ClusterJob
    # 2nd execute: select Photo
    # 3rd execute: delete Cluster (returns CursorResult)
    # 4th execute: select Job
    
    # It's hard to mock sequential execute calls with different returns accurately using simple side_effect list
    # because some calls return scalars(), others scalar_one_or_none().
    
    # Let's try to mock the scalar_one_or_none and scalars calls on the result object.
    
    # Result for ClusterJob
    result_cluster_job = MagicMock()
    result_cluster_job.scalar_one_or_none.return_value = mock_cluster_job
    
    # Result for Photo
    result_photos = MagicMock()
    result_photos.scalars.return_value.all.return_value = [mock_photo]
    
    # Result for Job
    result_job = MagicMock()
    result_job.scalar_one.return_value = mock_job
    
    # We can use side_effect on db.execute to return different mocks
    # But checking the query object passed to execute is hard.
    
    # For integration test, it might be better to just patch the create_clusters function 
    # and _mark_job_failed function if we want to test the routing logic.
    # The prompt asked for test code for the api.
    
    with patch("app.api.endpoints.callback.create_clusters", new_callable=AsyncMock) as mock_create:
        with patch("app.api.endpoints.callback.get_db") as mock_get_db:
            mock_get_db.return_value = mock_session
            
            # We still need db.execute for finding ClusterJob
            mock_session.execute.return_value = result_cluster_job
            
            response = await client.post("/api/cluster/callback", json=payload)
            
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            mock_create.assert_called_once()
            
            # Verify we committed (or at least logic flowed there)
            # In the code: await create_clusters ... await db.commit() is NOT called in status="completed" block?
            # Wait, check code:
            # await create_clusters(db, cluster_job, cluster_result)
            # ...
            # create_clusters function calls commit at the end? Yes.
            # "await db.commit()" is at end of create_clusters.
            
@pytest.mark.asyncio
async def test_handle_clustering_result_failure(client):
    payload = {
        "task_id": "task_123",
        "status": "failed",
        "request_id": "req_123",
        "error": "Some error"
    }

    mock_session = AsyncMock()
    mock_cluster_job = MagicMock(spec=ClusterJob)
    mock_cluster_job.job_id = "job_123"
    
    result_cluster_job = MagicMock()
    result_cluster_job.scalar_one_or_none.return_value = mock_cluster_job
    
    with patch("app.api.endpoints.callback._mark_job_failed", new_callable=AsyncMock) as mock_mark_failed:
        with patch("app.api.endpoints.callback.get_db") as mock_get_db:
            mock_get_db.return_value = mock_session
            mock_session.execute.return_value = result_cluster_job
            
            response = await client.post("/api/cluster/callback", json=payload)
            
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            mock_mark_failed.assert_called_once_with(mock_session, "job_123")
            mock_session.commit.assert_called_once()
