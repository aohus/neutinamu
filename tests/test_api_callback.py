from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.job import ClusterJob, Job, JobStatus
from app.models.photo import Photo


@pytest.mark.asyncio
async def test_handle_clustering_result_success(client):
    payload = {
        "task_id": "task_123",
        "status": "completed",
        "request_id": "req_123",
        "result": {"clusters": [{"id": 0, "photo_details": [{"path": "storage/path/photo1.jpg"}]}]},
    }

    # Mock DB interaction
    class MockResult:
        def __init__(self, item, is_list=False):
            self._item = item
            self._is_list = is_list

        def scalar_one_or_none(self):
            return self._item

        def scalar_one(self):
            return self._item

        def scalars(self):
            return self

        def all(self):
            return self._item if self._is_list else [self._item]

    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    # Mock ClusterJob
    mock_cluster_job = MagicMock(spec=ClusterJob)
    mock_cluster_job.job_id = "job_123"

    # Mock Photo
    mock_photo = MagicMock(spec=Photo)
    mock_photo.storage_path = "storage/path/photo1.jpg"

    # Mock Job
    mock_job = MagicMock(spec=Job)

    # Setup side_effect for db.execute
    # 1. Select ClusterJob
    # 2. Select Photo
    # 3. Delete Cluster (returns result but we don't use scalars on it usually, but let's provide a dummy)
    # 4. Select Job

    mock_session.execute.side_effect = [
        MockResult(mock_cluster_job),  # ClusterJob query
        MockResult([mock_photo], is_list=True),  # Photo query (scalars().all())
        MagicMock(),  # Delete query result
        MockResult(mock_job),  # Job query
    ]

    from app.db.database import get_db
    from app.main import app

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.api.endpoints.callback.create_clusters", new_callable=AsyncMock) as mock_create:
        # No patch(get_db) here
        response = await client.post("/api/cluster/callback", json=payload)

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_handle_clustering_result_failure(client):
    payload = {"task_id": "task_123", "status": "failed", "request_id": "req_123", "error": "Some error"}

    class MockResult:
        def __init__(self, item):
            self._item = item

        def scalar_one_or_none(self):
            return self._item

    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    mock_cluster_job = MagicMock(spec=ClusterJob)
    mock_cluster_job.job_id = "job_123"

    # Only one query in failure case (Select ClusterJob)
    mock_session.execute.return_value = MockResult(mock_cluster_job)

    from app.db.database import get_db
    from app.main import app

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.api.endpoints.callback._mark_job_failed", new_callable=AsyncMock) as mock_mark_failed:
        # No patch(get_db) here
        response = await client.post("/api/cluster/callback", json=payload)

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_mark_failed.assert_called_once_with(mock_session, "job_123")
        mock_session.commit.assert_called_once()
