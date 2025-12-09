import pytest
from unittest.mock import AsyncMock, patch
from app.models.job import Job, JobStatus
from app.schemas.enum import ExportStatus

@pytest.mark.asyncio
async def test_get_jobs_empty(client, mock_user):
    """Test getting jobs when list is empty."""
    with patch("app.api.endpoints.job.JobService") as MockService:
        mock_service_instance = MockService.return_value
        mock_service_instance.get_jobs = AsyncMock(return_value=[])
        
        # We also need to mock authentication to return our test user
        with patch("app.api.endpoints.job.get_current_user", return_value=mock_user):
            response = await client.get("/api/jobs")
            
            assert response.status_code == 200
            assert response.json() == []

@pytest.mark.asyncio
async def test_create_job(client, mock_user):
    """Test creating a new job."""
    payload = {
        "title": "Test Job",
        "construction_type": "Building",
        "company_name": "Test Corp"
    }
    
    expected_job = Job(
        id="job_123",
        title="Test Job",
        status=JobStatus.CREATED,
        construction_type="Building",
        company_name="Test Corp",
        user_id=mock_user.user_id
    )
    
    with patch("app.api.endpoints.job.JobService") as MockService:
        mock_service_instance = MockService.return_value
        mock_service_instance.create_job = AsyncMock(return_value=expected_job)
        
        with patch("app.api.endpoints.job.get_current_user", return_value=mock_user):
            response = await client.post("/api/jobs", json=payload)
            
            assert response.status_code == 201
            data = response.json()
            assert data["title"] == "Test Job"
            assert data["id"] == "job_123"
            assert data["status"] == JobStatus.CREATED
