from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi import status
from app.models.job import Job, ExportJob, JobStatus, ExportStatus
from datetime import datetime


@pytest.mark.asyncio
async def test_start_cluster_enqueues_celery_task(client, mock_uow):
    job_id = "test_job_id"
    payload = {"min_samples": 3, "max_dist_m": 10.0, "max_alt_diff_m": 20.0}

    # Setup Mock UoW behavior
    mock_job = Job(id=job_id, status=JobStatus.CREATED, title="Test Job")
    mock_uow.jobs.get_by_id = AsyncMock(return_value=mock_job)
    
    # We need to mock the save method to update the job status in place or ignore
    mock_uow.jobs.save = AsyncMock()
    mock_uow.commit = AsyncMock()

    with patch("app.services.job.run_clustering_pipeline_task_celery.delay") as mock_celery_delay:

        response = await client.post(f"/api/jobs/{job_id}/cluster", json=payload)

        assert response.status_code == status.HTTP_202_ACCEPTED
        
        # Verify Celery task was called
        mock_celery_delay.assert_called_once_with(
            job_id,
            payload["min_samples"],
            payload["max_dist_m"],
            payload["max_alt_diff_m"],
        )
        
        data = response.json()
        assert data["job_id"] == job_id
        # Job status should be updated to PENDING in the service
        assert mock_job.status == JobStatus.PENDING 
        assert data["status"] == "PENDING"
        assert data["message"] == "{'message': 'Clustering started'}"


@pytest.mark.asyncio
async def test_start_export_enqueues_celery_task(client, mock_uow):
    job_id = "test_job_id"
    payload = {
        "cover_title": "Test Report",
        "cover_company_name": "Test Co.",
        "labels": {"key1": "value1"},
    }

    # Setup Mock UoW behavior
    mock_job = Job(id=job_id, status=JobStatus.COMPLETED, title="Test Job", company_name="Old Co")
    mock_uow.jobs.get_by_id_with_user = AsyncMock(return_value=mock_job)
    mock_uow.jobs.get_latest_export_job = AsyncMock(return_value=None) # No existing export job
    
    mock_export_job = ExportJob(
        id="new_export_id",
        job_id=job_id,
        status=ExportStatus.PENDING,
        cover_title="Test Report",
        cover_company_name="Test Co.",
        labels={"key1": "value1"},
    )
    mock_uow.jobs.create_export_job = AsyncMock(return_value=mock_export_job)
    mock_uow.commit = AsyncMock()

    with patch("app.services.job.generate_pdf_for_session_celery.delay") as mock_celery_delay:

        response = await client.post(f"/api/jobs/{job_id}/export", json=payload)

        assert response.status_code == status.HTTP_202_ACCEPTED
        
        # Verify Celery task was called
        mock_celery_delay.assert_called_once_with(mock_export_job.id)
        
        data = response.json()
        assert data["status"] == "PENDING"