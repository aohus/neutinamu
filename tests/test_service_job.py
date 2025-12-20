import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import BackgroundTasks
from app.services.job import JobService
from app.models.job import Job, JobStatus
from app.models.user import User

@pytest.fixture
def mock_uow():
    uow = MagicMock()
    uow.jobs = MagicMock()
    uow.commit = AsyncMock()
    uow.jobs.save = AsyncMock()
    uow.jobs.create_job = AsyncMock(side_effect=lambda x: x)
    return uow

@pytest.mark.asyncio
async def test_get_jobs(mock_uow):
    service = JobService(mock_uow)
    user = User(user_id="u1")
    mock_uow.jobs.get_all_by_user_id = AsyncMock(return_value=[Job(id="j1")])
    
    jobs = await service.get_jobs(user)
    assert len(jobs) == 1

@pytest.mark.asyncio
async def test_create_job(mock_uow):
    service = JobService(mock_uow)
    user = User(user_id="u1", company_name="Corp")
    
    job = await service.create_job(user, title="New Job")
    
    assert job.title == "New Job"
    assert job.company_name == "Corp"
    mock_uow.jobs.create_job.assert_called_once()
    mock_uow.commit.assert_called_once()

@pytest.mark.asyncio
async def test_delete_job(mock_uow):
    service = JobService(mock_uow)
    job = Job(id="j1", user_id="u1")
    mock_uow.jobs.get_by_id = AsyncMock(return_value=job)
    mock_uow.jobs.delete_job = AsyncMock()
    
    with patch("app.services.job.get_storage_client") as mock_storage:
        mock_storage_client = mock_storage.return_value
        mock_storage_client.delete_directory = AsyncMock()
        
        await service.delete_job("j1")
        
        mock_uow.jobs.delete_job.assert_called_once()
        mock_storage_client.delete_directory.assert_called_once()

@pytest.mark.asyncio
async def test_get_job(mock_uow):
    service = JobService(mock_uow)
    job = Job(id="j1")
    mock_uow.jobs.get_by_id_with_unfinished_export = AsyncMock(return_value=job)
    
    found = await service.get_job("j1")
    assert found.id == "j1"

@pytest.mark.asyncio
async def test_start_cluster(mock_uow):
    service = JobService(mock_uow)
    job = Job(id="j1", status=JobStatus.CREATED)
    mock_uow.jobs.get_by_id = AsyncMock(return_value=job)
    
    bg_tasks = MagicMock(spec=BackgroundTasks)
    
    with patch("app.services.job.get_storage_client"):
        job, data = await service.start_cluster(
            job_id="j1", background_tasks=bg_tasks
        )
        
        assert job.status == JobStatus.PENDING
        mock_uow.commit.assert_called_once()
        bg_tasks.add_task.assert_called_once()
