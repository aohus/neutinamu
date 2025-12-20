from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.repository.job import JobRepository


@pytest.fixture
def mock_db_session():
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.mark.asyncio
async def test_get_all_by_user_id(mock_db_session):
    repo = JobRepository(mock_db_session)

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [Job(id="j1")]
    mock_db_session.execute.return_value = mock_result

    jobs = await repo.get_all_by_user_id("u1")
    assert len(jobs) == 1


@pytest.mark.asyncio
async def test_get_by_id(mock_db_session):
    repo = JobRepository(mock_db_session)

    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = Job(id="j1")
    mock_db_session.execute.return_value = mock_result

    job = await repo.get_by_id("j1")
    assert job.id == "j1"


@pytest.mark.asyncio
async def test_create_job(mock_db_session):
    repo = JobRepository(mock_db_session)
    job = Job(id="j1")

    await repo.create_job(job)

    mock_db_session.add.assert_called_once_with(job)
    mock_db_session.flush.assert_called_once()
    mock_db_session.refresh.assert_called_once_with(job)


@pytest.mark.asyncio
async def test_delete_job(mock_db_session):
    repo = JobRepository(mock_db_session)
    job = Job(id="j1")

    await repo.delete_job(job)
    mock_db_session.delete.assert_called_once_with(job)
