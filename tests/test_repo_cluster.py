from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cluster import Cluster
from app.repository.cluster import ClusterRepository


@pytest.fixture
def mock_db_session():
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.mark.asyncio
async def test_get_by_id(mock_db_session):
    repo = ClusterRepository(mock_db_session)

    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = Cluster(id="c1")
    mock_db_session.execute.return_value = mock_result

    cluster = await repo.get_by_id("c1")
    assert cluster.id == "c1"


@pytest.mark.asyncio
async def test_get_by_job_id(mock_db_session):
    repo = ClusterRepository(mock_db_session)

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [Cluster(id="c1")]
    mock_db_session.execute.return_value = mock_result

    clusters = await repo.get_by_job_id("j1")
    assert len(clusters) == 1


@pytest.mark.asyncio
async def test_create_cluster(mock_db_session):
    repo = ClusterRepository(mock_db_session)
    cluster = Cluster(id="c1")

    await repo.create(cluster)

    mock_db_session.add.assert_called_once_with(cluster)
    mock_db_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_delete_cluster(mock_db_session):
    repo = ClusterRepository(mock_db_session)
    cluster = Cluster(id="c1")

    await repo.delete(cluster)
    mock_db_session.delete.assert_called_once_with(cluster)
