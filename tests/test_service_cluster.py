from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.models.cluster import Cluster
from app.models.job import Job
from app.models.photo import Photo
from app.services.cluster import ClusterService


@pytest.fixture
def mock_uow():
    uow = MagicMock()
    uow.jobs = MagicMock()
    uow.clusters = MagicMock()
    uow.photos = MagicMock()
    uow.commit = AsyncMock()
    uow.refresh = AsyncMock()
    return uow


@pytest.mark.asyncio
async def test_list_clusters_success(mock_uow):
    service = ClusterService(mock_uow)
    job_id = "j1"

    job = Job(id=job_id, status="CREATED")
    mock_uow.jobs.get_by_id = AsyncMock(return_value=job)
    mock_uow.clusters.get_by_job_id = AsyncMock(return_value=[Cluster(id="c1")])

    clusters = await service.list_clusters(job_id)
    assert len(clusters) == 1
    assert clusters[0].id == "c1"


@pytest.mark.asyncio
async def test_create_cluster(mock_uow):
    service = ClusterService(mock_uow)
    job_id = "j1"

    job = Job(id=job_id, title="Job Title")
    mock_uow.jobs.get_by_id = AsyncMock(return_value=job)
    mock_uow.clusters.get_ordered_by_job_id = AsyncMock(return_value=[])
    mock_uow.clusters.create = AsyncMock()

    cluster, photos = await service.create_cluster(job_id, order_index=0)

    assert cluster.job_id == job_id
    assert cluster.name == "Job Title"
    mock_uow.clusters.create.assert_called_once()
    mock_uow.commit.assert_called_once()


@pytest.mark.asyncio
async def test_update_cluster(mock_uow):
    service = ClusterService(mock_uow)
    cluster_id = "c1"

    cluster = Cluster(id=cluster_id, name="Old Name", order_index=0)
    mock_uow.clusters.get_by_id_with_photos = AsyncMock(return_value=cluster)
    mock_uow.clusters.save = AsyncMock()

    updated = await service.update_cluster(job_id="j1", cluster_id=cluster_id, new_name="New Name")

    assert updated.name == "New Name"
    mock_uow.clusters.save.assert_called_once()
    mock_uow.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_cluster(mock_uow):
    service = ClusterService(mock_uow)
    cluster_id = "c1"

    mock_uow.photos.unassign_cluster = AsyncMock()
    mock_uow.clusters.delete_by_id_returning_order_index = AsyncMock(return_value=1)
    mock_uow.clusters.get_clusters_after_order_for_job = AsyncMock(return_value=[])

    await service.delete_cluster(job_id="j1", cluster_id=cluster_id)

    mock_uow.photos.unassign_cluster.assert_called_once_with(cluster_id)
    mock_uow.commit.assert_called_once()


@pytest.mark.asyncio
async def test_add_photos(mock_uow):
    service = ClusterService(mock_uow)
    cluster_id = "c1"

    mock_uow.clusters.get_by_id = AsyncMock(return_value=Cluster(id=cluster_id))
    mock_uow.photos.get_by_cluster_id = AsyncMock(return_value=[])
    p1 = Photo(id="p1")
    mock_uow.photos.get_by_ids = AsyncMock(return_value=[p1])

    await service.add_photos(job_id="j1", cluster_id=cluster_id, photo_ids=["p1"])

    assert p1.cluster_id == cluster_id
    mock_uow.commit.assert_called_once()
