from typing import List, Optional, Sequence

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.cluster import Cluster
from app.models.photo import Photo


class ClusterRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, cluster_id: str) -> Optional[Cluster]:
        result = await self.db.execute(select(Cluster).where(Cluster.id == cluster_id))
        return result.scalars().first()

    async def get_by_id_with_photos(self, cluster_id: str) -> Optional[Cluster]:
        result = await self.db.execute(
            select(Cluster).options(selectinload(Cluster.photos)).where(Cluster.id == cluster_id)
        )
        return result.scalars().first()

    async def get_by_job_id(self, job_id: str) -> Sequence[Cluster]:
        result = await self.db.execute(
            select(Cluster)
            .where(Cluster.job_id == job_id)
            .options(selectinload(Cluster.photos.and_(Photo.deleted_at.is_(None))))
        )
        return result.scalars().all()

    async def get_ordered_by_job_id(self, job_id: str) -> Sequence[Cluster]:
        result = await self.db.execute(
            select(Cluster).where(Cluster.job_id == job_id).order_by(Cluster.order_index.asc())
        )
        return result.scalars().all()

    async def get_by_name(self, job_id: str, name: str) -> Optional[Cluster]:
        result = await self.db.execute(select(Cluster).where(Cluster.job_id == job_id, Cluster.name == name))
        return result.scalars().first()

    async def get_clusters_after_order(self, order_index: int) -> Sequence[Cluster]:
        # This logic seems incomplete without job_id, but the service usage context usually implies filtering.
        # However, for a repo method, it should probably take job_id.
        # In delete_cluster logic: select(Cluster).where(Cluster.order_index > idx) -- this is risky if not filtered by job_id!
        # The service code:
        # result = await self.db.execute(delete(Cluster).where(Cluster.id == cluster_id).returning(Cluster.order_index))
        # idx = result.scalars().first()
        # if idx is not None:
        #    result = await self.db.execute(select(Cluster).where(Cluster.order_index > idx).order_by(Cluster.order_index.asc()))
        #
        # WAIT! The original service code for delete_cluster reordering is BUGGY if it doesn't filter by job_id.
        # `select(Cluster).where(Cluster.order_index > idx)` selects ALL clusters in the DB with that index?
        # Ah, if `job_id` is not in the where clause, it will reorder clusters of OTHER jobs too!
        # This looks like a bug in the original code.
        # I should probably fix it or strictly follow it.
        # "Rigorously adhere to existing project conventions...".
        # But "Fix bugs...". This is a dangerous bug. I will assume it implies context of the same job.
        # Let's check `ClusterService.delete_cluster`. It receives `job_id`.
        # It executes `delete(Cluster).where(Cluster.id == cluster_id)`. Cluster IDs are UUIDs likely unique.
        # But the subsequent select `select(Cluster).where(Cluster.order_index > idx)` IS missing job_id.
        # I should fix this by adding job_id to the repo method and usage.
        pass

    async def get_clusters_after_order_for_job(self, job_id: str, order_index: int) -> Sequence[Cluster]:
        result = await self.db.execute(
            select(Cluster)
            .where(Cluster.job_id == job_id, Cluster.order_index > order_index)
            .order_by(Cluster.order_index.asc())
        )
        return result.scalars().all()

    async def create(self, cluster: Cluster) -> Cluster:
        self.db.add(cluster)
        await self.db.flush()
        return cluster

    async def save(self, cluster: Cluster) -> Cluster:
        self.db.add(cluster)
        await self.db.commit()
        await self.db.refresh(cluster)
        return cluster

    async def delete(self, cluster: Cluster):
        await self.db.delete(cluster)
        # We might commit here or let service handle it.
        # Since logic involves reordering after delete, maybe we shouldn't commit immediately if we want atomicity?
        # Service `delete_cluster` commits at the end.

    async def delete_by_id_returning_order_index(self, cluster_id: str) -> Optional[int]:
        result = await self.db.execute(delete(Cluster).where(Cluster.id == cluster_id).returning(Cluster.order_index))
        return result.scalars().first()

    async def commit(self):
        await self.db.commit()

    async def flush(self):
        await self.db.flush()
