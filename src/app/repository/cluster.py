from typing import Optional, Sequence

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
        await self.db.flush()
        await self.db.refresh(cluster)
        return cluster

    async def delete(self, cluster: Cluster):
        await self.db.delete(cluster)

    async def delete_by_id_returning_order_index(self, cluster_id: str) -> Optional[int]:
        result = await self.db.execute(delete(Cluster).where(Cluster.id == cluster_id).returning(Cluster.order_index))
        return result.scalars().first()

    async def flush(self):
        await self.db.flush()
