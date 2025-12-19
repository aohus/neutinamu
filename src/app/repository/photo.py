from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.photo import Photo


class PhotoRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, photo_id: str) -> Optional[Photo]:
        result = await self.db.execute(select(Photo).where(Photo.id == photo_id))
        return result.scalars().first()

    async def get_by_job_id(self, job_id: str) -> Sequence[Photo]:
        result = await self.db.execute(select(Photo).where(Photo.job_id == job_id))
        return result.scalars().all()

    async def get_by_ids(self, photo_ids: List[str]) -> Sequence[Photo]:
        result = await self.db.execute(select(Photo).where(Photo.id.in_(photo_ids)).order_by(Photo.order_index.asc()))
        return result.scalars().all()

    async def get_by_cluster_id(self, cluster_id: str) -> Sequence[Photo]:
        result = await self.db.execute(select(Photo).where(Photo.cluster_id == cluster_id))
        return result.scalars().all()

    async def get_by_cluster_id_ordered(self, cluster_id: str) -> List[Photo]:
        result = await self.db.execute(
            select(Photo)
            .where(Photo.cluster_id == cluster_id, Photo.deleted_at.is_(None))
            .order_by(Photo.order_index.asc())
        )
        return list(result.scalars().all())

    async def get_active_by_cluster_id(self, cluster_id: str) -> Sequence[Photo]:
        result = await self.db.execute(select(Photo).where(Photo.cluster_id == cluster_id, Photo.deleted_at.is_(None)))
        return result.scalars().all()

    async def save(self, photo: Photo) -> Photo:
        self.db.add(photo)
        await self.db.flush()
        await self.db.refresh(photo)
        return photo

    async def add_all(self, photos: Sequence[Photo]):
        self.db.add_all(photos)

    async def unassign_cluster(self, cluster_id: str):
        await self.db.execute(
            update(Photo).where(Photo.cluster_id == cluster_id).values(cluster_id=None, deleted_at=datetime.now())
        )

    async def flush(self):
        await self.db.flush()
