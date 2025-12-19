from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.user import UserRepository
from app.repository.job import JobRepository
from app.repository.cluster import ClusterRepository
from app.repository.photo import PhotoRepository


class UnitOfWork:
    """
    Unit of Work pattern to manage repositories and database transactions.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.users = UserRepository(db)
        self.jobs = JobRepository(db)
        self.clusters = ClusterRepository(db)
        self.photos = PhotoRepository(db)

    async def commit(self):
        await self.db.commit()

    async def rollback(self):
        await self.db.rollback()

    async def flush(self):
        await self.db.flush()

    async def refresh(self, instance):
        await self.db.refresh(instance)
