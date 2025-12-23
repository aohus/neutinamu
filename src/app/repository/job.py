from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload, with_loader_criteria

from app.models.cluster import Cluster
from app.models.job import ExportJob, ExportStatus, Job, JobStatus
from app.models.photo import Photo


class JobRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_by_user_id(self, user_id: str) -> List[Job]:
        result = await self.db.execute(
            select(Job).where(Job.user_id == user_id).options(selectinload(Job.export_job))
        )
        return result.scalars().all()

    async def get_by_id(self, job_id: str, load_export_job: bool = False, load_user: bool = False) -> Optional[Job]:
        query = select(Job).where(Job.id == job_id)
        
        if load_export_job:
            # Only load unfinished export jobs based on original service logic in get_job?
            # Original get_job: options(selectinload(Job.export_job.and_(ExportJob.finished_at.is_(None))))
            # But start_export uses: options(selectinload(Job.user))
            # Let's handle specific loading in specific methods or arguments if simple.
            pass
            
        # For general purpose get, we might not want specific filtering on relations unless specified.
        # I will create specific methods for specific business needs to match strict refactoring.
        
        result = await self.db.execute(query)
        return result.scalars().first()

    async def get_by_id_with_unfinished_export(self, job_id: str) -> Optional[Job]:
        result = await self.db.execute(
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.export_job.and_(ExportJob.finished_at.is_(None))))
        )
        return result.scalars().first()
        
    async def get_by_id_with_user(self, job_id: str) -> Optional[Job]:
        result = await self.db.execute(select(Job).where(Job.id == job_id).options(selectinload(Job.user)))
        return result.scalars().first()

    async def get_by_id_with_export_job(self, job_id: str) -> Optional[Job]:
        result = await self.db.execute(select(Job).where(Job.id == job_id).options(selectinload(Job.export_job)))
        return result.scalars().first()

    async def get_job_details(self, job_id: str) -> Optional[Job]:
        query = (
            select(Job)
            .where(Job.id == job_id)
            .options(
                selectinload(Job.photos),
                selectinload(Job.clusters).selectinload(Cluster.photos),
                selectinload(Job.export_job),
                with_loader_criteria(Photo, Photo.deleted_at.is_(None)),
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create_job(self, job: Job) -> Job:
        self.db.add(job)
        await self.db.flush()
        await self.db.refresh(job)
        return job

    async def delete_job(self, job: Job):
        await self.db.delete(job)

    async def update_job_status(self, job_id: str, status: JobStatus, updated_at: Optional[datetime] = None):
        values = {"status": status}
        if updated_at:
            values["updated_at"] = updated_at
            
        # Using update statement as in original service for optimization or object update
        # Original set_job_uploading used object update. 
        # Original process_uploaded_files used update stmt.
        # Let's stick to object update if we have the object, or stmt if we don't.
        # Here we can be flexible.
        pass

    async def save(self, obj):
        self.db.add(obj)
        await self.db.flush()
        await self.db.refresh(obj)
        return obj

    async def add_all(self, objects: list):
        self.db.add_all(objects)

    async def update_status_by_id(self, job_id: str, status: JobStatus, timestamp_filter: Optional[datetime] = None):
        # Specific logic from process_uploaded_files
        stmt = (
            update(Job)
            .where(Job.id == job_id)
        )
        if timestamp_filter:
            stmt = stmt.where(Job.updated_at == timestamp_filter)
        
        stmt = stmt.values(status=status)
        await self.db.execute(stmt)

    async def get_latest_export_job(self, job_id: str) -> Optional[ExportJob]:
        result = await self.db.execute(
            select(ExportJob).where(ExportJob.job_id == job_id).order_by(desc(ExportJob.created_at))
        )
        return result.scalars().first()
    
    async def create_export_job(self, export_job: ExportJob) -> ExportJob:
        self.db.add(export_job)
        await self.db.flush()
        await self.db.refresh(export_job)
        return export_job
