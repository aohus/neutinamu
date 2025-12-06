import asyncio
import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.domain.cluster_background import run_pipeline_task
from app.domain.cluster_client import run_deep_cluster_for_job
from app.domain.generate_pdf import generate_pdf_for_session
from app.domain.storage.factory import (
    get_storage_client,  # Import storage client factory
)
from app.models.cluster import Cluster
from app.models.job import ExportJob, ExportStatus, Job, JobStatus
from app.models.photo import Photo
from app.models.user import User
from app.schemas.photo import (
    BatchPresignedUrlResponse,
    PhotoUploadRequest,
    PresignedUrlResponse,
)
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class AsyncBytesIO(io.BytesIO):
    async def read(self, *args, **kwargs):
        return super().read(*args, **kwargs)


def generate_thumbnail(image_data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            # Handle orientation if needed, but basic thumbnail is fine
            img.thumbnail((600, 400)) 
            if img.mode != 'RGB':
                img = img.convert('RGB')
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=85)
            return output.getvalue()
    except Exception as e:
        logger.warning(f"Failed to generate thumbnail: {e}")
        return None


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_jobs(self, user: User):
        logger.info(f"Fetching jobs for user: {user.user_id}")
        result = await self.db.execute(select(Job)
                                       .where(Job.user_id == user.user_id)
                                       .options(selectinload(Job.export_job)))
        jobs = result.scalars().all()
        logger.info(f"Found {len(jobs)} jobs for user: {user.user_id}")
        return jobs

    async def create_job(self, user: User, title: str, contractor_name: Optional[str] = None, work_date: Optional[datetime] = None):
        logger.info(f"User {user.user_id} creating job with title: '{title}'")
        job = Job(user_id=user.user_id, title=title, contractor_name=contractor_name, work_date=work_date)
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        logger.info(f"Job created successfully with ID: {job.id}")
        return job

    async def delete_job(self, job_id: str):
        logger.info(f"Deleting job: '{job_id}'")
        storage = get_storage_client()
        
        job = await self.db.get(Job, job_id)
        user_id = job.user_id
        if job is not None:
            await self.db.delete(job)
            await self.db.commit()

        job_object_path = f"{user_id}/{job_id}/"
        await storage.delete_directory(job_object_path)
        
        logger.info(f"Job deleted successfully with ID: {job_id}")
        return 

    async def get_job(self, job_id: str):
        logger.debug(f"Fetching job with ID: {job_id}")
        result = await self.db.execute(select(Job)
                                       .where(Job.id == job_id)            
                                       .options(selectinload(Job.export_job.and_(ExportJob.finished_at.is_(None)))))
        job = result.scalars().first()
        if not job:
            logger.warning(f"Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")
        
        logger.debug(f"Job {job_id} found with status {job.status}")
        return job

    async def generate_presigned_urls(
        self, job_id: str, files: list[PhotoUploadRequest]
    ) -> BatchPresignedUrlResponse:
        """
        Generate pre-signed URLs for direct upload.
        If storage is local, returns None for URLs, indicating proxy upload should be used.
        """
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()
        response_urls = []
        
        strategy = "proxy" if settings.STORAGE_TYPE == "local" else "direct"

        for file_req in files:
            target_path = f"{job.user_id}/{job.id}/photos/original/{file_req.filename}"
            upload_url = storage.generate_upload_url(target_path, content_type=file_req.content_type)
            response_urls.append(
                PresignedUrlResponse(
                    filename=file_req.filename,
                    upload_url=upload_url,
                    storage_path=target_path
                )
            )

        return BatchPresignedUrlResponse(strategy=strategy, urls=response_urls)

    async def process_uploaded_files(
        self, job_id: str, file_info_list: list[dict]
    ) -> list[Photo]:
        """
        Register files that have been uploaded directly to storage.
        file_info_list: list of dicts with 'filename' and 'storage_path'
        """
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()
        photos = []
        for info in file_info_list:
            storage_path = info['storage_path'] # This is the original image path

            original_filename = os.path.basename(storage_path)
            original_filename_parts = os.path.splitext(original_filename)
            thumb_filename = f"{original_filename_parts[0]}_thumb.jpg"
            
            base_dir = Path(storage_path).parent.parent  # photos/
            derived_thumbnail_path = str(base_dir / "thumbnail" / thumb_filename)

            thumbnail_path = info.get('thumbnail_path', derived_thumbnail_path)
            
            if thumbnail_path == storage_path:
                thumbnail_path = derived_thumbnail_path
            
            photo = Photo(
                job_id=job_id,
                original_filename=info['filename'],
                storage_path=storage_path,
                thumbnail_path=thumbnail_path,
                url=storage.get_url(storage_path),
                thumbnail_url=storage.get_url(thumbnail_path) if thumbnail_path else None
            )
            photos.append(photo)
        
        self.db.add_all(photos)
        await self.db.commit()
        
        # Trigger metadata extraction async task here if needed
        
        logger.info(f"Registered {len(photos)} uploaded photos for job {job_id}")
        return photos

    async def upload_photos(
        self,
        job_id: str,
        files: list[UploadFile] = File(...),
    ):
        logger.info(f"Uploading {len(files)} files for job ID: {job_id}")
        # Check if job exists
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            logger.error(f"Upload failed: Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        # config = JobConfig(job_id=job.id) # JobConfig is for clustering params, not storage instantiation
        storage = get_storage_client() # Use the factory to get the correct storage service

        async def process_file(file: UploadFile):
            logger.debug(f"Processing file: {file.filename}")

            # Determine the storage path based on storage type
            target_path = f"{job.user_id}/{job.id}/photos/original/{file.filename}"
            content = await file.read()
            
            # Wrap content in AsyncBytesIO because storage.save_file expects async read
            saved_path = await storage.save_file(AsyncBytesIO(content), target_path, file.content_type)

            # Generate and save thumbnail
            thumb_content = generate_thumbnail(content)
            thumbnail_path = None
            if thumb_content:
                # Thumbnail path: user_id/job_id/photos/thumbnail/filename_thumb.jpg
                original_filename_parts = os.path.splitext(os.path.basename(file.filename))
                thumb_filename = f"{original_filename_parts[0]}_thumb.jpg"
                thumb_target_path = f"{job.user_id}/{job.id}/photos/thumbnail/{thumb_filename}"
                thumbnail_path = await storage.save_file(AsyncBytesIO(thumb_content), thumb_target_path, "image/jpeg")

            photo = Photo(
                job_id=job_id,
                original_filename=file.filename,
                storage_path=saved_path, 
                thumbnail_path=thumbnail_path, 
                url=storage.get_url(saved_path),
                thumbnail_url=storage.get_url(thumbnail_path) if thumbnail_path else None
            )
            logger.debug(f"Saved {file.filename} to {saved_path}, thumbnail: {thumbnail_path}")
            return photo

        # Process files concurrently
        photos = await asyncio.gather(*[process_file(file) for file in files])

        self.db.add_all(photos)
        await self.db.commit()

        logger.info(
            f"Successfully uploaded and saved {len(photos)} photos for job {job_id}."
        )
        return photos

    async def start_cluster(self, 
                            job_id: str, 
                            background_tasks: BackgroundTasks, 
                            min_samples: int = 3, 
                            max_dist_m: float = 10.0, 
                            max_alt_diff_m: float = 20.0):
        """Trigger the clustering pipeline."""
        logger.info(f"Received request to start clustering for job {job_id}")
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            logger.error(f"Clustering trigger failed: Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status == JobStatus.PROCESSING:
            raise HTTPException(status_code=404, detail="Job is now PROCESSING")

        # Update status
        job.status = JobStatus.PENDING
        await self.db.commit()
        logger.info(f"Job {job_id} status updated to PENDING.")

        # Trigger pipeline in background
        background_tasks.add_task(run_pipeline_task, job_id, min_samples, max_dist_m, max_alt_diff_m)
        logger.info(f"Clustering task for job {job_id} added to background tasks.")
        
        data = {"message": "Local clustering started"}
        return job, data
 
    async def start_cluster_server(self,
                                    job_id: str, 
                                    background_tasks: BackgroundTasks, 
                                    min_samples: int = 3, 
                                    max_dist_m: float = 10.0, 
                                    max_alt_diff_m: float = 20.0):
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # If retrying, clear previous results
        # if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        #     logger.info(f"Clearing previous clusters for job {job_id}")
        #     cluster_ids_subq = (
        #         select(Cluster.id)
        #         .where(Cluster.job_id == job_id)
        #         .subquery()
        #     )

        #     # Unassign photos
        #     await self.db.execute(
        #         update(Photo)
        #         .where(Photo.cluster_id.in_(select(cluster_ids_subq.c.id)))
        #         .values(cluster_id=None)
        #     )

        #     # Delete clusters
        #     await self.db.execute(
        #         delete(Cluster).where(Cluster.id.in_(select(cluster_ids_subq.c.id)))
        #     )

        # 상태를 PENDING -> RUNNING(soon) 으로 바꾸기 전에 표시만
        job.status = JobStatus.PENDING
        await self.db.commit()

        logger.info(f"Request deep_cluster logic start ")
        # Pass parameters to the deep cluster runner
        data = await run_deep_cluster_for_job(
            job_id, 
            min_samples=min_samples, 
            max_dist_m=max_dist_m, 
            max_alt_diff_m=max_alt_diff_m
        )
        return job, data

    async def start_export(self, job_id: str, background_tasks: BackgroundTasks):
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        result = await self.db.execute(
            select(ExportJob)
            .where(ExportJob.job_id == job_id)
            .order_by(ExportJob.created_at.desc())
        )
        
        export_job = result.scalars().first()
        if export_job and export_job.status in {
            ExportStatus.PENDING,
            ExportStatus.PROCESSING,
        }:
            raise HTTPException(status_code=400, detail="Export already in progress")

        export_job = ExportJob(job_id=job.id, status=ExportStatus.PENDING)
        self.db.add(export_job)
        await self.db.commit()
        await self.db.refresh(export_job)
        background_tasks.add_task(generate_pdf_for_session, export_job.id)
        return export_job
    
    async def get_export_job(self, job_id):
        result = await self.db.execute(
            select(ExportJob)
            .where(ExportJob.job_id == job_id)
            .order_by(ExportJob.created_at.desc())
        )
        export_job = result.scalars().first()
        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")
        return export_job.status, export_job.pdf_path, export_job.error_message

    async def download_export_pdf(self, job_id):
        logger.debug(f"Fetching job with ID: {job_id}")
        result = await self.db.execute(select(Job)
                                       .where(Job.id == job_id)
                                       .options(selectinload(Job.export_job)))
        job = result.scalars().first()
        if not job:
            logger.warning(f"Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")
        
        filename = f"{job.title}.pdf"
        export_job = job.export_job
        if not export_job or export_job.status != ExportStatus.EXPORTED or not export_job.pdf_path:
            raise HTTPException(status_code=404, detail="No finished export for this session")

        target_path = export_job.pdf_path
        if not target_path:
            raise HTTPException(status_code=404, detail="PDF file not found")
        return target_path, filename
