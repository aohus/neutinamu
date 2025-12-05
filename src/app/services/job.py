import asyncio
import io
import logging
import os

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
                                       .options(selectinload(Job.export_job.and_(ExportJob.finished_at.is_(None)))))
        jobs = result.scalars().all()
        logger.info(f"Found {len(jobs)} jobs for user: {user.user_id}")
        return jobs

    async def create_job(self, user: User, title: str):
        logger.info(f"User {user.user_id} creating job with title: '{title}'")
        job = Job(user_id=user.user_id, title=title)
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        logger.info(f"Job created successfully with ID: {job.id}")
        return job

    async def delete_job(self, job_id: str):
        logger.info(f"Deleting job: '{job_id}'")

        job = await self.db.get(Job, job_id)
        if job is not None:
            await self.db.delete(job)
            await self.db.commit()
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
            # Define path logic consistent with upload_photos
            if settings.STORAGE_TYPE == "local":
                target_path = f"{job.id}/{file_req.filename}"
            else:
                target_path = f"{job.user_id}/{job.id}/{file_req.filename}"

            upload_url = storage.generate_upload_url(target_path, content_type=file_req.content_type)
            
            # If storage service returns None (e.g. Local), we fallback to proxy strategy implicitly
            # But here we set strategy based on config.
            
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
            storage_path = info['storage_path']
            # Placeholder for thumbnail path if client uploaded it or we generate it later
            thumbnail_path = info.get('thumbnail_path', storage_path) 
            
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
            if settings.STORAGE_TYPE == "local":
                # Local storage path: job_id/filename
                target_path = f"{job.id}/{file.filename}"
            else:
                # GCS/S3 storage path: user_id/job_id/filename
                target_path = f"{job.user_id}/{job.id}/{file.filename}"

            # Read file content to memory
            content = await file.read()
            
            # Save original file
            # Wrap content in AsyncBytesIO because storage.save_file expects async read
            saved_path = await storage.save_file(AsyncBytesIO(content), target_path, file.content_type)
            
            # Generate and save thumbnail
            thumb_content = generate_thumbnail(content)
            thumbnail_path = None
            if thumb_content:
                # Insert _thumb before extension
                path_parts = os.path.splitext(target_path)
                thumb_target_path = f"{path_parts[0]}_thumb.jpg" # Force jpg for thumbnail
                
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

        if job.export_job and job.export_job.status in {
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
        result = await self.db.execute(select(ExportJob).where(ExportJob.job_id == job_id))
        export_job = result.scalars().first()
        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")
        return export_job.status, export_job.pdf_path, export_job.error_message


# ============================================
# PDF 생성 관련
# ============================================
# async def generate_pdf_for_session(export_job_id: str):
#     """
#     BackgroundTasks에서 호출되는 함수.
#     세션의 클러스터/사진 정보를 읽어서 PDF를 생성한다.
#     여기서는 간단하게 ReportLab으로 텍스트+파일명만 넣는 예시.
#     """
#     from reportlab.lib.pagesizes import A4
#     from reportlab.pdfgen import canvas

#     async with AsyncSessionLocal() as session:
#         try:
#             result = await session.execute(select(ExportJob).where(ExportJob.id == export_job_id))
#             export_job = result.scalars().first()
#             if not export_job:
#                 return

#             export_job.status = Status.RUNNING
#             await session.commit()

#             job_id = export_job.job_id
#             if not job_id:
#                 export_job.status = Status.FAILED
#                 export_job.error_message = "Job not found"
#                 await session.commit()
#                 return

#             pdf_path = Path("/app/assets") / f"job_{export_job.id}_{int(datetime.now().timestamp())}.pdf"
#             c = canvas.Canvas(str(pdf_path), pagesize=A4)
#             width, height = A4

#             # 클러스터 순서대로 페이지 생성
#             clusters = (
#                 session.query(Cluster)
#                 .filter(Cluster.session_id == session.id)
#                 .order_by(Cluster.order_index.asc())
#                 .all()
#             )

#             for cluster in clusters:
#                 c.setFont("Helvetica-Bold", 16)
#                 c.drawString(50, height - 50, f"장소: {cluster.name or cluster.id}")
#                 c.setFont("Helvetica", 12)
#                 y = height - 80

#                 pcs = (
#                     session.query(Photo)
#                     .filter(Photo.cluster_id == cluster.id)
#                     .order_by(Photo.order_index.asc())
#                     .all()
#                 )

#                 for pc in pcs:
#                     text = f"- 사진 #{pc.photo_id} : {pc.photo.original_filename}"
#                     c.drawString(60, y, text)
#                     y -= 20
#                     if y < 80:
#                         c.showPage()
#                         c.setFont("Helvetica-Bold", 16)
#                         c.drawString(50, height - 50, f"장소: {cluster.name or cluster.id} (계속)")
#                         c.setFont("Helvetica", 12)
#                         y = height - 80
#                 c.showPage()
#             c.save()

#             export_job.status = Status.DONE
#             export_job.pdf_path = str(pdf_path)
#             export_job.finished_at = datetime.now()
#             export_job.status = Status.EXPORTED
#             await session.commit()
#         except Exception as e:
#             export_job = await session.query(ExportJob).filter(ExportJob.id == job_id).first()
#             if export_job:
#                 export_job.status = Status.FAILED
#                 export_job.error_message = str(e)
#                 export_job.finished_at = datetime.now()
#                 await session.commit()
#         finally:
#             await session.close()
