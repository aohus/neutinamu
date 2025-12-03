import asyncio
import logging
from typing import List

from app.core.config import JobConfig
from app.db.database import AsyncSessionLocal
from app.domain.cluster_background import run_pipeline_task
from app.domain.cluster_client import run_deep_cluster_for_job
from app.domain.generate_pdf import generate_pdf_for_session
from app.domain.pipeline import PhotoClusteringPipeline
from app.domain.storage.local import LocalStorageService as StorageService
from app.models.cluster import Cluster
from app.models.job import ExportJob, ExportStatus, Job, JobStatus
from app.models.photo import Photo
from app.models.user import User
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_jobs(self, user: User):
        logger.info(f"Fetching jobs for user: {user.user_id}")
        result = await self.db.execute(select(Job)
                                       .where(Job.user_id == user.user_id)
                                       .options(selectinload(Job.export_job.and_(ExportJob.finished_at.is_(None)))))
        jobs = result.scalars().all()
        
        if not jobs:
            logger.warning(f"No jobs found for user: {user.user_id}")
            raise HTTPException(status_code=404, detail="Job not found")
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

    async def upload_photos(
        self,
        job_id: str,
        files: List[UploadFile] = File(...),
    ):
        logger.info(f"Uploading {len(files)} files for job ID: {job_id}")
        # Check if job exists
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            logger.error(f"Upload failed: Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        config = JobConfig(job_id=job.id)
        storage = StorageService(config)

        async def process_file(file: UploadFile):
            logger.debug(f"Processing file: {file.filename}")
            file_path = await storage.save_file(file)
            photo = Photo(
                job_id=job_id,
                original_filename=file.filename,
                storage_path=str(file_path.relative_to(config.MEDIA_ROOT)),
                thumbnail_path=str(file_path.relative_to(config.MEDIA_ROOT)),
            )
            logger.debug(f"Saved {file.filename} to {file_path}")
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
        
        if job.status == JobStatus.COMPLETED:
            cluster_ids_subq = (
                select(Cluster.id)
                .where(Cluster.job_id == job_id)
                .subquery()
            )

            # 2) 그 cluster 들을 참조하는 Photo.cluster_id -> NULL
            await self.db.execute(
                update(Photo)
                .where(Photo.cluster_id.in_(select(cluster_ids_subq.c.id)))
                .values(cluster_id=None)
            )

            # 3) Cluster 삭제
            await self.db.execute(
                delete(Cluster).where(Cluster.id.in_(select(cluster_ids_subq.c.id)))
            )

        # Update status
        job.status = JobStatus.PENDING
        await self.db.commit()
        logger.info(f"Job {job_id} status updated to PROCESSING.")

        # Trigger pipeline in background
        # background_tasks.add_task(run_pipeline_task, job_id, min_samples, max_dist_m, max_alt_diff_m)
        background_tasks.add_task(run_pipeline_task, job_id, min_samples, 3.0, max_alt_diff_m)
        logger.info(f"Clustering task for job {job_id} added to background tasks.")
        return 
 
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

        # 이미 돌고 있으면 막기
        # if job.status in {JobStatus.PROCESSING}:
        #     raise HTTPException(
        #         status_code=409,
        #         detail=f"Job {job_id} is already running deep clustering.",
        #     )

        # 상태를 PENDING -> RUNNING(soon) 으로 바꾸기 전에 표시만
        job.status = JobStatus.PENDING
        await self.db.commit()

        logger.info(f"Request deep_cluster logic start ")
        data = await run_deep_cluster_for_job(job_id)
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
