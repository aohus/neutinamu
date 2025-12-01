import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence

from app.core.config import JobConfig
from app.db.database import AsyncSessionLocal
from app.domain.pipeline import PhotoClusteringPipeline
from app.domain.storage.local import LocalStorageService as StorageService
from app.models.cluster import Cluster
from app.models.job import ExportJob, Job, Status
from app.models.photo import Photo
from app.models.user import User
from app.utils.generate_pdf import generate_pdf_for_session
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
        result = await self.db.execute(select(Job).where(Job.user_id == user.user_id))
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

        await self.db.execute(
            delete(Job).where(Job.id == job_id)
        )
        await self.db.commit()
        await self.db.refresh()
        logger.info(f"Job deleted successfully with ID: {job_id}")
        return 

    async def get_job(self, job_id: str):
        logger.debug(f"Fetching job with ID: {job_id}")
        result = await self.db.execute(select(Job).where(Job.id == job_id))
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

        if job.status == Status.RUNNING:
            raise HTTPException(status_code=404, detail="Job is now PROCESSING")
        
        if job.status == "COMPLETED":
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
        job.status = Status.PENDING
        await self.db.commit()
        logger.info(f"Job {job_id} status updated to PROCESSING.")

        # Trigger pipeline in background
        # background_tasks.add_task(run_pipeline_task, job_id, min_samples, max_dist_m, max_alt_diff_m)
        background_tasks.add_task(run_pipeline_task, job_id, min_samples, 3.0, max_alt_diff_m)
        logger.info(f"Clustering task for job {job_id} added to background tasks.")
        return 

    async def start_export(self, job_id: str, background_tasks: BackgroundTasks):
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # if job.export_job and job.export_job.status in {
        #     Status.PENDING,
        #     Status.RUNNING,
        # }:
        #     raise HTTPException(status_code=400, detail="Export already in progress")

        export_job = ExportJob(job_id=job.id, status=Status.PENDING)
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
# Clustering 
# ============================================
async def run_pipeline_task(
    job_id: str,
    min_samples: int = 3, 
    max_dist_m: float = 10.0, 
    max_alt_diff_m: float = 20.0
    # session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Background task to run the clustering pipeline for a job.

    - job을 PROCESSING 으로 변경
    - 파이프라인 실행
    - 클러스터/포토 업데이트
    - 성공 시 COMPLETED, 실패 시 FAILED
    """
    logger.info("Starting pipeline for job %s", job_id)

    async with AsyncSessionLocal() as session:
        try:
            await _mark_job_status(session, job_id, "PROCESSING")
            cluster_groups = await _run_pipeline(job_id, min_samples, max_dist_m, max_alt_diff_m)
            await _create_clusters_from_result(session, job_id, cluster_groups)
            await _mark_job_status(session, job_id, "COMPLETED")
            await session.commit()
            logger.info("Pipeline for job %s completed successfully.", job_id)
        except Exception as exc:
            logger.exception("Pipeline for job %s failed: %s", job_id, exc)
            await session.rollback()

            # 실패 시 상태를 FAILED 로 기록
            async with AsyncSessionLocal() as session2:
                await _mark_job_status(session2, job_id, "FAILED")
                await session2.commit()
                logger.warning("Job %s status updated to FAILED.", job_id)

async def _run_pipeline(
    job_id: str,
    min_samples: int, 
    max_dist_m: float, 
    max_alt_diff_m: float,
) -> Sequence[Sequence[Any]]:
    """
    실제 파이프라인 실행만 담당.
    cluster_groups: [[photo_obj, ...], ...] 형태를 기대.
    photo_obj 는 최소한 .timestamp, .path 를 가진 객체라고 가정.
    """
    config = JobConfig(job_id=job_id, 
                         min_samples=min_samples, 
                         max_dist_m=max_dist_m, 
                         max_alt_diff_m=max_alt_diff_m)
    storage = StorageService(config)
    pipeline = PhotoClusteringPipeline(storage)

    clusters = await pipeline.run()
    return clusters


async def _mark_job_status(
    session: AsyncSession,
    job_id: int,
    status: str,
) -> None:
    job = await session.get(Job, job_id)
    if not job:
        logger.warning("Job %s not found while setting status to %s", job_id, status)
        return
    job.status = status
    job.updated_at = datetime.now()  # updated_at 필드가 있다면


async def _create_clusters_from_result(
    session: AsyncSession,
    job_id: int,
    cluster_groups: Sequence[Sequence[Any]],
) -> list[Photo]:
    """
    파이프라인 결과(클러스터별 사진 그룹)를 기반으로
    Cluster / Photo 테이블을 업데이트.

    cluster_groups: 각 요소는 사진 객체 리스트
    사진 객체는 최소한 .timestamp, .path 속성을 가진다고 가정.
    """
    job = await session.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    updated_photos: list[Photo] = []
    db_cluster = Cluster(
        job_id=job_id,
        name="reserve",
        order_index=-1,
    )
    session.add(db_cluster)

    job_title = job.title
    for cluster_index, photo_group in enumerate(cluster_groups):
        # 1) Cluster 레코드 생성
        db_cluster = Cluster(
            job_id=job_id,
            name=f"{job_title}",
            order_index=cluster_index,
        )
        session.add(db_cluster)
        # cluster_id 를 Photo에 써야 하므로 flush 로 PK 확보
        await session.flush()

        # 2) 그룹 내 사진 정렬
        sorted_group = sorted(
            photo_group,
            key=lambda p: (
                p.timestamp is None,
                p.timestamp if p.timestamp is not None else 0.0,
                p.path,
            ),
        )

        # 3) Photo 레코드 업데이트
        for order_index, photo_obj in enumerate(sorted_group):
            result = await session.execute(
                select(Photo).where(Photo.original_filename == photo_obj.original_name)
            )
            photo = result.scalars().first()

            if not photo:
                logger.warning(
                    "Photo not found for path %s in job %s", photo_obj.path, job_id
                )
                continue

            photo.cluster_id = db_cluster.id
            photo.order_index = order_index
            updated_photos.append(photo)
    return updated_photos


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
