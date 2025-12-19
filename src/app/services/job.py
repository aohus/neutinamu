import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from PIL import ImageFile

from app.common.uow import UnitOfWork
from app.core.config import settings
from app.domain.cluster_background import run_pipeline_task
from app.domain.cluster_client import run_deep_cluster_for_job
from app.domain.generate_pdf import generate_pdf_for_session
from app.domain.metadata_extractor import MetadataExtractor
from app.domain.storage.factory import get_storage_client
from app.models.job import ExportJob, ExportStatus, Job, JobStatus
from app.models.photo import Photo
from app.models.user import User
from app.schemas.photo import (
    BatchPresignedUrlResponse,
    PhotoUploadRequest,
    PresignedUrlResponse,
)
from app.utils.fileIO import AsyncBytesIO
from app.utils.image import generate_thumbnail

logger = logging.getLogger(__name__)
# 부분 다운로드된 파일(Truncated Image) 처리 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True


class JobService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def get_jobs(self, user: User):
        logger.info(f"Fetching jobs for user: {user.user_id}")
        jobs = await self.uow.jobs.get_all_by_user_id(user.user_id)
        logger.info(f"Found {len(jobs)} jobs for user: {user.user_id}")
        return jobs

    async def create_job(
        self, user: User, title: str, construction_type: Optional[str] = None, company_name: Optional[str] = None
    ):
        logger.info(f"User {user.user_id} creating job with title: '{title}'")

        if not company_name:
            company_name = user.company_name

        job = Job(user_id=user.user_id, title=title, construction_type=construction_type, company_name=company_name)
        job = await self.uow.jobs.create_job(job)
        await self.uow.commit()
        logger.info(f"Job created successfully with ID: {job.id}")
        return job

    async def delete_job(self, job_id: str):
        logger.info(f"Deleting job: '{job_id}'")
        storage = get_storage_client()

        job = await self.uow.jobs.get_by_id(job_id)
        if job is not None:
            user_id = job.user_id
            await self.uow.jobs.delete_job(job)
            await self.uow.commit()

            job_object_path = f"{user_id}/{job_id}/"
            await storage.delete_directory(job_object_path)

        logger.info(f"Job deleted successfully with ID: {job_id}")
        return

    async def get_job(self, job_id: str):
        logger.debug(f"Fetching job with ID: {job_id}")
        job = await self.uow.jobs.get_by_id_with_unfinished_export(job_id)
        if not job:
            logger.warning(f"Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        logger.debug(f"Job {job_id} found with status {job.status}")
        return job

    async def get_job_details(self, job_id: str) -> Job:
        """Get full job details including photos and clusters."""
        job = await self.uow.jobs.get_job_details(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Adjust timestamp if meta_timestamp is present
        for cluster in job.clusters:
            for photo in cluster.photos:
                if photo.meta_timestamp:
                    photo.timestamp = photo.meta_timestamp

        return job

    async def generate_presigned_urls(self, job_id: str, files: list[PhotoUploadRequest]) -> BatchPresignedUrlResponse:
        """
        Generate pre-signed URLs for direct upload.
        If storage is local, returns None for URLs, indicating proxy upload should be used.
        """
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()
        response_urls = []

        strategy = "proxy" if settings.STORAGE_TYPE == "local" else "presigned"

        for file_req in files:
            target_path = f"{job.user_id}/{job.id}/photos/original/{file_req.filename}"
            upload_url = storage.generate_upload_url(target_path, content_type=file_req.content_type)
            response_urls.append(
                PresignedUrlResponse(filename=file_req.filename, upload_url=upload_url, storage_path=target_path)
            )

        return BatchPresignedUrlResponse(strategy=strategy, urls=response_urls)

    async def generate_upload_urls(
        self, job_id: str, files: list[PhotoUploadRequest], origin: str = None
    ) -> BatchPresignedUrlResponse:

        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()
        response_urls = []

        strategy = "resumable"
        try:
            for file_req in files:
                target_path = f"{job.user_id}/{job.id}/photos/original/{file_req.filename}"

                session_url = storage.generate_resumable_session_url(
                    target_path=target_path, content_type=file_req.content_type, origin=origin
                )
                logger.info(f"request session_url from: '{origin}', generated session_url: '{session_url}'")

                response_urls.append(
                    PresignedUrlResponse(
                        filename=file_req.filename,
                        upload_url=session_url,  # 열린 세션 URL
                        storage_path=target_path,
                    )
                )
            return BatchPresignedUrlResponse(strategy=strategy, urls=response_urls)
        except Exception as e:
            logger.warning(f"Failed resumable: {e}")
            return await self.generate_presigned_urls(job_id, files)

    async def set_job_uploading(self, job_id: str) -> datetime:
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job.status = JobStatus.UPLOADING
        job.updated_at = datetime.now().astimezone()
        await self.uow.jobs.save(job)
        await self.uow.commit()
        return job.updated_at

    async def process_uploaded_files(
        self, job_id: str, file_info_list: list[dict], trigger_timestamp: Optional[datetime] = None
    ) -> list[Photo]:
        """
        Register files that have been uploaded directly to storage.
        Optimized for 4GB RAM environment using Semaphores and Temporary Files.
        """
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()
        photos = []

        semaphore = asyncio.Semaphore(10)
        extractor = MetadataExtractor()

        if file_info_list:
            photos = await asyncio.gather(
                *[self._process_single_file(job_id, info, storage, extractor, semaphore) for info in file_info_list]
            )
            photos = [p for p in photos if p is not None]

        if photos:
            await self.uow.jobs.add_all(photos)

        if trigger_timestamp:
            await self.uow.jobs.update_status_by_id(job_id, JobStatus.CREATED, trigger_timestamp)

        await self.uow.commit()

        logger.info(f"Registered {len(photos)} uploaded photos for job {job_id}")
        return photos

    async def _process_single_file(
        self, job_id: str, info: dict, storage, extractor: MetadataExtractor, semaphore: asyncio.Semaphore
    ) -> Optional[Photo]:
        async with semaphore:
            storage_path = info["storage_path"]
            original_filename = os.path.basename(storage_path)
            original_filename_parts = os.path.splitext(original_filename)
            thumb_filename = f"{original_filename_parts[0]}_thumb.jpg"

            base_dir = Path(storage_path).parent.parent
            derived_thumbnail_path = str(base_dir / "thumbnail" / thumb_filename)

            thumbnail_path = info.get("thumbnail_path", derived_thumbnail_path)
            if thumbnail_path == storage_path:
                thumbnail_path = derived_thumbnail_path

            thumb_content = None
            meta_data = None

            # 임시 디렉토리 생성 (작업 후 자동 삭제됨)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir) / original_filename

                if hasattr(storage, "download_partial_bytes"):
                    try:
                        # 100KB만 다운로드하여 embedded thumbnail 확인
                        partial_data = await asyncio.to_thread(storage.download_partial_bytes, storage_path, 100 * 1024)
                        if partial_data:
                            thumb_content = generate_thumbnail(partial_data)
                            if thumb_content:
                                logger.info(f"Generated thumbnail from partial bytes for {original_filename}")
                    except Exception as e:
                        logger.debug(f"Partial thumbnail generation failed for {original_filename}: {e}")

                full_file_downloaded = False
                if not thumb_content:
                    try:
                        await storage.download_file(storage_path, temp_file_path)
                        full_file_downloaded = True

                        if temp_file_path.exists():
                            full_data = await asyncio.to_thread(temp_file_path.read_bytes)
                            thumb_content = generate_thumbnail(full_data, is_full_image=True)
                            if thumb_content:
                                logger.info(f"Generated thumbnail from full file for {original_filename}")
                    except Exception as e:
                        logger.warning(f"Full download/thumbnail generation failed for {original_filename}: {e}")

                if thumb_content:
                    try:
                        await storage.save_file(AsyncBytesIO(thumb_content), derived_thumbnail_path, "image/jpeg")
                        thumbnail_path = derived_thumbnail_path
                    except Exception as e:
                        logger.warning(f"Failed to save thumbnail for {original_filename}: {e}")
                        thumbnail_path = None
                else:
                    thumbnail_path = None

                try:
                    if not full_file_downloaded:
                        await storage.download_file(storage_path, temp_file_path)

                    if temp_file_path.exists():
                        meta = await extractor.extract(str(temp_file_path))
                        if meta:
                            meta_data = meta
                except Exception as e:
                    logger.warning(f"Failed to extract metadata for {original_filename}: {e}")

            photo = Photo(
                job_id=job_id,
                original_filename=info["filename"],
                storage_path=storage_path,
                thumbnail_path=thumbnail_path,
                url=storage.get_url(storage_path),
                thumbnail_url=storage.get_url(thumbnail_path) if thumbnail_path else None,
            )

            if meta_data:
                photo.meta_lat = meta_data.lat
                photo.meta_lon = meta_data.lon
                photo.meta_timestamp = datetime.fromtimestamp(meta_data.timestamp) if meta_data.timestamp else None

            return photo

    async def upload_photos(
        self,
        job_id: str,
        files: list[UploadFile] = File(...),
    ):
        logger.info(f"Uploading {len(files)} files for job ID: {job_id}")
        # Check if job exists
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            logger.error(f"Upload failed: Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()

        async def process_file(file: UploadFile):
            logger.debug(f"Processing file: {file.filename}")

            target_path = f"{job.user_id}/{job.id}/photos/original/{file.filename}"
            content = await file.read()
            saved_path = await storage.save_file(AsyncBytesIO(content), target_path, file.content_type)

            thumb_content = generate_thumbnail(content)
            thumbnail_path = None
            if thumb_content:
                original_filename_parts = os.path.splitext(os.path.basename(file.filename))
                thumb_filename = f"{original_filename_parts[0]}_thumb.jpg"
                thumb_target_path = f"{job.user_id}/{job.id}/photos/thumbnail/{thumb_filename}"
                thumbnail_path = await storage.save_file(AsyncBytesIO(thumb_content), thumb_target_path, "image/jpeg")

            extractor = MetadataExtractor()
            meta = extractor.extract_from_bytes(content, file.filename)

            photo = Photo(
                job_id=job_id,
                original_filename=file.filename,
                storage_path=saved_path,
                thumbnail_path=None,
                url=storage.get_url(saved_path),
                thumbnail_url=None,
                # thumbnail_url=storage.get_url(thumbnail_path) if thumbnail_path else None
                meta_lat=meta.lat,
                meta_lon=meta.lon,
                meta_timestamp=datetime.fromtimestamp(meta.timestamp) if meta.timestamp else None,
            )
            logger.debug(f"Saved {file.filename} to {saved_path}, thumbnail: {thumbnail_path}")
            return photo

        photos = await asyncio.gather(*[process_file(file) for file in files])

        await self.uow.jobs.add_all(photos)
        await self.uow.commit()

        logger.info(f"Successfully uploaded and saved {len(photos)} photos for job {job_id}.")
        return photos

    async def start_cluster(
        self,
        job_id: str,
        background_tasks: BackgroundTasks,
        min_samples: int = 3,
        max_dist_m: float = 10.0,
        max_alt_diff_m: float = 20.0,
    ):
        """Trigger the clustering pipeline."""
        logger.info(f"Received request to start clustering for job {job_id}")
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            logger.error(f"Clustering trigger failed: Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status == JobStatus.PROCESSING:
            raise HTTPException(status_code=404, detail="Job is now PROCESSING")

        job.status = JobStatus.PENDING
        await self.uow.jobs.save(job)
        await self.uow.commit()
        logger.info(f"Job {job_id} status updated to PENDING.")

        storage = get_storage_client()
        background_tasks.add_task(run_pipeline_task, job_id, storage, min_samples, max_dist_m, max_alt_diff_m)
        logger.info(f"Clustering task for job {job_id} added to background tasks.")

        data = {"message": "Local clustering started"}
        return job, data

    async def start_cluster_server(
        self,
        job_id: str,
        background_tasks: BackgroundTasks,
        min_samples: int = 3,
        max_dist_m: float = 10.0,
        max_alt_diff_m: float = 20.0,
    ):
        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job.status = JobStatus.PENDING
        await self.uow.jobs.save(job)
        await self.uow.commit()

        logger.info(f"Request deep_cluster logic start ")
        data = await run_deep_cluster_for_job(
            job_id, min_samples=min_samples, max_dist_m=max_dist_m, max_alt_diff_m=max_alt_diff_m
        )
        return job, data

    async def start_export(
        self,
        job_id: str,
        background_tasks: BackgroundTasks,
        cover_title: Optional[str] = None,
        cover_company_name: Optional[str] = None,
        labels: Optional[dict] = {},
    ):
        job = await self.uow.jobs.get_by_id_with_user(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        export_job = await self.uow.jobs.get_latest_export_job(job_id)

        if export_job and export_job.status in {
            ExportStatus.PENDING,
            ExportStatus.PROCESSING,
        }:
            return export_job

        export_job = ExportJob(
            job_id=job.id,
            status=ExportStatus.PENDING,
            cover_title=cover_title or job.title,
            cover_company_name=cover_company_name or job.company_name,
            labels=labels,
        )

        export_job = await self.uow.jobs.create_export_job(export_job)
        await self.uow.commit()
        background_tasks.add_task(generate_pdf_for_session, export_job.id)
        return export_job

    async def get_export_job(self, job_id):
        export_job = await self.uow.jobs.get_latest_export_job(job_id)
        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")
        return export_job.status, export_job.pdf_path, export_job.error_message

    async def download_export_pdf(self, job_id):
        logger.debug(f"Fetching job with ID: {job_id}")

        job = await self.uow.jobs.get_by_id(job_id)
        if not job:
            logger.warning(f"Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        export_job = await self.uow.jobs.get_latest_export_job(job_id)
        if not export_job or export_job.status != ExportStatus.EXPORTED or not export_job.pdf_path:
            raise HTTPException(status_code=404, detail="No finished export for this session")

        target_path = export_job.pdf_path
        if not target_path:
            raise HTTPException(status_code=404, detail="PDF file not found")
        filename = f"{job.title}.pdf"
        return target_path, filename
