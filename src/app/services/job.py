import asyncio
import io
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import piexif
from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from PIL import Image, ImageFile
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.domain.cluster_background import run_pipeline_task
from app.domain.cluster_client import run_deep_cluster_for_job
from app.domain.generate_pdf import generate_pdf_for_session
from app.domain.metadata_extractor import MetadataExtractor
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob, ExportStatus, Job, JobStatus
from app.models.photo import Photo
from app.models.user import User
from app.schemas.photo import (
    BatchPresignedUrlResponse,
    PhotoUploadRequest,
    PresignedUrlResponse,
)

logger = logging.getLogger(__name__)
# 부분 다운로드된 파일(Truncated Image) 처리 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AsyncBytesIO(io.BytesIO):
    async def read(self, *args, **kwargs):
        return super().read(*args, **kwargs)


def generate_thumbnail(image_data: bytes) -> bytes | None:
    """
    image_data: 파일의 전체 또는 일부(Partial) 바이트 데이터
    반환값: 썸네일 JPEG 바이트 또는 None
    """
    try:
        exif_dict = piexif.load(image_data)
        if exif_dict and exif_dict.get("thumbnail"):
            logger.info("Extracted embedded thumbnail via piexif.")
            return exif_dict["thumbnail"]
    except Exception:
        logger.info("Extracted embedded thumbnail via piexif.")
        pass
    return


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_jobs(self, user: User):
        logger.info(f"Fetching jobs for user: {user.user_id}")
        result = await self.db.execute(
            select(Job).where(Job.user_id == user.user_id).options(selectinload(Job.export_job))
        )
        jobs = result.scalars().all()
        logger.info(f"Found {len(jobs)} jobs for user: {user.user_id}")
        return jobs

    async def create_job(
        self, user: User, title: str, construction_type: Optional[str] = None, company_name: Optional[str] = None
    ):
        logger.info(f"User {user.user_id} creating job with title: '{title}'")

        if not company_name:
            company_name = user.company_name

        job = Job(user_id=user.user_id, title=title, construction_type=construction_type, company_name=company_name)
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
        result = await self.db.execute(
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.export_job.and_(ExportJob.finished_at.is_(None))))
        )
        job = result.scalars().first()
        if not job:
            logger.warning(f"Job with ID {job_id} not found.")
            raise HTTPException(status_code=404, detail="Job not found")

        logger.debug(f"Job {job_id} found with status {job.status}")
        return job

    async def generate_presigned_urls(self, job_id: str, files: list[PhotoUploadRequest]) -> BatchPresignedUrlResponse:
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

        # 1. Job 확인
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 2. Storage Client 초기화 (DI로 주입받거나 설정에서 가져옴)
        storage = get_storage_client()
        response_urls = []

        # MVP에서는 로컬 스토리지보다 GCS Resumable을 기본으로 추천

        strategy = "resumable"
        try:
            # 3. 각 파일별 Session URL 생성
            for file_req in files:
                target_path = f"{job.user_id}/{job.id}/photos/original/{file_req.filename}"

                # 여기서 GCS와 통신하여 세션을 엽니다.
                session_url = storage.generate_resumable_session_url(
                    target_path=target_path, content_type=file_req.content_type, origin=origin  # Use the passed origin
                )
                logger.info(f"request session_url from: '{origin}', generated session_url: '{session_url}'")

                response_urls.append(
                    PresignedUrlResponse(
                        filename=file_req.filename,
                        upload_url=session_url,  # 이것은 단순 URL이 아니라 이미 열린 세션 URL입니다.
                        storage_path=target_path,
                    )
                )
            return BatchPresignedUrlResponse(strategy=strategy, urls=response_urls)
        except Exception as e:
            logger.warning(f"Failed resumable: {e}")
            return await self.generate_presigned_urls(job_id, files)

    async def process_uploaded_files(self, job_id: str, file_info_list: list[dict]) -> list[Photo]:
        """
        Register files that have been uploaded directly to storage.
        Optimized for 4GB RAM environment using Semaphores and Temporary Files.
        """
        # 1. Job Validation
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        storage = get_storage_client()
        photos = []

        # 동시 실행 제한 (메모리 보호를 위해 동시에 10개까지만 무거운 작업 수행)
        semaphore = asyncio.Semaphore(10)
        extractor = MetadataExtractor()

        async def process_single_file(info: dict) -> Photo:
            async with semaphore:  # 여기서 동시 실행 수 제한
                storage_path = info["storage_path"]
                original_filename = os.path.basename(storage_path)
                original_filename_parts = os.path.splitext(original_filename)
                thumb_filename = f"{original_filename_parts[0]}_thumb.jpg"

                base_dir = Path(storage_path).parent.parent
                derived_thumbnail_path = str(base_dir / "thumbnail" / thumb_filename)

                # 사용자가 지정한 썸네일 경로가 있으면 사용, 없으면 유도된 경로 사용
                thumbnail_path = info.get("thumbnail_path", derived_thumbnail_path)
                if thumbnail_path == storage_path:
                    thumbnail_path = derived_thumbnail_path

                thumb_content = None
                meta_data = None

                # 임시 디렉토리 생성 (작업 후 자동 삭제됨)
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = Path(temp_dir) / original_filename

                    # --- [Step 1] 썸네일 생성 시도 (Partial Download) ---
                    # 네트워크 비용 절감을 위해 앞부분만 먼저 시도
                    if hasattr(storage, "download_partial_bytes"):
                        try:
                            # 100KB만 다운로드하여 embedded thumbnail 확인
                            partial_data = await asyncio.to_thread(
                                storage.download_partial_bytes, storage_path, 100 * 1024
                            )
                            if partial_data:
                                thumb_content = generate_thumbnail(partial_data)
                                if thumb_content:
                                    logger.info(f"Generated thumbnail from partial bytes for {original_filename}")
                        except Exception as e:
                            logger.debug(f"Partial thumbnail generation failed for {original_filename}: {e}")

                    # --- [Step 2] 전체 다운로드 (필요한 경우) ---
                    # 썸네일 생성에 실패했거나, 메타데이터 추출을 위해 원본이 필요한 경우
                    # 1MB 파일이므로 Disk에 쓰는 것이 메모리 관리에 유리함

                    full_file_downloaded = False

                    # 썸네일이 없거나 메타데이터 추출을 위해 파일 다운로드가 필요한 경우
                    if not thumb_content:
                        try:
                            # 전체 파일을 임시 경로로 다운로드
                            await storage.download_file(storage_path, temp_file_path)
                            full_file_downloaded = True

                            # Full byte로 썸네일 재시도
                            if temp_file_path.exists():
                                full_data = await asyncio.to_thread(temp_file_path.read_bytes)
                                thumb_content = generate_thumbnail(full_data)
                                if thumb_content:
                                    logger.info(f"Generated thumbnail from full file for {original_filename}")
                        except Exception as e:
                            logger.warning(f"Full download/thumbnail generation failed for {original_filename}: {e}")

                    # --- [Step 3] 썸네일 저장 ---
                    if thumb_content:
                        try:
                            await storage.save_file(AsyncBytesIO(thumb_content), derived_thumbnail_path, "image/jpeg")
                            thumbnail_path = derived_thumbnail_path
                        except Exception as e:
                            logger.warning(f"Failed to save thumbnail for {original_filename}: {e}")
                            thumbnail_path = None
                    else:
                        thumbnail_path = None

                    # --- [Step 4] 메타데이터 추출 (로컬 파일 기반) ---
                    # GCS URL(p.url) 대신 다운로드한 로컬 파일을 사용
                    try:
                        # 아직 다운로드 안 했다면(썸네일은 partial로 성공했지만 메타데이터가 필요한 경우) 다운로드
                        if not full_file_downloaded:
                            await storage.download_file(storage_path, temp_file_path)

                        if temp_file_path.exists():
                            # extractor가 파일 경로를 지원한다고 가정 (대부분 지원함)
                            # 만약 bytes만 지원한다면: await extractor.extract(temp_file_path.read_bytes())
                            meta = await extractor.extract(str(temp_file_path))
                            if meta:
                                meta_data = meta
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata for {original_filename}: {e}")

                # --- [Step 5] 객체 생성 ---
                photo = Photo(
                    job_id=job_id,
                    original_filename=info["filename"],
                    storage_path=storage_path,
                    thumbnail_path=thumbnail_path,
                    url=storage.get_url(storage_path),
                    thumbnail_url=storage.get_url(thumbnail_path) if thumbnail_path else None,
                )

                # 추출된 메타데이터 적용
                if meta_data:
                    photo.meta_lat = meta_data.lat
                    photo.meta_lon = meta_data.lon
                    photo.meta_timestamp = datetime.fromtimestamp(meta_data.timestamp) if meta_data.timestamp else None

                return photo

        # 모든 파일에 대해 비동기 작업 스케줄링 및 실행
        if file_info_list:
            photos = await asyncio.gather(*[process_single_file(info) for info in file_info_list])
            # None이 반환될 경우(에러 등)를 대비해 필터링 (필요 시)
            photos = [p for p in photos if p is not None]

        # DB 저장
        if photos:
            self.db.add_all(photos)
            await self.db.commit()

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
        storage = get_storage_client()  # Use the factory to get the correct storage service

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

            # Extract Metadata
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

        # Process files concurrently
        photos = await asyncio.gather(*[process_file(file) for file in files])

        self.db.add_all(photos)
        await self.db.commit()

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
        storage = get_storage_client()  # Use the factory to get the correct storage service
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
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 상태를 PENDING -> RUNNING(soon) 으로 바꾸기 전에 표시만
        job.status = JobStatus.PENDING
        await self.db.commit()

        logger.info(f"Request deep_cluster logic start ")
        # Pass parameters to the deep cluster runner
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
        result = await self.db.execute(select(Job).where(Job.id == job_id).options(selectinload(Job.user)))
        job = result.scalars().first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        result = await self.db.execute(
            select(ExportJob).where(ExportJob.job_id == job_id).order_by(ExportJob.created_at.desc())
        )

        export_job = result.scalars().first()
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

        self.db.add(export_job)
        await self.db.commit()
        await self.db.refresh(export_job)
        background_tasks.add_task(generate_pdf_for_session, export_job.id)
        return export_job

    async def get_export_job(self, job_id):
        result = await self.db.execute(
            select(ExportJob).where(ExportJob.job_id == job_id).order_by(ExportJob.created_at.desc())
        )
        export_job = result.scalars().first()
        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")
        return export_job.status, export_job.pdf_path, export_job.error_message

    async def download_export_pdf(self, job_id):
        logger.debug(f"Fetching job with ID: {job_id}")
        result = await self.db.execute(select(Job).where(Job.id == job_id).options(selectinload(Job.export_job)))
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
