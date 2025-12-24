import logging
from typing import Optional

from app.common.uow import UnitOfWork
from app.common.core_clients import call_pdf_service
from app.models.job import ExportJob, Job
from app.schemas.enum import ExportStatus
from app.models.photo import Photo
from app.models.user import User
from fastapi import HTTPException


logger = logging.getLogger(__name__)


class ExportService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def start_export(
        self,
        job_id: str,
        cover_title: Optional[str] = None,
        cover_company_name: Optional[str] = None,
        labels: Optional[dict] = {},
    ):
        job = await self.uow.jobs.get_job_details(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        export_job = await self.get_export_job(job_id)
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

        async with self.uow:
            export_job = await self.uow.jobs.create_export_job(export_job)
        
        if not cover_title:
            cover_title = job.title
        if not cover_company_name:
            cover_company_name = job.company_name
        
        try:
            response = await self.request_pdf_generation(
                job_id=job_id,
                export_job_id=export_job.id,
                cover_title=cover_title,
                cover_company_name=cover_company_name,
                clusters=job.clusters,
                labels=labels,
            )
            async with self.uow:
                if response["status"] == "processing":
                    export_job.status = ExportStatus.PROCESSING
                elif response["status"] == "failed":
                    export_job.status = ExportStatus.FAILED
                    export_job.error_message = response["message"]
                    export_job.finished_at = func.now()
            return export_job
        except Exception as e:
            logger.exception(f"job_id={job_id} export_job_id={export_job.id} export failed: {e}")
            async with self.uow:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = str(e)
                export_job.finished_at = func.now()
            return export_job

    async def request_pdf_generation(
        self, 
        job_id: str, 
        export_job_id: str,
        cover_title: str, 
        cover_company_name: str, 
        clusters: list,
        labels: dict,
    ):
        pdf_clusters = []
        for cluster in clusters:
            if cluster.title == "reserved":
                continue

            pdf_cluster = {
                "id": cluster.id,
                "title": cluster.title,
                "photos": [
                    {
                        "id": photo.id,
                        "url": photo.thumbnail_path if photo.thumbnail_path else photo.storage_path,
                        "timestamp": photo.timestamp,
                        "labels": photo.labels,
                    }
                    for photo in cluster.photos
                ],
            }
            pdf_clusters.append(pdf_cluster)

        return await call_pdf_service(
            request_id=export_job_id,
            cover_title=cover_title,
            cover_company_name=cover_company_name,
            clusters=pdf_clusters,
            labels=labels,
        )
    
    async def get_export_job(self, job_id):
        export_job = await self.uow.jobs.get_latest_export_job(job_id)
        if not export_job:
            raise HTTPException(status_code=404, detail="Export job not found")
        return export_job

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
