import logging
import httpx
from app.core.config import configs

logger = logging.getLogger(__name__)

async def request_pdf_generation(export_job_id: str) -> dict:
    async with httpx.AsyncClient(base_url=str(configs.CLUSTER_SERVICE_URL), timeout=10.0) as client:
        resp = await client.post("/api/pdf", json={"export_job_id": export_job_id})
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"PDF generation requested for {export_job_id}")
        return data
