from __future__ import annotations

import logging
import os
import re
from typing import Any, List

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.core.config import configs
from app.db.database import AsyncSessionLocal
from app.domain.storage.factory import get_storage_client
from app.domain.storage.local import LocalStorageService
from app.models.job import ClusterJob, Job, JobStatus
from app.models.photo import Photo

logger = logging.getLogger(__name__)


async def call_cluster_service(
    bucket_path: str,
    photo_cnt: int,
    request_id: str,
    min_samples: int = 3,
    max_dist_m: float = 10.0,
    max_alt_diff_m: float = 20.0,
    similarity_threshold: float = 0.8,
    use_cache: bool = True,
    remove_people: bool = True,
) -> None:
    """
    image_cluster_server 의 /cluster 를 호출.
    결과는 webhook으로 수신.
    """
    # Use the configured callback base URL
    webhook_url = f"{configs.CALLBACK_BASE_URL}/cluster/callback"

    payload = {
        "bucket_path": bucket_path,
        "photo_cnt": photo_cnt,
        "webhook_url": webhook_url,
        "request_id": request_id,
        "min_samples": min_samples,
        "similarity_threshold": similarity_threshold,
        "use_cache": use_cache,
    }
    async with httpx.AsyncClient(
        base_url=str(configs.CLUSTER_SERVICE_URL),
        timeout=10.0,
    ) as client:
        resp = await client.post("/api/cluster", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Task submitted successfully: {data.get('task_id')}, request_id: {request_id}")
        return data

async def call_pdf_service(
    request_id: str, 
    bucket_path: str,
    cover_title: str,
    cover_company_name: str,
    clusters: list[dict],
    labels: dict
) -> dict:
    webhook_url = f"{configs.CALLBACK_BASE_URL}/pdf/callback"

    payload = {
        "request_id": request_id,
        "bucket_path": bucket_path,
        "cover_title": cover_title,
        "cover_company_name": cover_company_name,
        "clusters": clusters,
        "labels": labels,
        "webhook_url": webhook_url,
    }

    async with httpx.AsyncClient(
        base_url=str(configs.CLUSTER_SERVICE_URL), 
        timeout=10.0,
    ) as client:
        resp = await client.post("/api/pdf", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"PDF generation requested for {export_job_id}")
        return data
