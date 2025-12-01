from __future__ import annotations

import logging
from typing import Any, List

import httpx
from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.cluster import Cluster
from app.models.job import Job, Status
from app.models.photo import Photo
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def run_deep_cluster_for_job(job_id: str):
    """
    1) job_id 에 속한 Photo들의 file_path 조회
    2) image_cluster_server 호출
    3) Cluster / Photo.cluster_id 갱신
    4) Job.status 업데이트
    """
    logger.info(f"[DeepClusterRunner] start job_id={job_id}")

    # 새 세션을 직접 열어 사용 (BackgroundTask/asyncio.create_task 에서 종종 이렇게 처리)
    async with AsyncSessionLocal() as session:
        # 1. Job & Photo 조회
        job = await _get_job(session, job_id)
        if job is None:
            logger.error(f"[DeepClusterRunner] job_id={job_id} not found")
            return

        # 상태 업데이트: RUNNING
        job.status = Status.RUNNING
        await session.commit()

        photos = await _get_photos_for_job(session, job_id)
        if not photos:
            logger.warning(f"[DeepClusterRunner] job_id={job_id} has no photos")
            job.status = Status.FAILED
            await session.commit()
            return

        photo_paths = [("/Users/aohus/Workspaces/github/job-report-creator/backend/src/assets/" + p.storage_path) for p in photos]

        try:
            # 2. 클러스터 서버 호출
            cluster_json = await call_cluster_service(photo_paths=photo_paths)

            # 3. DB 반영
            await _apply_cluster_result(session, job_id, photos, cluster_json)

            # 4. Job 완료
            job.status = Status.DONE
            await session.commit()
            logger.info(
                f"[DeepClusterRunner] job_id={job_id} done. "
                f"{cluster_json.get('total_clusters')} clusters."
            )

        except Exception as e:
            logger.exception(f"[DeepClusterRunner] job_id={job_id} failed: {e}")
            job.status = Status.FAILED
            await session.commit()


async def _get_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def _get_photos_for_job(session: AsyncSession, job_id: str) -> List[Photo]:
    result = await session.execute(select(Photo).where(Photo.job_id == job_id))
    return list(result.scalars().all())


async def _apply_cluster_result(
    session: AsyncSession,
    job_id: str,
    photos: List[Photo],
    cluster_json: dict,
):
    """
    cluster_json:
    {
      "clusters": [
        {
          "id": 0,
          "photos": ["/path/a.jpg", ...],
          "count": ...,
          "avg_similarity": ...,
          "quality_score": ...
        }, ...
      ],
      ...
    }
    """
    # 기존 클러스터/매핑 삭제 후 다시 저장 (idempotent하게 가는 전략)
    await session.execute(delete(Cluster).where(Cluster.job_id == job_id))
    for p in photos:
        p.cluster_id = None

    # file_path -> Photo 객체 맵
    path_to_photo = {p.file_path: p for p in photos}

    clusters = cluster_json.get("clusters", [])
    for c in clusters:
        label = int(c["id"])
        count = int(c["count"])
        avg_sim = float(c["avg_similarity"])
        quality = float(c["quality_score"])

        cluster_row = Cluster(
            job_id=job_id,
            label=label,
            count=count,
            avg_similarity=avg_sim,
            quality_score=quality,
        )
        session.add(cluster_row)
        await session.flush()  # cluster_row.id 확보

        for path in c["photos"]:
            photo = path_to_photo.get(path)
            if not photo:
                logger.warning(
                    f"[DeepClusterRunner] job_id={job_id} path not found in DB: {path}"
                )
                continue
            photo.cluster_id = cluster_row.id

    await session.commit()


async def call_cluster_service(
    photo_paths: list[str],
    similarity_threshold: float = 0.6,
    use_cache: bool = True,
    remove_people: bool = True,
) -> dict[str, Any]:
    """
    image_cluster_server 의 /cluster 를 호출해서 결과 JSON 을 반환.
    """
    payload = {
        "photo_paths": photo_paths,
        "similarity_threshold": similarity_threshold,
        "use_cache": use_cache,
        "remove_people": remove_people,
    }

    async with httpx.AsyncClient(
        base_url=str(settings.CLUSTER_SERVICE_URL),
        timeout=600.0,  # 사진 수에 따라 넉넉하게
    ) as client:
        resp = await client.post("/cluster", json=payload)
        resp.raise_for_status()
        return resp.json()