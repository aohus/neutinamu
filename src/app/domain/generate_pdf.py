import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
import fitz  # PyMuPDF
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# --- [Project Dependencies] ---
from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob, Job
from app.models.photo import Photo
from app.schemas.enum import ExportStatus

logger = logging.getLogger(__name__)


@dataclass
class PDFLayoutConfig:
    """A4 레이아웃 및 템플릿 설정값"""

    # 폰트
    FONT_PATH: str = "/app/fonts/AppleGothic.ttf"
    FONT_NAME: str = "AppleGothic"

    # 템플릿 GCS 경로 (설정 파일 등에서 관리 권장)
    TEMPLATE_GCS_PATH: str = (
        settings.PDF_BASE_TEMPLATE_PATH if settings.PDF_BASE_TEMPLATE_PATH else "templates/base_template.pdf"
    )

    # 레이아웃 치수 (pt 단위)
    PAGE_WIDTH: float = 595
    PAGE_HEIGHT: float = 842

    # 데이터 배치 영역 (템플릿의 선 위치에 맞춰 조정 필요)
    # 헤더
    HEADER_TITLE_RECT = fitz.Rect(146, 122, 400, 142)  # 공종/제목 들어갈 곳

    # 좌측 라벨 컬럼 너비
    LABEL_COL_WIDTH: float = 61

    # 사진 배치 영역 시작점 및 크기
    # 첫번째 행(전)의 이미지 박스 영역 (테두리 제외하고 사진이 들어갈 내부 좌표)
    ROW_1_TOP: float = 148
    ROW_HEIGHT: float = 214

    PHOTO_HEIGHT: float = 213  # 사진 높이
    PHOTO_WIDTH: float = 373  # 사진 비율에 맞춰야함.

    # 사진 캡션 박스
    CAPTION_WIDTH: float = 78
    CAPTION_HEIGHT: float = 26


LAYOUT = PDFLayoutConfig()


class PDFGenerator:
    """
    데이터를 받아 PDF를 생성하는 역할만 전담하는 클래스
    """

    def __init__(self, tmp_dir: str):
        self.tmp_dir = Path(tmp_dir)
        self.font_name = Path(LAYOUT.FONT_NAME)
        self.font_path = Path(LAYOUT.FONT_PATH)
        self.template_path = self.tmp_dir / "template.pdf"
        self.output_path = self.tmp_dir / f"result_{int(datetime.now().timestamp())}.pdf"

    def _register_font(self, doc: fitz.Document):
        if self.font_path.exists():
            # 페이지마다 폰트를 쓰기 위해 글로벌하게 등록하거나, insert_text시 fontfile 지정
            # 여기서는 편의상 파일 경로를 멤버 변수로 유지하여 사용
            pass
        else:
            logger.warning(f"Font file missing: {self.font_path}")

    def _download_template_if_needed(self, storage_client):
        """
        GCS에서 템플릿 다운로드 (로컬 캐싱 가능)
        """
        if self.template_path.exists():
            return

        try:
            # 이 메소드는 run_in_executor 내부에서 불리므로 동기 client나
            # 비동기 함수라면 loop.run_until_complete 등을 고려해야 함.
            # 여기서는 편의상 외부에서 다운로드 받아 경로만 넘겨주는 방식을 추천하지만,
            # 구조상 여기서 처리하려면 동기식 다운로드가 필요함.
            pass
        except Exception as e:
            logger.error(f"Failed to download template: {e}")

    def generate(self, export_info: dict, clusters: List[dict], template_local_path: Path) -> str:
        """
        메인 생성 로직 (CPU Bound)
        :param export_info: {cover_title, cover_company_name, label_config: {visible_keys, overrides}, ...}
        :param clusters: [{name, photos: [{local_path, db_labels, timestamp}, ...]}, ...]
        :param template_local_path: 다운로드 된 템플릿 경로
        :return: 생성된 PDF 경로
        """
        if template_local_path and template_local_path.exists():
            src_doc = fitz.open(template_local_path)
        else:
            logger.error("Template not found.")
            return None

        out_doc = fitz.open()

        font_file = str(self.font_path) if self.font_path.exists() else None
        FONTNAME = "AppleGothic"
        font_alias = None

        if font_file:
            try:
                font_xref = out_doc.add_font(fontfile=str(font_file), fontname=FONTNAME)
                font_alias = FONTNAME
            except Exception as e:
                logger.error(f"FATAL: Failed to embed font {font_file}: {e}. Falling back to default.")
                font_alias = "kr"

        label_config = export_info.get("label_config", {})
        visible_keys = label_config.get("visible_keys", [])
        global_overrides = label_config.get("overrides", {})

        for cluster in clusters:
            out_doc.insert_pdf(src_doc, from_page=0, to_page=0)
            page = out_doc[-1]

            page.insert_textbox(
                LAYOUT.HEADER_TITLE_RECT,
                cluster["name"],
                fontname=font_alias,
                fontfile=font_file,
                fontsize=12,
                align=fitz.TEXT_ALIGN_LEFT,
            )

            photos = cluster["photos"]
            row_start_y = LAYOUT.ROW_1_TOP

            for idx in range(3):
                if idx >= len(photos):
                    continue

                photo_data = photos[idx]
                img_path = photo_data.get("local_path")

                if not img_path or not os.path.exists(img_path):
                    continue

                current_row_y = row_start_y + (idx * LAYOUT.ROW_HEIGHT)
                row_center_y = current_row_y + (LAYOUT.ROW_HEIGHT / 2)

                img_w = LAYOUT.PHOTO_WIDTH
                img_h = LAYOUT.PHOTO_HEIGHT
                img_x = LAYOUT.LABEL_COL_WIDTH + (LAYOUT.PAGE_WIDTH - LAYOUT.LABEL_COL_WIDTH - img_w) / 2
                img_y = row_center_y - (img_h / 2)

                img_rect = fitz.Rect(img_x, img_y, img_x + img_w, img_y + img_h)

                try:
                    page.insert_image(img_rect, filename=str(img_path))
                except Exception as e:
                    logger.error(f"Image insert failed: {e}")
                    continue

                # 라벨 텍스트 생성 (Individual > Global > Special > Default)
                final_labels = {}
                db_labels = photo_data.get("db_labels") or {}
                photo_timestamp = photo_data.get("timestamp")

                for key in visible_keys:
                    val = None
                    if key in db_labels and db_labels[key]:
                        val = db_labels[key]
                    elif key in global_overrides and global_overrides[key]:
                        val = global_overrides[key]
                    elif key == "일자" and photo_timestamp:
                        try:
                            val = photo_timestamp.strftime("%Y.%m.%d")
                        except:
                            val = str(photo_timestamp)
                    elif key == "시행처":
                        val = export_info.get("cover_company_name") or "-"

                    if val is not None:
                        final_labels[key] = val

                logger.info(f"Final labels for photo: {final_labels}, {visible_keys}, {db_labels}, {global_overrides}")
                if final_labels:
                    self._draw_caption(page, img_rect, final_labels, font_alias, font_file)

        out_doc.save(self.output_path)
        out_doc.close()
        src_doc.close()
        return str(self.output_path)

    def _draw_caption(self, page, img_rect, labels, font_alias, font_file):
        """사진 위에 설명 박스 그리기"""
        label_lines = len(labels)
        cap_w = LAYOUT.CAPTION_WIDTH
        cap_h = LAYOUT.CAPTION_HEIGHT * (label_lines / 2)

        cap_x0 = img_rect.x0 - 2
        cap_y0 = img_rect.y0
        cap_x1 = cap_x0 + cap_w
        cap_y1 = cap_y0 + cap_h

        cap_rect = fitz.Rect(cap_x0, cap_y0, cap_x1, cap_y1)

        page.draw_rect(cap_rect, color=(1, 1, 1), fill=(1, 1, 1))
        page.draw_rect(cap_rect, color=(0.7, 0.7, 0.7), width=0.5)

        PAD_LEFT = 2
        PAD_TOP = 4

        text_rect = fitz.Rect(
            cap_rect.x0 + PAD_LEFT,
            cap_rect.y0 + PAD_TOP,
            cap_rect.x1,
            cap_rect.y1,
        )

        # 딕셔너리 순회
        logger.info(f"Drawing caption: {labels}")
        text = "\n".join(f"{k} : {v}" for k, v in labels.items())
        page.insert_textbox(
            text_rect, text, fontname=font_alias, fontfile=font_file, fontsize=6, align=fitz.TEXT_ALIGN_LEFT
        )


async def generate_pdf_for_session(export_job_id: str):
    """
    ExportJob 처리 메인 함수 (Config-Driven)
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            select(ExportJob)
            .options(
                selectinload(ExportJob.job).selectinload(Job.user),
                selectinload(ExportJob.job).selectinload(Job.photos),
            )
            .where(ExportJob.id == export_job_id)
        )
        result = await session.execute(stmt)
        export_job = result.scalars().first()

        if not export_job:
            logger.error(f"ExportJob {export_job_id} not found.")
            return

        # 상태 업데이트
        export_job.status = ExportStatus.PROCESSING
        await session.commit()

        try:
            job = export_job.job
            user = job.user

            label_config = export_job.labels or {}  # { visible_keys: [], overrides: {} }
            cover_date_str = label_config.get("overrides", {}).get("일자")
            if not cover_date_str:
                timestamps = [p.meta_timestamp for p in job.photos if p.meta_timestamp]
                cover_date_str = (
                    min(timestamps).strftime("%Y.%m.%d") if timestamps else datetime.now().strftime("%Y.%m.%d")
                )

            export_info = {
                "cover_title": export_job.cover_title,
                "cover_company_name": export_job.cover_company_name,
                "date": cover_date_str,
                "label_config": label_config,
            }

            stmt_c = (
                select(Cluster)
                .where(Cluster.job_id == job.id)
                .where(Cluster.name != "reserve")
                .order_by(Cluster.order_index.asc())
            )
            result_c = await session.execute(stmt_c)
            clusters_db = result_c.scalars().all()

            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                storage_client = get_storage_client()

                template_local_path = tmpdir / "template_base.pdf"
                try:
                    if settings.STORAGE_TYPE in ["gcs", "s3"]:
                        await storage_client.download_file(LAYOUT.TEMPLATE_GCS_PATH, template_local_path)
                    else:
                        # 로컬 개발 환경용 fallback
                        import shutil

                        shutil.copy("/app/assets/template1.pdf", template_local_path)
                        pass
                except Exception as e:
                    logger.error(f"Template download failed: {e}")
                    raise e

                processed_clusters = []
                for cluster in clusters_db:
                    cluster_data = {"name": cluster.name or f"Cluster #{cluster.id}", "photos": []}

                    stmt_p = (
                        select(Photo)
                        .where(Photo.cluster_id == cluster.id, Photo.deleted_at.is_(None))
                        .order_by(Photo.order_index.asc())
                        .limit(3)
                    )
                    res_p = await session.execute(stmt_p)
                    photos_db = res_p.scalars().all()

                    for photo in photos_db:
                        photo_local_path = tmpdir / Path(photo.url).name
                        if settings.STORAGE_TYPE in ["gcs", "s3"] and photo.url:
                            try:
                                await storage_client.download_file(photo.url, photo_local_path)
                            except Exception as e:
                                logger.warning(f"Photo download failed {photo.id}: {e}")
                                photo_local_path = None
                        else:
                            photo_local_path = Path("/app/assets") / photo.storage_path

                        cluster_data["photos"].append(
                            {
                                "local_path": photo_local_path,
                                "id": str(photo.id),
                                "db_labels": photo.labels,
                                "timestamp": photo.meta_timestamp,
                            }
                        )

                    processed_clusters.append(cluster_data)

                pdf_gen = PDFGenerator(tmpdir_str)
                loop = asyncio.get_event_loop()

                def _run_gen():
                    return pdf_gen.generate(export_info, processed_clusters, template_local_path)

                generated_pdf_path = await loop.run_in_executor(None, _run_gen)

                if not generated_pdf_path:
                    raise Exception("PDF Generation returned None")

                file_name = f"{job.id}_report_{datetime.now()}.pdf"
                final_url = None

                if settings.STORAGE_TYPE in ["gcs", "s3"]:
                    storage_path = f"{user.user_id}/{job.id}/exports/{file_name}"
                    async with aiofiles.open(generated_pdf_path, "rb") as f:
                        await storage_client.save_file(f, storage_path, content_type="application/pdf")
                    final_url = storage_client.get_url(storage_path)
                else:
                    target_dir = Path(settings.MEDIA_ROOT) / "exports"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / file_name
                    import shutil

                    shutil.copy(generated_pdf_path, target_path)
                    final_url = f"{settings.MEDIA_URL}/exports/{file_name}"

                export_job.status = ExportStatus.EXPORTED
                export_job.pdf_path = final_url
                export_job.finished_at = datetime.now()
                await session.commit()

                logger.info(f"PDF Generated successfully: {final_url}")

        except Exception as e:
            logger.exception("PDF Generation Failed")
            export_job.status = ExportStatus.FAILED
            export_job.error_message = str(e)
            export_job.finished_at = datetime.now()
            await session.commit()
