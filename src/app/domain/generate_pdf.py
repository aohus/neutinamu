import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
import fitz  # PyMuPDF
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# --- [Project Dependencies] ---
from app.core.config import configs
from app.db.database import AsyncSessionLocal
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob, Job
from app.models.photo import Photo
from app.schemas.enum import ExportStatus
from app.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class PDFLayoutConfig:
    """A4 레이아웃 및 템플릿 설정값"""

    # 폰트
    FONT_PATH: str = "/app/fonts/AppleGothic.ttf"
    FONT_NAME: str = "AppleGothic"
    FALLBACK_FONT: str = "kr"  # PyMuPDF built-in Korean font alias

    # 템플릿 GCS 경로
    BASE_TEMPLATE_GCS_PATH: str = (
        configs.PDF_BASE_TEMPLATE_PATH if configs.PDF_BASE_TEMPLATE_PATH else "templates/base_template.pdf"
    )
    COVER_TEMPLATE_GCS_PATH: str = "templates/cover_template.pdf"

    # 레이아웃 치수 (pt 단위)
    PAGE_WIDTH: float = 595
    PAGE_HEIGHT: float = 842

    # --- Cover Layout ---
    # A4 상단 100pt 아래
    COVER_TITLE_Y: float = 100
    COVER_TITLE_FONT_SIZE: float = 24
    # A4 상단 750pt 아래
    COVER_COMPANY_Y: float = 750
    COVER_COMPANY_FONT_SIZE: float = 20
    # 자간 (Wide letter spacing)
    COVER_CHAR_SPACE: float = 3.0

    # --- Content Layout ---
    # 헤더 (공종/제목)
    HEADER_TITLE_RECT = fitz.Rect(146, 122, 400, 142)

    # 좌측 라벨 컬럼 너비
    LABEL_COL_WIDTH: float = 61

    # 사진 배치 영역
    ROW_1_TOP: float = 148
    ROW_HEIGHT: float = 214
    PHOTO_HEIGHT: float = 213
    PHOTO_WIDTH: float = 373

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
        self.output_path = self.tmp_dir / f"result_{int(datetime.now().timestamp())}.pdf"
        self.font_path = Path(LAYOUT.FONT_PATH)
        self.font_name = LAYOUT.FONT_NAME

    def generate(
        self,
        export_info: dict,
        clusters: List[dict],
        base_template_path: Path,
        cover_template_path: Optional[Path] = None,
    ) -> str:
        """
        메인 생성 로직 (CPU Bound)
        """
        if not base_template_path or not base_template_path.exists():
            logger.error("Base template not found.")
            raise FileNotFoundError(f"Base template not found at {base_template_path}")

        out_doc = fitz.open()

        # 폰트 등록
        font_file = str(self.font_path) if self.font_path.exists() else None
        font_alias = LAYOUT.FALLBACK_FONT
        if font_file:
            try:
                # 폰트를 전역적으로 등록하지 않고 페이지마다 필요시 참조하거나
                # 문서를 열 때 등록할 수도 있음. 여기서는 각 사용처에서 지정.
                # 편의를 위해 첫 페이지 생성 전 등록 시도 (문서 레벨 폰트)
                pass
            except Exception as e:
                logger.warning(f"Font setup warning: {e}")

        # 1. Cover Page
        if cover_template_path and cover_template_path.exists():
            try:
                self._add_cover_page(out_doc, cover_template_path, export_info, font_file)
            except Exception as e:
                logger.error(f"Failed to add cover page: {e}")
        else:
            logger.warning("Cover template path invalid or not provided. Skipping cover.")

        # 2. Content Pages
        src_doc = fitz.open(base_template_path)
        try:
            self._add_content_pages(out_doc, src_doc, clusters, export_info, font_file)
        finally:
            src_doc.close()

        out_doc.save(self.output_path)
        out_doc.close()

        return str(self.output_path)

    def _add_cover_page(self, out_doc, cover_template_path, export_info, font_file):
        """커버 페이지 추가 및 텍스트 작성"""
        with fitz.open(cover_template_path) as cover_doc:
            out_doc.insert_pdf(cover_doc, from_page=0, to_page=0)

        page = out_doc[0]  # 방금 추가된 커버 페이지

        title = export_info.get("cover_title") or ""
        company = export_info.get("cover_company_name") or ""

        font_alias = LAYOUT.FALLBACK_FONT

        # --- Draw Title ---
        if title:
            # 중앙 정렬을 위해 너비는 페이지 전체, 높이는 적당히 설정
            rect = fitz.Rect(
                0,
                LAYOUT.COVER_TITLE_Y,
                LAYOUT.PAGE_WIDTH,
                LAYOUT.COVER_TITLE_Y + LAYOUT.COVER_TITLE_FONT_SIZE * 2,
            )
            page.insert_textbox(
                rect,
                title,
                fontname=font_alias,
                fontfile=font_file,
                fontsize=LAYOUT.COVER_TITLE_FONT_SIZE,
                align=fitz.TEXT_ALIGN_CENTER,
            )

        # --- Draw Company Name ---
        if company:
            rect = fitz.Rect(
                0,
                LAYOUT.COVER_COMPANY_Y,
                LAYOUT.PAGE_WIDTH,
                LAYOUT.COVER_COMPANY_Y + LAYOUT.COVER_COMPANY_FONT_SIZE * 2,
            )
            page.insert_textbox(
                rect,
                company,
                fontname=font_alias,
                fontfile=font_file,
                fontsize=LAYOUT.COVER_COMPANY_FONT_SIZE,
                align=fitz.TEXT_ALIGN_CENTER,
            )

    def _add_content_pages(self, out_doc, src_doc, clusters, export_info, font_file):
        """클러스터별 컨텐츠 페이지 추가"""
        label_config = export_info.get("label_config", {})
        visible_keys = label_config.get("visible_keys", [])
        global_overrides = label_config.get("overrides", {})

        font_alias = LAYOUT.FALLBACK_FONT
        if font_file:
            # 문서에 폰트 리소스 추가 (별칭 부여)
            try:
                # 이미 추가되었을 수 있으므로 try/except 또는 확인 로직 필요
                # 단순화를 위해 페이지마다 insert_textbox시 fontfile 인자 사용
                pass
            except:
                pass

        for cluster in clusters:
            # 템플릿 페이지 복사
            out_doc.insert_pdf(src_doc, from_page=0, to_page=0)
            page = out_doc[-1]

            # 헤더 제목
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

                # 중앙 정렬 (Label Column 제외한 나머지 영역의 중앙)
                remaining_width = LAYOUT.PAGE_WIDTH - LAYOUT.LABEL_COL_WIDTH
                img_x = LAYOUT.LABEL_COL_WIDTH + (remaining_width - img_w) / 2
                img_y = row_center_y - (img_h / 2)

                img_rect = fitz.Rect(img_x, img_y, img_x + img_w, img_y + img_h)

                try:
                    page.insert_image(img_rect, filename=str(img_path))
                except Exception as e:
                    logger.error(f"Image insert failed: {e}")
                    continue

                # 라벨 결정
                final_labels = self._resolve_labels(photo_data, export_info, visible_keys, global_overrides)

                if final_labels:
                    self._draw_caption(page, img_rect, final_labels, font_alias, font_file)

    def _resolve_labels(self, photo_data, export_info, visible_keys, global_overrides):
        """사진별 라벨 데이터 결정"""
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
        return final_labels

    def _draw_caption(self, page, img_rect, labels, font_alias, font_file):
        """사진 위에 설명 박스 그리기"""
        label_lines = len(labels)
        # 캡션 높이 자동 조정 (줄 수에 비례)
        # 기본 26pt가 몇 줄 기준인지 불명확하나, 기존 로직 유지
        # 기존: CAPTION_HEIGHT * (label_lines / 2) -> 2줄이면 1배, 4줄이면 2배?
        # 안전하게 줄당 높이를 주는게 나을 수 있음. 여기선 기존 로직 존중하되 최소값 보장
        base_h = LAYOUT.CAPTION_HEIGHT
        calculated_h = base_h * (max(1, label_lines) / 2)

        cap_w = LAYOUT.CAPTION_WIDTH

        cap_x0 = img_rect.x0 - 2
        cap_y0 = img_rect.y0
        cap_x1 = cap_x0 + cap_w
        cap_y1 = cap_y0 + calculated_h

        cap_rect = fitz.Rect(cap_x0, cap_y0, cap_x1, cap_y1)

        # 배경 및 테두리
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

        text = "\n".join(f"{k} : {v}" for k, v in labels.items())
        page.insert_textbox(
            text_rect, text, fontname=font_alias, fontfile=font_file, fontsize=6, align=fitz.TEXT_ALIGN_LEFT
        )


async def generate_pdf_for_session(export_job_id: str):
    """
    ExportJob 처리 메인 함수
    """
    pm = PerformanceMonitor()
    pm.start()

    async with AsyncSessionLocal() as session:
        # 1. Fetch Job Data
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

        export_job.status = ExportStatus.PROCESSING
        await session.commit()

        try:
            job = export_job.job
            user = job.user

            # Prepare Export Info
            label_config = export_job.labels or {}

            export_info = {
                "cover_title": export_job.cover_title,
                "cover_company_name": export_job.cover_company_name,
                "label_config": label_config,
            }

            # Fetch Clusters
            stmt_c = (
                select(Cluster)
                .where(Cluster.job_id == job.id)
                .where(Cluster.name != "reserve")
                .order_by(Cluster.order_index.asc())
            )
            result_c = await session.execute(stmt_c)
            clusters_db = result_c.scalars().all()

            # --- File Preparation (in Temp Dir) ---
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                storage_client = get_storage_client()

                # Download Templates
                base_template_path = tmpdir / "template_base.pdf"
                cover_template_path = tmpdir / "template_cover.pdf"

                await _download_file_safe(storage_client, LAYOUT.BASE_TEMPLATE_GCS_PATH, base_template_path)

                # 커버 템플릿은 필수가 아닐 수도 있으나, 여기선 시도함
                await _download_file_safe(storage_client, LAYOUT.COVER_TEMPLATE_GCS_PATH, cover_template_path)

                # Download Photos & Build Data Structure
                processed_clusters = []
                for cluster in clusters_db:
                    cluster_data = {"name": cluster.name or f"Cluster #{cluster.id}", "photos": []}

                    # Fetch Top 3 Photos
                    stmt_p = (
                        select(Photo)
                        .where(Photo.cluster_id == cluster.id, Photo.deleted_at.is_(None))
                        .order_by(Photo.order_index.asc())
                        .limit(3)
                    )
                    res_p = await session.execute(stmt_p)
                    photos_db = res_p.scalars().all()

                    for photo in photos_db:
                        local_path = tmpdir / f"p_{photo.id}_{Path(photo.url).name if photo.url else 'img'}"

                        success = False
                        if configs.STORAGE_TYPE in ["gcs", "s3"] and photo.url:
                            success = await _download_file_safe(storage_client, photo.url, local_path)
                        else:
                            # Local storage fallback
                            src_path = Path(configs.MEDIA_ROOT) / photo.storage_path
                            if src_path.exists():
                                import shutil

                                shutil.copy(src_path, local_path)
                                success = True

                        if success:
                            cluster_data["photos"].append(
                                {
                                    "local_path": local_path,
                                    "id": str(photo.id),
                                    "db_labels": photo.labels,
                                    "timestamp": photo.meta_timestamp,
                                }
                            )

                    processed_clusters.append(cluster_data)

                # --- PDF Generation ---
                pdf_gen = PDFGenerator(tmpdir_str)
                loop = asyncio.get_event_loop()

                def _run_gen():
                    return pdf_gen.generate(
                        export_info,
                        processed_clusters,
                        base_template_path,
                        cover_template_path if cover_template_path.exists() else None,
                    )

                generated_pdf_path = await loop.run_in_executor(None, _run_gen)

                if not generated_pdf_path:
                    raise Exception("PDF Generation returned None")

                # --- Upload / Save Result ---
                file_name = f"{job.id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                final_url = await _save_result_file(storage_client, generated_pdf_path, user.user_id, job.id, file_name)

                export_job.status = ExportStatus.EXPORTED
                export_job.pdf_path = final_url
                export_job.finished_at = datetime.now()
                await session.commit()

                logger.info(f"PDF Generated successfully: {final_url}")
                
                pm.stop()
                pm.report("GeneratePDF")

        except Exception as e:
            logger.exception("PDF Generation Failed")
            export_job.status = ExportStatus.FAILED
            export_job.error_message = str(e)
            export_job.finished_at = datetime.now()
            await session.commit()


async def _download_file_safe(client, remote_path, local_path):
    """Exception safe file download helper"""
    try:
        if configs.STORAGE_TYPE in ["gcs", "s3"]:
            await client.download_file(remote_path, local_path)
        else:
            # For local dev without GCS
            # If requesting templates/..., check local assets
            # This is a mock fallback for dev environment
            if "template" in str(remote_path):
                mock_src = Path("/app/assets") / Path(remote_path).name
                if mock_src.exists():
                    import shutil

                    shutil.copy(mock_src, local_path)
        return local_path.exists()
    except Exception as e:
        logger.warning(f"Download failed for {remote_path}: {e}")
        return False


async def _save_result_file(client, local_path, user_id, job_id, file_name):
    if configs.STORAGE_TYPE in ["gcs", "s3"]:
        storage_path = f"{user_id}/{job_id}/exports/{file_name}"
        async with aiofiles.open(local_path, "rb") as f:
            await client.save_file(f, storage_path, content_type="application/pdf")
        return client.get_url(storage_path)
    else:
        target_dir = Path(configs.MEDIA_ROOT) / "exports"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        import shutil

        shutil.copy(local_path, target_path)
        return f"{configs.MEDIA_URL}/exports/{file_name}"
