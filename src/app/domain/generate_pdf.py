import glob
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import aiofiles
from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob, Job
from app.models.photo import Photo
from app.schemas.enum import ExportStatus  # 상태 Enum 이라고 가정
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# --------- 폰트 등록 ---------
font_dir = Path("/app/fonts")

logger = logging.getLogger(__name__)

pdfmetrics.registerFont(
    TTFont("NanumGothic", str(font_dir / "AppleGothic.ttf"))
)
pdfmetrics.registerFont(
    TTFont("NanumGothic-Bold", str(font_dir / "AppleGothic.ttf"))
)

def _draw_vertical_text(c: canvas.Canvas, x: float, y_bottom: float, y_top: float, text: str, font="NanumGothic-Bold", font_size=12):
    """
    세로(90도 회전) 텍스트 그리기 유틸.
    x : 세로 글자의 기준 x (원래 좌표계)
    y_bottom, y_top : 세로 셀의 아래/위
    """
    c.saveState()
    c.setFont(font, font_size)
    mid_y = (y_bottom + y_top) / 2.0
    # 기준점으로 이동 후 회전
    c.translate(x, mid_y)
    # c.rotate(90)
    c.drawCentredString(0, 0, text)
    c.restoreState()


async def generate_pdf_for_session(export_job_id: str):
    """
    ExportJob 을 읽어서,
    각 Cluster 를 1페이지로 하는 "전/중/후 사진 보고서" PDF 생성.
    """
    async with AsyncSessionLocal() as session:
        export_job = None
        try:
            # --------- ExportJob 조회 및 상태 변경 ---------
            result = await session.execute(
                select(ExportJob)
                .options(
                    selectinload(ExportJob.job).selectinload(Job.user),
                    selectinload(ExportJob.job).selectinload(Job.photos),
                )
                .where(ExportJob.id == export_job_id)
            )
            export_job = result.scalars().first()
            if not export_job:
                return

            export_job.status = ExportStatus.PROCESSING
            await session.commit()

            job_id = export_job.job_id
            if not job_id:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = "Job not found in export job."
                await session.commit()
                return

            if not export_job.job:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = "Associated Job not found."
                await session.commit()
                return

            job = export_job.job
            user = job.user

            # Determine contractor
            contractor = job.contractor_name
            if not contractor and user:
                contractor = user.company_name
            
            if not contractor:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = "Contractor name not provided and user company name is missing."
                await session.commit()
                return

            # Determine work_date
            work_date_final = job.work_date
            if not work_date_final and job.photos:
                # Get the earliest meta_timestamp from photos
                photo_timestamps = [p.meta_timestamp for p in job.photos if p.meta_timestamp]
                if photo_timestamps:
                    work_date_final = min(photo_timestamps)
                else:
                    work_date_final = datetime.now() # Fallback if no meta_timestamp found in photos
            elif not work_date_final:
                work_date_final = datetime.now() # Fallback if no work_date and no photos

            date_str = work_date_final.strftime("%Y.%m.%d")


            # --------- PDF 파일 준비 ---------
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdf_path = Path(tmpdir) / f"job_{export_job.id}_{int(datetime.now().timestamp())}.pdf"
                c = canvas.Canvas(str(tmp_pdf_path), pagesize=A4)
                width, height = A4

                # 레이아웃 기본 값
                margin_x = 30
                margin_top = 30
                margin_bottom = 30

                table_left = margin_x
                table_right = width - margin_x
                table_top = height - margin_top
                table_bottom = margin_bottom

                # 새 레이아웃: 2열 (라벨 / 이미지)
                # 라벨 컬럼 너비
                label_col_w = 100 # "작업" + "공종" 너비 합친 개념
                header_h = 60      # 상단 제목 행 높이

                content_height = table_top - table_bottom - header_h
                row_h = content_height / 3.0  # 전/중/후 3행

                # --------- Cluster 목록 조회 ---------
                result = await session.execute(
                    select(Cluster)
                    .where(Cluster.job_id == job_id)
                    .order_by(Cluster.order_index.asc())
                )
                clusters = result.scalars().all()

                for cluster in clusters:
                    # 페이지마다 동일한 프레임/표 구조

                    # 외곽 테두리
                    c.rect(table_left, table_bottom,
                           table_right - table_left,
                           table_top - table_bottom)

                    # 세로 선 (라벨 열과 이미지 열 구분)
                    x_col_split = table_left + label_col_w
                    c.line(x_col_split, table_bottom, x_col_split, table_top)

                    # 가로 선 (header, 전/중/후 구분)
                    y_header_bottom = table_top - header_h
                    c.line(table_left, y_header_bottom, table_right, y_header_bottom)  # header 아래

                    y_row1_bottom = y_header_bottom - row_h
                    y_row2_bottom = y_row1_bottom - row_h
                    
                    c.line(table_left, y_row1_bottom, x_col_split, y_row1_bottom) # 라벨 열의 전/중 구분
                    c.line(table_left, y_row2_bottom, x_col_split, y_row2_bottom) # 라벨 열의 중/후 구분

                    # --------- "작업 \ 공종" 대각선 텍스트 ---------
                    # 대각선 그리기
                    c.line(table_left, y_header_bottom, x_col_split, table_top)

                    # "작업" 텍스트 (아래 왼쪽)
                    c.setFont("NanumGothic", 10)
                    c.drawCentredString(table_left + label_col_w / 4, y_header_bottom + header_h / 4, "작업")
                    
                    # "공종" 텍스트 (위 오른쪽)
                    c.drawCentredString(table_left + label_col_w * 3 / 4, table_top - header_h / 4, "공종")

                    # --------- "전/중/후" 라벨 ---------
                    row_labels = ["전", "중", "후"]
                    row_bottoms = [y_row1_bottom, y_row2_bottom, table_bottom]
                    row_tops = [y_header_bottom, y_row1_bottom, y_row2_bottom]
                    for label, yb, yt in zip(row_labels, row_bottoms, row_tops):
                        _draw_vertical_text( # 기존 함수 재활용. 세로 텍스트가 아니므로 변경 필요
                            c,
                            x=table_left + label_col_w / 2.0,
                            y_bottom=yb,
                            y_top=yt,
                            text=label,
                            font_size=14,
                        )

                    # --------- 상단 제목(공종명 + 차수 등) ---------
                    c.setFont("NanumGothic-Bold", 14)
                    title = cluster.name or f"Cluster #{cluster.id}"
                    c.drawString(
                        x_col_split + 12,
                        table_top - header_h / 2.0,
                        title,
                    )

                    # --------- 사진들 조회 (deleted_at 이 없는 것만, 순서대로) ---------
                    result_p = await session.execute(
                        select(Photo)
                        .where(
                            Photo.cluster_id == cluster.id,
                            Photo.deleted_at.is_(None),
                        )
                        .order_by(Photo.order_index.asc())
                    )
                    photos = result_p.scalars().all()

                    # GCS/S3 스토리지 클라이언트 초기화 (필요시)
                    storage_client_for_photos = None
                    if settings.STORAGE_TYPE in ["gcs", "s3"]:
                        storage_client_for_photos = get_storage_client()

                    # 전/중/후 3장 기준으로 배치 (사진이 더 적어도 상관없음)
                    image_x = x_col_split + 5
                    image_w = table_right - image_x - 5

                    for idx in range(3):
                        if idx >= len(photos):
                            break  # 사진이 부족하면 그 행은 비워둠

                        photo = photos[idx]

                        image_path = None
                        if storage_client_for_photos: # GCS/S3의 경우 photo.url에서 이미지 다운로드
                            try:
                                if photo.url:
                                    # Construct a unique temporary file path for the image
                                    # Use a hash or a similar unique identifier to avoid name clashes
                                    # And preserve the original extension if possible
                                    image_filename = Path(photo.url).name
                                    temp_image_path = Path(tmpdir) / image_filename

                                    # Download the file
                                    await storage_client_for_photos.download_file(photo.url, temp_image_path)
                                    image_path = temp_image_path
                                else:
                                    logger.warning(f"Photo {photo.id} has no URL for GCS/S3 storage.")
                            except Exception as download_e:
                                logger.error(f"Failed to download image {photo.id} from {photo.url}: {download_e}")
                        
                        if not image_path: # Fallback for local or if cloud download failed
                            # 실제 이미지 파일 경로 가져오기
                            #    없으면 original_filename 기반으로 /app/assets/uploads 아래에서 찾게 수정
                            image_path = Path("/app/assets") / Path(photo.storage_path)  # 실제 필드명에 맞게 수정
                        
                        # 이 행의 top / bottom
                        row_top = row_tops[idx]
                        row_bottom = row_bottoms[idx]

                        image_y = row_bottom + 5
                        image_h = row_top - image_y - 5

                        try:
                            # Ensure image_path is a string for c.drawImage
                            if image_path:
                                c.drawImage(
                                    str(image_path),
                                    image_x,
                                    image_y,
                                    width=image_w,
                                    height=image_h,
                                    preserveAspectRatio=True,
                                    anchor="sw",
                                )
                            else:
                                c.setFont("NanumGothic", 10)
                                c.drawString(image_x, row_top - 20, f"이미지 로드 실패: 경로 없음")
                        except Exception as e:
                            logger.error(f"Failed to Draw Image: {e}")
                            raise e
                        
                        # --------- 사진 위에 "일자 / 시행처" 박스 ---------
                        label_w = 160
                        label_h = 40
                        c.setFillColorRGB(1, 1, 1)
                        c.rect(
                            image_x,
                            image_y + image_h - label_h,
                            label_w,
                            label_h,
                            fill=1,
                            stroke=1,
                        )
                        c.setFillColorRGB(0, 0, 0)
                        c.setFont("NanumGothic", 9)
                        c.drawString(
                            image_x + 6,
                            image_y + image_h - 15,
                            f"일자 : {date_str}",
                        )
                        c.drawString(
                            image_x + 6,
                            image_y + image_h - 30,
                            f"시행처 : {contractor}",
                        )
                    # 페이지 종료
                    c.showPage()
                c.save()

                final_pdf_path = None
                storage_client = get_storage_client()
                file_name = tmp_pdf_path.name
                
                if settings.STORAGE_TYPE == "local":
                    # For local storage, save to a persistent path within MEDIA_ROOT
                    local_storage_dir = Path(settings.MEDIA_ROOT) / "exports"
                    local_storage_dir.mkdir(parents=True, exist_ok=True)
                    persistent_pdf_path = local_storage_dir / file_name
                    
                    # Copy the generated PDF from temp to persistent storage
                    import shutil
                    shutil.copy(tmp_pdf_path, persistent_pdf_path)
                    
                    final_pdf_path = str(persistent_pdf_path)
                    # For local storage, we also need a URL to serve it
                    final_pdf_path = f"{settings.MEDIA_URL}/exports/{file_name}"

                elif settings.STORAGE_TYPE in ["gcs", "s3"]:
                    try:
                        # 저장 경로: {user_id}/{job_id}/exports/{filename}
                        storage_path = f"{user.user_id}/{job_id}/exports/{file_name}"
                        
                        async with aiofiles.open(tmp_pdf_path, 'rb') as f:
                            await storage_client.save_file(f, storage_path, content_type="application/pdf")
                        
                        # 업로드 후 URL 또는 경로로 업데이트
                        final_pdf_path = storage_client.get_url(storage_path)

                        # 로컬 임시 파일 삭제는 tempfile.TemporaryDirectory가 처리
                    except Exception as e:
                        logger.error(f"Failed to upload PDF to storage: {e}")
                        raise e
                
                # If for some reason final_pdf_path is still None (e.g., storage type not handled),
                # it might be an error or a case to default.
                # For now, if local_storage, we construct URL. If cloud, we get URL.
                # If neither is set, final_pdf_path might be None.
                if final_pdf_path is None:
                    logger.error(f"Failed to determine final PDF path for storage type: {settings.STORAGE_TYPE}")
                    export_job.status = ExportStatus.FAILED
                    export_job.error_message = "Failed to determine final PDF storage path."
                    await session.commit()
                    return

            export_job.status = ExportStatus.EXPORTED
            export_job.pdf_path = final_pdf_path
            export_job.finished_at = datetime.now()
            await session.commit()

        except Exception as e:
            # 실패 시 상태 업데이트
            if export_job is None:
                result = await session.execute(
                    select(ExportJob).where(ExportJob.id == export_job_id)
                )
                export_job = result.scalars().first()

            if export_job:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = str(e)
                export_job.finished_at = datetime.now()
                await session.commit()
        finally:
            await session.close()