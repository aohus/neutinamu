import glob
import logging
import os
from datetime import datetime
from pathlib import Path

import aiofiles
from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.domain.storage.factory import get_storage_client
from app.models.cluster import Cluster
from app.models.job import ExportJob
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
                .options(selectinload(ExportJob.job))
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
                export_job.error_message = "Job not found"
                await session.commit()
                return

            # user_id 가져오기 (ExportJob.job 관계 활용)
            # Job이 로드되지 않았을 경우를 대비한 방어 로직 (이미 selectinload 했으므로 job은 존재해야 함)
            if not export_job.job:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = "Associated Job not found"
                await session.commit()
                return
            
            user_id = str(export_job.job.user_id)

            # (선택) 일자/시행처는 Job 또는 ExportJob 에서 가져온다고 가정
            # 실제 필드명에 맞게 수정 필요
            work_date = getattr(export_job, "work_date", None)  # 예: date 필드
            if work_date is None:
                # 없으면 finished_at 또는 created_at 등으로 대체
                base_dt = getattr(export_job, "created_at", datetime.now())
                work_date = base_dt.date()
            date_str = work_date.strftime("%Y.%m.%d")

            contractor = getattr(export_job, "contractor_name", "미정")

            # --------- PDF 파일 준비 ---------
            pdf_path = Path("/app/assets") / f"job_{export_job.id}_{int(datetime.now().timestamp())}.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            width, height = A4

            # 레이아웃 기본 값
            margin_x = 30
            margin_top = 30
            margin_bottom = 30

            table_left = margin_x
            table_right = width - margin_x
            table_top = height - margin_top
            table_bottom = margin_bottom

            first_col_w = 40   # "작업" 세로
            second_col_w = 60  # "공종 / 전·중·후" 세로
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

                # 세로 선 (열 구분)
                x_col1 = table_left + first_col_w
                x_col2 = x_col1 + second_col_w
                c.line(x_col1, table_bottom, x_col1, table_top)
                c.line(x_col2, table_bottom, x_col2, table_top)

                # 가로 선 (header, 전/중/후 구분)
                y_header_bottom = table_top - header_h
                c.line(table_left, y_header_bottom, table_right, y_header_bottom)  # header 아래

                y_row1_bottom = y_header_bottom - row_h
                y_row2_bottom = y_row1_bottom - row_h
                # 전/중/후 행 구분은 두 번째 열부터
                c.line(x_col1, y_row1_bottom, table_right, y_row1_bottom)
                c.line(x_col1, y_row2_bottom, table_right, y_row2_bottom)

                # --------- 세로 텍스트(작업 / 공종 / 전·중·후) ---------
                # "작업" : 첫 번째 열 전체 중앙
                _draw_vertical_text(
                    c,
                    x=table_left + first_col_w / 2.0,
                    y_bottom=table_bottom,
                    y_top=table_top,
                    text="작업",
                    font_size=14,
                )

                # "공종" : 두 번째 열의 header 영역 중앙
                _draw_vertical_text(
                    c,
                    x=x_col1 + second_col_w / 2.0,
                    y_bottom=y_header_bottom,
                    y_top=table_top,
                    text="공종",
                    font_size=14,
                )

                # "전/중/후" : 두 번째 열의 각 행 중앙
                row_labels = ["전", "중", "후"]
                row_bottoms = [y_row1_bottom, y_row2_bottom, table_bottom]
                row_tops = [y_header_bottom, y_row1_bottom, y_row2_bottom]
                for label, yb, yt in zip(row_labels, row_bottoms, row_tops):
                    _draw_vertical_text(
                        c,
                        x=x_col1 + second_col_w / 2.0,
                        y_bottom=yb,
                        y_top=yt,
                        text=label,
                        font_size=14,
                    )

                # --------- 상단 제목(공종명 + 차수 등) ---------
                c.setFont("NanumGothic-Bold", 14)
                # 예시: Cluster.name 이 "초화류사이제초(대원지하차도)_3차" 형태라고 가정
                title = cluster.name or f"Cluster #{cluster.id}"
                c.drawString(
                    x_col2 + 10,
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

                # 전/중/후 3장 기준으로 배치 (사진이 더 적어도 상관없음)
                image_x = x_col2 + 5
                image_w = table_right - image_x - 5

                for idx in range(3):
                    if idx >= len(photos):
                        break  # 사진이 부족하면 그 행은 비워둠

                    photo = photos[idx]

                    # 실제 이미지 파일 경로 가져오기
                    #    없으면 original_filename 기반으로 /app/assets/uploads 아래에서 찾게 수정
                    image_path = f"/app/assets" / Path(photo.storage_path)  # 실제 필드명에 맞게 수정

                    # 이 행의 top / bottom
                    row_top = row_tops[idx]
                    row_bottom = row_bottoms[idx]

                    image_y = row_bottom + 5
                    image_h = row_top - image_y - 5

                    try:
                        c.drawImage(
                            str(image_path),
                            image_x,
                            image_y,
                            width=image_w,
                            height=image_h,
                            preserveAspectRatio=True,
                            anchor="sw",
                        )
                    except Exception:
                        # 이미지 로드 실패하면 파일명만 텍스트로 표시
                        c.setFont("NanumGothic", 10)
                        c.drawString(image_x, row_top - 20, f"이미지 로드 실패: {getattr(photo, 'original_filename', '')}")

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

            final_pdf_path = str(pdf_path)

            # GCS (또는 S3) 사용 시 업로드
            if settings.STORAGE_TYPE in ["gcs", "s3"]:
                try:
                    storage_client = get_storage_client()
                    # 저장 경로: {user_id}/{job_id}/exports/{filename}
                    file_name = pdf_path.name
                    storage_path = f"{user_id}/{job_id}/exports/{file_name}"
                    
                    async with aiofiles.open(pdf_path, 'rb') as f:
                        await storage_client.save_file(f, storage_path, content_type="application/pdf")
                    
                    # 업로드 후 URL 또는 경로로 업데이트
                    final_pdf_path = storage_client.get_url(storage_path)

                    # 로컬 임시 파일 삭제 (선택 사항, 클라우드 환경에서는 용량 관리를 위해 삭제 권장)
                    if pdf_path.exists():
                        os.remove(pdf_path)
                except Exception as e:
                    logger.error(f"Failed to upload PDF to storage: {e}")
                    # 업로드 실패 시에도 일단 로컬 경로는 남아있으나, 
                    # 운영 환경에 따라 로컬 파일 접근이 불가능할 수 있음.
                    # 여기서는 에러를 다시 던져서 Job을 Failed로 처리하거나, 
                    # 로컬 경로라도 반환할지 결정해야 함. 
                    # 현재 구조상 finally 블록이 없으므로 여기서 raise 하면 catch 블록으로 감.
                    raise e

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