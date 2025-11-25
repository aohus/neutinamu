import asyncio
import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import List

import aiofiles
import aiofiles.os
import aioshutil
from PIL import Image, ImageOps

from app.config import AppConfig
from app.photometa import PhotoMeta

logger = logging.getLogger(__name__)


class OutputGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.output_root = Path(self.config.IMG_OUTPUT_DIR)
        self.meta_root = Path(self.config.META_OUTPUT_DIR)

    async def copy_and_rename_group(self, group: List[PhotoMeta], scene_idx: int) -> Path:
        sorted_group = sorted(
            group,
            key=lambda p: (p.timestamp is None, p.timestamp if p.timestamp is not None else 0.0, p.path)
        )

        scene_dir = self.output_root / f"scene_{scene_idx:03d}"
        await aiofiles.os.makedirs(scene_dir, exist_ok=True)

        copy_tasks = []
        for i, photo in enumerate(sorted_group, start=1):
            src = Path(photo.path)
            ext = src.suffix.lower()
            dst_name = f"img_{i:03d}{ext}"
            dst = scene_dir / dst_name
            copy_tasks.append(aioshutil.copy2(src, dst))
        
        await asyncio.gather(*copy_tasks)
        return scene_dir

    async def save_group_montage(self, scene_dir: Path):
        try:
            image_files = sorted(
                [p for p in await aiofiles.os.scandir(scene_dir) if p.is_file() and p.name.lower().endswith((".jpg", ".jpeg", ".png"))]
            )
        except FileNotFoundError:
            logger.warning(f"Scene directory not found for montage: {scene_dir}")
            return

        if not image_files:
            return

        thumbs = []
        for f in image_files:
            try:
                # Note: Pillow's image processing is synchronous and will block the event loop.
                thumb = self._create_thumbnail(Path(f.path))
                thumbs.append(thumb)
            except Exception as e:
                logger.warning(f"Failed to create thumbnail for {f.path}: {e}")

        if not thumbs:
            return

        # Note: Pillow's image creation is synchronous and will block the event loop.
        montage = self._create_montage_image(thumbs)
        montage_path = scene_dir / "scene_montage.jpg"
        
        buffer = io.BytesIO()
        montage.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        async with aiofiles.open(montage_path, 'wb') as f:
            await f.write(buffer.read())

    def _create_thumbnail(self, file_path: Path) -> Image.Image:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail(self.config.THUMB_SIZE)
        return img

    def _create_montage_image(self, thumbs: List[Image.Image]) -> Image.Image:
        cols = self.config.MONTAGE_COLS
        rows = math.ceil(len(thumbs) / cols)
        thumb_w, thumb_h = self.config.THUMB_SIZE
        montage_w = cols * thumb_w
        montage_h = rows * thumb_h

        montage = Image.new("RGB", (montage_w, montage_h), color=(240, 240, 240))

        for idx, thumb in enumerate(thumbs):
            row, col = idx // cols, idx % cols
            x = col * thumb_w + (thumb_w - thumb.width) // 2
            y = row * thumb_h + (thumb_h - thumb.height) // 2
            montage.paste(thumb, (x, y))
        return montage

    async def create_mosaic(self, scene_groups: List[List[PhotoMeta]]):
        if not scene_groups:
            logger.info("No clusters to visualize in mosaic.")
            return
        
        # Note: This image creation part is synchronous and will block the event loop.
        mosaic = self._create_mosaic_image(scene_groups)
        
        mosaic_path = Path(self.config.IMG_OUTPUT_DIR) / "clusters_mosaic.jpg"
        
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        async with aiofiles.open(mosaic_path, 'wb') as f:
            await f.write(buffer.read())

        logger.info(f"Mosaic saved to: {mosaic_path}")

    def _create_mosaic_image(self, scene_groups: List[List[PhotoMeta]]) -> Image.Image:
        rows_paths = [
            [p.path for p in sorted(g, key=lambda p: (p.timestamp is None, p.timestamp if p.timestamp is not None else 0.0, p.path))[:self.config.MONTAGE_COLS]]
            for g in scene_groups
        ]
        
        num_rows = len(rows_paths)
        num_cols = max(len(row) for row in rows_paths) if rows_paths else 0
        if num_cols == 0:
            return Image.new("RGB", (1, 1))

        thumb_w, thumb_h = self.config.THUMB_SIZE
        mosaic = Image.new("RGB", (num_cols * thumb_w, num_rows * thumb_h), (255, 255, 255))

        for row_idx, paths in enumerate(rows_paths):
            for col_idx, img_path in enumerate(paths):
                try:
                    thumb = self._create_thumbnail(Path(img_path))
                    thumb_bg = Image.new("RGB", self.config.THUMB_SIZE, (240, 240, 240))
                    x = (self.config.THUMB_SIZE[0] - thumb.size[0]) // 2
                    y = (self.config.THUMB_SIZE[1] - thumb.size[1]) // 2
                    thumb_bg.paste(thumb, (x, y))
                except Exception as e:
                    logger.warning(f"Failed to open {img_path} for mosaic: {e}")
                    thumb_bg = Image.new("RGB", self.config.THUMB_SIZE, (200, 200, 200))
                
                mosaic.paste(thumb_bg, (col_idx * thumb_w, row_idx * thumb_h))
        return mosaic

    async def save_meta(self, scene_groups: List[List[PhotoMeta]]):
        meta_result = {"meta": {}, "scenes": {}}
        scene_info = {}
        for idx, scene_group in enumerate(scene_groups, start=1):
            loc = f"s_{idx:03d}"
            scene_info[loc] = len(scene_group)
            scene_details = {
                os.path.basename(scene.path): {
                    "lat": scene.lat, "lon": scene.lon, "alt": scene.alt, "timestamp": scene.timestamp
                } for scene in scene_group
            }
            meta_result["scenes"][loc] = scene_details
        meta_result["meta"] = scene_info

        await aiofiles.os.makedirs(self.meta_root, exist_ok=True)
        filepath = self.meta_root / f"meta_info_{time.time()}.json"
        
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(json.dumps(meta_result, indent=4, ensure_ascii=False))
        logger.info(f"Saved metadata for {len(scene_groups)} scenes to {filepath}")