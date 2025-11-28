import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from app.core.config import SessionConfig
from app.core.logger import get_logger
from app.models.photometa import PhotoMeta
from app.services.clusterers.base_clusterer import Clusterer
from app.services.clusterers.camera_settings_clusterer import CameraSettingsClusterer
from app.services.clusterers.deep_clusterer import DeepClusterer
from app.services.clusterers.ensemble_clusterer import EnsembleClusterer
from app.services.clusterers.image_clusterer import ImageClusterer
from app.services.clusterers.image_loc_clusterer import ImageLocClusterer
from app.services.clusterers.location_clusterer import LocationClusterer
from app.services.clusterers.time_clusterer import TimeSplitClusterer
from app.services.metadata_extractor import MetadataExtractor
from app.services.output_generator import OutputGenerator
from app.services.runner import ClusterRunner

logger = get_logger(__name__)


class PhotoClusteringPipeline:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.metadata_extractor = MetadataExtractor()
        self.output_generator = OutputGenerator(config)

        # Initialize clusterers
        clusterers = self._create_clusterers(config)
        self.clusterer = ClusterRunner(clusterers)

    def _create_clusterers(self, config: SessionConfig) -> List[Clusterer]:
        """Factory method to create clustering clusterers based on config."""
        clusterer_map: Dict[str, Clusterer] = {"location": LocationClusterer(config)}
        
        # deep_clusterer = DeepClusterer(
        #     input_path=config.IMAGE_DIR, similarity_threshold=0.13, use_cache=True
        # )

        # clusterer_map: Dict[str, Clusterer] = {
        #     "location": LocationClusterer(config),
        #     "time": TimeSplitClusterer(config),
        #     "camera_settings": CameraSettingsClusterer(),
        #     "image": ImageClusterer(
        #         deep_clusterer=deep_clusterer, executor=process_executor
        #     ),
        #     "image_loc": ImageLocClusterer(
        #         deep_clusterer=deep_clusterer,
        #         executor=process_executor,
        #         location_weight=0.5,
        #         direction_weight=0.2,
        #     ),
        #     "ensemble": EnsembleClusterer(
        #         deep_clusterer=deep_clusterer,
        #     ),
        # }

        active_clusterers = []
        for name in config.CLUSTERERS:
            if name in clusterer_map:
                active_clusterers.append(clusterer_map[name])
            else:
                logger.warning(
                    f"Unknown clustering clusterer '{name}' in config. Ignoring."
                )
        return active_clusterers

    def _list_image_paths(self) -> List[str]:
        image_dir = self.config.IMAGE_DIR
        if not image_dir.exists():
            logger.error(f"Image directory does not exist: {image_dir}")
            return []
        return [
            str(image_dir / fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(("png", "jpg", "jpeg"))
        ]

    async def run(self):
        start_time = time.time()

        self.config.setup_output_dirs(clean=True)

        image_paths = self._list_image_paths()
        logger.info(f"Found {len(image_paths)} images.")

        logger.info("Extracting metadata from images asynchronously...")
        # Run metadata extraction in parallel
        tasks = [self.metadata_extractor.extract(p) for p in image_paths]
        photos = await asyncio.gather(*tasks)
        logger.info(f"Metadata extracted for {len(photos)} photos.")

        logger.info("Starting clustering pipeline...")
        final_scenes = await self.clusterer.process(photos)
        logger.info(
            f"Clustering pipeline finished. Final scene count: {len(final_scenes)}"
        )

        logger.info("Generating outputs asynchronously...")
        await self.output_generator.save_meta(final_scenes)

        output_tasks = [
            self.output_generator.copy_and_rename_group(group, idx)
            for idx, group in enumerate(final_scenes, start=1)
        ]
        scene_dirs = await asyncio.gather(*output_tasks)

        await self.output_generator.create_mosaic(final_scenes)

        # Generate and save the cluster visualization
        self.output_generator.save_cluster_visualization_sync(final_scenes)

        end_time = time.time()
        logger.info(f"Done. Total time: {end_time - start_time:.2f} seconds.")


# class ClustererFactory:
