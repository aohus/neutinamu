import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from app.core.config import SessionConfig
from app.domain.clusterers.base_clusterer import Clusterer
from app.domain.clusterers.camera_settings_clusterer import CameraSettingsClusterer
from app.domain.clusterers.deep_clusterer import DeepClusterer
from app.domain.clusterers.ensemble_clusterer import EnsembleClusterer
from app.domain.clusterers.image_clusterer import ImageClusterer
from app.domain.clusterers.image_loc_clusterer import ImageLocClusterer
from app.domain.clusterers.location_clusterer import LocationClusterer
from app.domain.clusterers.time_clusterer import TimeSplitClusterer
from app.domain.metadata_extractor import MetadataExtractor
from app.domain.output_generator import OutputGenerator
from app.domain.runner import ClusterRunner

# if TYPE_CHECKING:
#     from app.domain.storage.base import StorageService
from app.domain.storage.base import StorageService
from app.models.photometa import PhotoMeta

logger = logging.getLogger(__name__)


class PhotoClusteringPipeline:
    def __init__(self, storage: StorageService):
        self.storage = storage
        self.config = storage.config
        logger.debug(f"Initializing pipeline for job_id: {self.config.job_id}")

        self.metadata_extractor = MetadataExtractor()
        self.output_generator = OutputGenerator(self.config)

        # Initialize clusterers
        clusterers = self._create_clusterers(self.config)
        self.clusterer = ClusterRunner(clusterers)
        logger.debug(f"Pipeline initialized with {len(clusterers)} clusterers.")

    def _create_clusterers(self, config: "SessionConfig") -> List[Clusterer]:
        """Factory method to create clustering clusterers based on config."""
        logger.debug("Creating clusterers...")
        clusterer_map: Dict[str, Clusterer] = {"location": LocationClusterer()}
        
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
                logger.debug(f"Activating clusterer: {name}")
                active_clusterers.append(clusterer_map[name])
            else:
                logger.warning(
                    f"Unknown clustering clusterer '{name}' in config. Ignoring."
                )
        return active_clusterers


    async def run(self):
        start_time = time.time()
        logger.info(f"Pipeline run started for job {self.config.job_id}.")

        self.config.setup_output_dirs(clean=True)
        image_paths = self.storage.list_image_paths()
        logger.info(f"Found {len(image_paths)} images.")

        if not image_paths:
            logger.warning("No images found to process. Aborting pipeline.")
            return

        logger.info("Extracting metadata from images asynchronously...")
        tasks = [self.metadata_extractor.extract(p) for p in image_paths]
        photos = await asyncio.gather(*tasks)
        logger.info(f"Metadata extracted for {len(photos)} photos.")

        logger.info("Starting clustering pipeline...")
        final_scenes = await self.clusterer.process(photos)
        logger.info(
            f"Clustering pipeline finished. Final scene count: {len(final_scenes)}"
        )

        if not final_scenes:
            logger.warning("No scenes were generated. Skipping output generation.")
            return
        return final_scenes
        logger.info("Generating outputs asynchronously...")
        await self.output_generator.save_meta(final_scenes)

        output_tasks = [
            self.output_generator.copy_and_rename_group(group, idx)
            for idx, group in enumerate(final_scenes, start=1)
        ]
        scene_dirs = await asyncio.gather(*output_tasks)
        logger.info(f"Finished copying {len(scene_dirs)} scenes to output directories.")

        await self.output_generator.create_mosaic(final_scenes)

        # This is a synchronous operation
        self.output_generator.save_cluster_visualization_sync(final_scenes)

        end_time = time.time()
        logger.info(f"Pipeline for job {self.config.job_id} finished. Total time: {end_time - start_time:.2f} seconds.")


