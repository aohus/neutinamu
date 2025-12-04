import asyncio
import logging
import time
from typing import Dict, List

from app.core.config import JobConfig
from app.domain.clusterers.base import Clusterer
from app.domain.clusterers.gps import GPSCluster
from app.domain.metadata_extractor import MetadataExtractor
from app.domain.output_generator import OutputGenerator
from app.domain.storage.local import LocalStorageService

# if TYPE_CHECKING:
#     from app.domain.storage.base import StorageService
from app.domain.storage.base import StorageService
from app.models.photo import Photo
from app.models.photometa import PhotoMeta

logger = logging.getLogger(__name__)


class ClusterRunner:
    def __init__(self, clusterers: List[Clusterer]):
        self.clusterers = clusterers

    async def process(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Processes photos by applying a series of clustering clusterers in sequence.
        """
        # Start with a single cluster containing all photos
        clusters = [photos]
        
        for clusterer in self.clusterers:
            logger.info(f"Applying clusterer: {clusterer.__class__.__name__}")
            
            new_clusters = []
            # Apply the clusterer to each existing cluster
            for cluster in clusters:
                if not cluster:
                    continue
                if clusterer.condition(cluster):
                    sub_clusters = await clusterer.cluster(cluster)
                    new_clusters.extend(sub_clusters)
                else:
                    new_clusters.append(cluster)
            
            clusters = new_clusters
            logger.info(f"Resulted in {len(clusters)} clusters.")
        return clusters


class PhotoClusteringPipeline:
    def __init__(self, config: JobConfig, storage: StorageService, photos: list[Photo]):
        self.config = config
        self.storage = storage
        self.photos = photos
        logger.debug(f"Initializing pipeline for job_id: {self.config.job_id}")

        self.metadata_extractor = MetadataExtractor()

        # Initialize clusterers
        clusterers = self._create_clusterers(self.config)
        self.clusterer = ClusterRunner(clusterers)
        logger.debug(f"Pipeline initialized with {len(clusterers)} clusterers.")

    def _create_clusterers(self, config: "JobConfig") -> List[Clusterer]:
        """Factory method to create clustering clusterers based on config."""
        logger.debug("Creating clusterers...")
        return [GPSCluster()]
    
    async def run(self):
        logger.info(f"Pipeline run started for job {self.config.job_id}.")
        
        logger.info("Resolving photo paths and extracting metadata...")
        tasks = []
        for p in self.photos:
            if isinstance(self.storage, LocalStorageService):
                full_path = str(self.storage.media_root / p.storage_path)
            else:
                full_path = self.storage.get_url(p.storage_path)
            tasks.append(self.metadata_extractor.extract(full_path))
            
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