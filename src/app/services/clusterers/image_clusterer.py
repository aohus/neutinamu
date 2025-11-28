import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List

from app.models.photometa import PhotoMeta
from app.services.clusterers.base_clusterer import Clusterer
from app.services.clusterers.deep_clusterer import DeepClusterer


class ImageClusterer(Clusterer):
    def __init__(self, deep_clusterer: DeepClusterer, executor: ProcessPoolExecutor):
        self.deep_clusterer = deep_clusterer
        self.executor = executor

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Clusters photos based on image content using a deep learning model.
        The heavy computation is run in a separate process to avoid blocking.
        """
        if len(photos) <= 1:
            return [photos]

        loop = asyncio.get_running_loop()
        
        photo_paths = [p.path for p in photos]
        
        # Run the CPU/GPU-bound task in a process pool
        clustered_paths = await loop.run_in_executor(
            self.executor,
            self.deep_clusterer.cluster,
            photo_paths
        )

        # Create a map for quick lookup
        photo_map = {p.path: p for p in photos}

        # Reconstruct the clusters with PhotoMeta objects
        sub_clusters = []
        for path_group in clustered_paths:
            photo_group = [photo_map[path] for path in path_group if path in photo_map]
            if photo_group:
                sub_clusters.append(photo_group)
        return sub_clusters

    @staticmethod
    def condition(cluster: List[PhotoMeta]) -> bool:
        return len(cluster) > 5