import logging
import math
from typing import List

from scipy import stats

from app.clusterers.base_clusterer import Clusterer
from app.config import AppConfig
from app.photometa import PhotoMeta

logger = logging.getLogger(__name__)


class LocationClusterer(Clusterer):
    def __init__(self, config: AppConfig):
        self.config = config
        self.max_dist_m = self.config.MAX_LOCATION_DIST_M
        self.max_alt_diff_m = self.config.MAX_LOCATION_DIST_M

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        return await self.cluster_loop(photos, is_main=True)
    
    def get_stats(self, clusters):
        len_list = [len(c) for c in clusters if len(c) < 8]
        mean = sum(len_list) / len(len_list)
        mode = stats.mode([len(c) for c in clusters])[0]
        return mode, mean
        
    async def cluster_loop(self, photos: List[PhotoMeta], is_main=False) -> List[List[PhotoMeta]]:
        while True:
            clusters = await self.execute(photos)
            mode, mean = self.get_stats(clusters)
            logger.info(f"Iteration: photos: {len(photos)},mode={mode}, mean={mean}, max_dist_m={self.max_dist_m}, total clusters={len(clusters)}")
            
            if 3 <= mean <= 4 or self.max_dist_m <= 3 or self.max_dist_m >= self.config.MAX_LOCATION_DIST_M + 40.0:
                if is_main:
                    return await self.process_outlier(clusters)
                else:
                    return clusters
            if mean < 3:
                self.max_dist_m = self.max_dist_m + 2
            if mean > 4:
                self.max_dist_m = self.max_dist_m - 2

    async def execute(self, 
                      photos: List[PhotoMeta],
                      ) -> List[List[PhotoMeta]]:
        """Clusters photos based on their GPS location."""
        clusters: List[List[PhotoMeta]] = []

        # Filter out photos without GPS data first
        photos_with_gps = [p for p in photos if p.lat is not None and p.lon is not None]

        for photo in photos_with_gps:
            assigned = False
            for cluster in clusters:
                center = cluster[0]
                
                dist = self._haversine_distance_m(photo.lat, photo.lon, center.lat, center.lon)
                
                alt_diff = 0.0
                if photo.alt is not None and center.alt is not None:
                    alt_diff = abs(photo.alt - center.alt)

                if dist <= self.max_dist_m and alt_diff <= self.max_alt_diff_m:
                    cluster.append(photo)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([photo])
        
        # Include photos without GPS data as their own individual clusters
        photos_without_gps = [p for p in photos if p.lat is None or p.lon is None]
        for photo in photos_without_gps:
            clusters.append([photo])
        
        return clusters

    def _haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    async def process_outlier(self, clusters: List[List[PhotoMeta]]) -> List[List[PhotoMeta]]:
        final_clusters = []
        single_photo_cluster = []
        max_dist_m = self.max_dist_m

        for cluster in clusters:
            if len(cluster) < 3:
                single_photo_cluster.extend(cluster)
            elif len(cluster) > 10:
                sub_clusters = await self.split_large_clusters(cluster, max_dist_m=max_dist_m)
                for sub_cluster in sub_clusters:
                    if len(sub_cluster) == 1:
                        single_photo_cluster.append(sub_cluster[0])
                    else:
                        final_clusters.append(sub_cluster)
            else:
                final_clusters.append(cluster)
        
        if single_photo_cluster:
            sub_clusters = await self.gather_single_clusters(single_photo_cluster, max_dist_m=max_dist_m)
            single_photo_cluster = []
            for sub_cluster in sub_clusters:
                if len(sub_cluster) == 1:
                    single_photo_cluster.append(sub_cluster[0])
            final_clusters.append(single_photo_cluster)
        return final_clusters
    
    async def split_large_clusters(self, cluster: List[PhotoMeta], max_dist_m: float = None) -> List[List[PhotoMeta]]:
        if len(cluster) > 10:
            self.max_dist_m = max_dist_m - max_dist_m / 4.0
        return await self.cluster_loop(cluster)

    async def gather_single_clusters(self, cluster: List[PhotoMeta], max_dist_m: float = None) -> List[PhotoMeta]:
        self.max_dist_m = max_dist_m + max_dist_m / 5.0
        return await self.cluster_loop(cluster)