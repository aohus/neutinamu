import asyncio
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.models.photometa import PhotoMeta
from app.services.clusterers.base_clusterer import Clusterer
from app.services.clusterers.deep_clusterer import DeepClusterer


class ImageLocClusterer(Clusterer):
    def __init__(
        self,
        deep_clusterer: DeepClusterer,
        executor: ProcessPoolExecutor,
        location_weight: float = 0.5,
        direction_weight: float = 0.2,
        max_location_dist_m: float = 25.0,
        max_direction_diff_degrees: float = 45.0,
        dbscan_eps: float = 0.5,
    ):
        self.deep_clusterer = deep_clusterer
        self.executor = executor
        
        image_weight = 1.0 - location_weight - direction_weight
        total_weight = location_weight + direction_weight + image_weight
        self.location_weight = location_weight / total_weight
        self.direction_weight = direction_weight / total_weight
        self.image_weight = image_weight / total_weight

        self.max_location_dist_m = max_location_dist_m
        self.max_direction_diff_degrees = max_direction_diff_degrees
        self.dbscan_eps = dbscan_eps

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        if len(photos) <= 1:
            return [photos]

        loop = asyncio.get_running_loop()
        clustered_indices = await loop.run_in_executor(
            self.executor, self._perform_clustering, photos
        )

        clusters = []
        for index_group in clustered_indices:
            photo_group = [photos[i] for i in index_group]
            if photo_group:
                clusters.append(photo_group)
        return clusters

    def _perform_clustering(self, photos: List[PhotoMeta]) -> List[List[int]]:
        features_list, valid_indices = self._extract_features(photos)

        if len(features_list) < 2:
            return [[i] for i in range(len(photos))]

        features_array = np.array(features_list)
        valid_photos = [photos[i] for i in valid_indices]

        d_image = self._get_image_distance_matrix(features_array)
        d_loc = self._get_location_distance_matrix(valid_photos)
        d_dir = self._get_direction_distance_matrix(valid_photos)

        d_combined = (
            self.location_weight * d_loc
            + self.direction_weight * d_dir
            + self.image_weight * d_image
        )

        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=2, metric="precomputed")
        cluster_labels = clustering.fit_predict(d_combined)

        return self._format_clusters(cluster_labels, valid_indices, len(photos))

    def _extract_features(self, photos: List[PhotoMeta]):
        features_list = []
        valid_indices = []
        for i, p in enumerate(photos):
            combined_features, _ = self.deep_clusterer.extract_deep_features(p.path)
            if combined_features is not None:
                features_list.append(combined_features)
                valid_indices.append(i)
        return features_list, valid_indices

    def _get_image_distance_matrix(self, features_array: np.ndarray) -> np.ndarray:
        features_normalized = normalize(features_array, norm="l2")
        image_similarity = cosine_similarity(features_normalized)
        d_image = np.clip(1 - image_similarity, 0, 2)
        return d_image / 2.0

    def _get_location_distance_matrix(self, photos: List[PhotoMeta]) -> np.ndarray:
        num_photos = len(photos)
        d_loc = np.zeros((num_photos, num_photos))
        for i in range(num_photos):
            for j in range(i, num_photos):
                photo1, photo2 = photos[i], photos[j]
                dist = self.max_location_dist_m
                if photo1.lat is not None and photo1.lon is not None and \
                   photo2.lat is not None and photo2.lon is not None:
                    dist = self._haversine_distance_m(photo1.lat, photo1.lon, photo2.lat, photo2.lon)
                d_loc[i, j] = d_loc[j, i] = dist
        return np.clip(d_loc / self.max_location_dist_m, 0, 1)

    def _get_direction_distance_matrix(self, photos: List[PhotoMeta]) -> np.ndarray:
        num_photos = len(photos)
        d_dir = np.zeros((num_photos, num_photos))
        for i in range(num_photos):
            for j in range(i, num_photos):
                photo1, photo2 = photos[i], photos[j]
                diff = self.max_direction_diff_degrees
                if photo1.gps_img_direction is not None and photo2.gps_img_direction is not None:
                    angle1, angle2 = photo1.gps_img_direction, photo2.gps_img_direction
                    diff = 180 - abs(abs(angle1 - angle2) - 180)
                d_dir[i, j] = d_dir[j, i] = diff
        return np.clip(d_dir / self.max_direction_diff_degrees, 0, 1)

    def _format_clusters(self, cluster_labels: np.ndarray, valid_indices: List[int], total_photos: int) -> List[List[int]]:
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        clustered_indices: List[List[int]] = [[] for _ in range(num_clusters)]
        noise_indices = []
        
        label_map = {}
        next_label = 0
        for i, label in enumerate(cluster_labels):
            original_index = valid_indices[i]
            if label == -1:
                noise_indices.append(original_index)
            else:
                if label not in label_map:
                    label_map[label] = next_label
                    next_label += 1
                clustered_indices[label_map[label]].append(original_index)

        all_indices = set(range(total_photos))
        processed_indices = set(i for g in clustered_indices for i in g) | set(noise_indices)
        unprocessed_indices = all_indices - processed_indices
        
        for i in noise_indices:
            clustered_indices.append([i])
        for i in unprocessed_indices:
            clustered_indices.append([i])

        return clustered_indices

    def _haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    @staticmethod
    def condition(cluster: List[PhotoMeta]) -> bool:
        return len(cluster) > 10
