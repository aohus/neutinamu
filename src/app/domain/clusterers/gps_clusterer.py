import logging
import math
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from app.domain.clusterers.base_clusterer import Clusterer
from app.models.photometa import PhotoMeta
from pyproj import Geod
from sklearn.cluster import DBSCAN, OPTICS

logger = logging.getLogger(__name__)

# Try importing HDBSCAN
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logger.warning("HDBSCAN not found. Falling back to OPTICS/DBSCAN.")

class GPSCluster(Clusterer):
    def __init__(self):
        self.max_dist_m = 15
        self.min_samples = 2
        
    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        valid_photos = [p for p in photos if p.lat is not None and p.lon is not None]
        no_gps_photos = [p for p in photos if p.lat is None or p.lon is None]
        
        if not valid_photos:
            return [[p] for p in photos]

        if HAS_HDBSCAN:
            # Initial epsilon ~ 2.5m
            initial_epsilon = 2.5 / 6371000.0
            clusters = self._cluster_hdbscan_recursive(valid_photos, initial_epsilon)
        else:
            clusters = self._cluster_optics(valid_photos)
            
        for p in no_gps_photos:
            clusters.append([p])
            
        return clusters

    def _cluster_hdbscan_recursive(self, photos: List[PhotoMeta], epsilon_rad: float) -> List[List[PhotoMeta]]:
        """
        Recursively cluster using HDBSCAN.
        If a cluster is too large (> 8), try clustering it again with a tighter epsilon.
        """
        if len(photos) < 2:
            return [photos]

        coords = np.radians([[p.lat, p.lon] for p in photos])
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='haversine',
            cluster_selection_epsilon=epsilon_rad, 
            cluster_selection_method='eom',
            allow_single_cluster=True # Important for recursion to work on subsets
        )
        labels = clusterer.fit_predict(coords)
        
        initial_clusters = self._group_by_labels(photos, labels)
        
        final_clusters = []
        
        # Check each cluster size
        for group in initial_clusters:
            # If group is large and we can tighten epsilon further
            if len(group) > 8 and epsilon_rad > (0.5 / 6371000.0):
                # If the clusterer returned just one cluster (itself) and it's still large,
                # we force a split by significantly reducing epsilon or increasing min_cluster_size?
                # Actually, if allow_single_cluster=True, it might just return all as one.
                # We need to ensure we are making progress. 
                # If labels are all the same (and not -1), we haven't split.
                
                unique_labels = set(labels)
                if len(unique_labels) == 1 and -1 not in unique_labels:
                     # It found 1 cluster. Tighten epsilon.
                     new_epsilon = epsilon_rad * 0.75
                     logger.info(f"Refining large cluster (size {len(group)}) with epsilon {new_epsilon * 6371000.0:.2f}m")
                     sub_clusters = self._cluster_hdbscan_recursive(group, new_epsilon)
                     
                     # If recursion didn't split anything (returned same group), just keep it to avoid infinite loop
                     if len(sub_clusters) == 1 and len(sub_clusters[0]) == len(group):
                         final_clusters.append(group)
                     else:
                         final_clusters.extend(sub_clusters)
                else:
                    # It split somewhat. Check sub-clusters recursively?
                    # For now, let's assume the split was good enough, or recurse on THEM.
                    # Let's recurse on each result to be safe.
                    new_epsilon = epsilon_rad # Keep same epsilon or tighten? 
                    # If we split, the sub-clusters might still be dense.
                    # Let's keep tightening to be aggressive on large groups.
                    new_epsilon = epsilon_rad * 0.75
                    
                    # We must be careful not to infinite loop if geometry is identical.
                    # Simple check: if variance is zero (all same point), don't split.
                    if self._is_same_location(group):
                        final_clusters.append(group)
                    else:
                        sub_clusters = self._cluster_hdbscan_recursive(group, new_epsilon)
                        final_clusters.extend(sub_clusters)
            else:
                final_clusters.append(group)
                
        return final_clusters

    def _is_same_location(self, photos: List[PhotoMeta]) -> bool:
        if not photos: return True
        lats = [p.lat for p in photos]
        lons = [p.lon for p in photos]
        return (max(lats) - min(lats) < 1e-7) and (max(lons) - min(lons) < 1e-7)

    def _cluster_optics(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        coords = np.radians([[p.lat, p.lon] for p in photos])
        
        clusterer = OPTICS(
            min_samples=2,
            metric='haversine',
            max_eps=50.0 / 6371000.0,
            xi=0.05 
        )
        labels = clusterer.fit_predict(coords)
        return self._group_by_labels(photos, labels)

    def _group_by_labels(self, photos: List[PhotoMeta], labels: np.ndarray) -> List[List[PhotoMeta]]:
        clusters = {}
        noise = []
        for p, label in zip(photos, labels):
            if label == -1:
                noise.append(p)
            else:
                clusters.setdefault(label, []).append(p)
        
        result = list(clusters.values())
        for p in noise:
            result.append([p])
        return result


class BaseExecuter:
    def __init__(self):
        self.geod = Geod(ellps="WGS84")

    def _haversine_distance_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return self._distance_3d_m(lat1, lon1, lat2, lon2, None, None)
    
    def _geo_distanc_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        az12, az21, dist_geod = self.geod.inv(lon1, lat1, lon2, lat2)
        return dist_geod
    
    def _distance_3d_m(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        alt1: Optional[float] = None,
        alt2: Optional[float] = None,
    ) -> float:
        """
        위도/경도 + (옵션) 고도를 사용하는 3D 거리 (미터 단위).

        - alt1, alt2 둘 다 주어지면: 3D 거리 = sqrt(수평거리^2 + 고도차^2)
        - 하나라도 None이면: 수평 거리만 반환
        """
        R = 6371000.0  # Earth radius in meters

        d2d = self._geo_distanc_m(lat1, lon1, lat2, lon2)
        
        # 고도 포함 3D 거리
        w_alt = 0.3
        if alt1 is not None and alt2 is not None:
            dz = alt2 - alt1
            return math.sqrt(d2d**2 + (w_alt * dz)**2)
        else:
            return d2d