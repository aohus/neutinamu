import logging
import math
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from app.domain.clusterers.base import Clusterer
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
        self.max_dist_m = 20
        self.min_samples = 2
        
    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        # GPS 오차 보정: 20초 이내 연속 촬영 시 선행 사진의 위치를 후행 사진 기준으로 보정
        self._adjust_gps_inaccuracy(photos)

        valid_photos = [p for p in photos if p.lat is not None and p.lon is not None]
        no_gps_photos = [p for p in photos if p.lat is None or p.lon is None]
        
        if not valid_photos:
            return [[p] for p in photos]

        if HAS_HDBSCAN:
            clusters = self._cluster_hdbscan(valid_photos)
        else:
            clusters = self._cluster_optics(valid_photos)
            
        for p in no_gps_photos:
            clusters.append([p])
            
        return clusters

    def _adjust_gps_inaccuracy(self, photos: List[PhotoMeta]) -> None:
        """
        촬영 시간 간격이 짧은(20초 이내) 사진들의 GPS 오차를 보정합니다.
        핸드폰 카메라 실행 직후(Cold Start)에는 GPS 정밀도가 낮아 이전 위치나 부정확한 위치가 기록될 수 있습니다.
        따라서 시간상 뒤에 찍힌(GPS가 안정화되었을 가능성이 높은) 사진의 위치 정보를
        앞선 사진에 덮어씌워 위치 정확도를 높입니다.
        """
        # 타임스탬프가 있는 사진만 추출하여 시간순 정렬
        timed_photos = [p for p in photos if p.timestamp is not None]
        timed_photos.sort(key=lambda x: x.timestamp)

        # 뒤에서부터 순회하여 나중 사진의 위치 정보를 앞 사진으로 전파 (체이닝 효과)
        for i in range(len(timed_photos) - 2, -1, -1):
            p1 = timed_photos[i]
            p2 = timed_photos[i+1]

            # 시간 차이 계산
            diff = (p2.timestamp - p1.timestamp)
            logger.info(f"=============================={diff}")

            # 20초 이내이고, p2(후행 사진)가 유효한 위치 정보를 가지고 있다면
            if 0 <= diff <= 20:
                logger.info(f"=============================={p1.original_name}, {p2.original_name}")
                if p2.lat is not None and p2.lon is not None:
                    p1.lat = p2.lat
                    p1.lon = p2.lon
                    # 고도 정보도 있다면 함께 업데이트 (선택 사항이나 일관성을 위해 권장)
                    if p2.alt is not None:
                        p1.alt = p2.alt

    def _cluster_hdbscan(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Cluster using HDBSCAN with relative density.
        We do not enforce a fixed cluster_selection_epsilon to allow for varying densities.
        """
        if len(photos) < 2:
            return [photos]

        coords = np.radians([[p.lat, p.lon] for p in photos])
        
        # Use HDBSCAN defaults for variable density clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_samples, # default 2
            min_samples=self.min_samples,      # default 2
            metric='haversine',
            cluster_selection_method='eom',
            # cluster_selection_epsilon is omitted/0.0 to allow detecting clusters of varying densities
            allow_single_cluster=True 
        )
        labels = clusterer.fit_predict(coords)
        
        return self._group_by_labels(photos, labels)

    def _cluster_optics(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        coords = np.radians([[p.lat, p.lon] for p in photos])
        
        clusterer = OPTICS(
            min_samples=3,
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
        
        # w_alt = 0.3
        # if alt1 is not None and alt2 is not None:
        #     dz = alt2 - alt1
        #     return math.sqrt(d2d**2 + (w_alt * dz)**2)
        # else:
        #     return d2d
        return d2d
