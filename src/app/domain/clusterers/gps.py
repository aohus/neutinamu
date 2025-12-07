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
        self.geod = Geod(ellps="WGS84")
        
    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        self._adjust_gps_inaccuracy(photos)
        self._correct_outliers_by_speed(photos)

        valid_photos = [p for p in photos if p.lat is not None and p.lon is not None]
        no_gps_photos = [p for p in photos if p.lat is None or p.lon is None]
        
        if not valid_photos:
            return [photos]

        if HAS_HDBSCAN:
            clusters = self._cluster_hdbscan(valid_photos)
        else:
            clusters = self._cluster_optics(valid_photos)
            
        # 노이즈(1개짜리 클러스터)를 시간상 직전 사진이 포함된 클러스터로 병합
        # clusters = self._merge_noise_to_prev_cluster(clusters)
            
        if no_gps_photos:
            clusters.append(no_gps_photos)
        return clusters

    def _correct_outliers_by_speed(self, photos: List[PhotoMeta]) -> None:                                                                                                                                                                                                                      
        """                                                                                                                                                                                                                                                                                     
        도보 이동 기준 속도(5m/s)를 초과하는 GPS 튐 현상을 감지하여,                                                                                                                                                                                                                            
        이전 위치로 보정합니다. (튀는 점 제거 효과)                                                                                                                                                                                                                                             
        """                                                                                                                                                                                                                                                                                     
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]                                                                                                                                                                                                     
        timed_photos.sort(key=lambda x: x.timestamp)                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                
        max_speed_mps = 4.0 # 도보 기준 넉넉하게 약 18km/h                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                
        for i in range(1, len(timed_photos)):                                                                                                                                                                                                                                                   
            prev = timed_photos[i-1]                                                                                                                                                                                                                                                            
            curr = timed_photos[i]                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                
            dt = curr.timestamp - prev.timestamp                                                                                                                                                                                                                                                
            if dt <= 0: continue                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                
            # 거리 계산 (pyproj Geod 사용)                                                                                                                                                                                                                                                      
            # inv(lon1, lat1, lon2, lat2) -> az12, az21, dist                                                                                                                                                                                                                                   
            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                
            speed = dist / dt                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                
            if speed > max_speed_mps:                                                                                                                                                                                                                                                           
                logger.info(f"GPS Outlier detected: {curr.original_name} (Speed: {speed:.2f} m/s). Correcting to previous location.")                                                                                                                                                           
                # 이전 유효 위치로 강제 보정                                                                                                                                                                                                                                                    
                curr.lat = prev.lat                                                                                                                                                                                                                                                             
                curr.lon = prev.lon                                                                                                                                                                                                                                                             
                # 고도가 있다면 함께 보정                                                                                                                                                                                                                                                       
                if prev.alt is not None:                                                                                                                                                                                                                                                        
                    curr.alt = prev.alt  

    def _merge_noise_to_prev_cluster(self, clusters: List[List[PhotoMeta]]) -> List[List[PhotoMeta]]:
        """
        1개짜리 클러스터(노이즈)를 시간상 바로 직전 사진이 속한 '정상 클러스터(2개 이상)'에 병합합니다.
        직전 사진이 없거나, 직전 사진도 노이즈라면 병합하지 않습니다(또는 로직에 따라 처리).
        여기서는 '2개 이상인 클러스터'에 속한 사진들 중에서 가장 가까운 과거 사진을 찾습니다.
        """
        # 1. 클러스터 분류
        valid_clusters = [] # 리스트로 유지 (수정 가능해야 함)
        noise_clusters = []
        
        for c in clusters:
            if len(c) >= 2:
                valid_clusters.append(c)
            else:
                noise_clusters.append(c)
        
        if not valid_clusters:
            return clusters

        # 2. 유효 클러스터 내의 모든 사진을 시간순 정렬하여 검색 인덱스 생성
        # (timestamp, cluster_index) 튜플 리스트
        # timestamp가 없는 경우 제외
        valid_photos_map = []
        for idx, cluster in enumerate(valid_clusters):
            for p in cluster:
                if p.timestamp is not None:
                    valid_photos_map.append((p.timestamp, idx))
        
        valid_photos_map.sort(key=lambda x: x[0])
        
        # 병합되지 못한 노이즈들
        remaining_noise = []

        for noise_c in noise_clusters:
            noise_p = noise_c[0]
            if noise_p.timestamp is None:
                remaining_noise.append(noise_c)
                continue
                
            # 이 노이즈 사진보다 시간이 이르면서 가장 가까운 사진 찾기
            # valid_photos_map은 시간순 정렬되어 있으므로 bisect 등을 쓸 수도 있지만,
            # 간단히 역순 순회하거나 조건에 맞는 마지막 요소 찾기
            
            target_cluster_idx = -1
            
            # 시간순 정렬되어 있으므로, noise_p.timestamp보다 작은 것 중 가장 큰(마지막) 것 찾기
            # 이진 탐색(bisect_left)을 쓰면 더 빠르겠지만 데이터 크기가 크지 않다고 가정
            import bisect

            # bisect는 키 비교가 복잡하므로 단순 순회 (또는 키만 추출한 리스트 사용)
            timestamps = [vp[0] for vp in valid_photos_map]
            insert_pos = bisect.bisect_left(timestamps, noise_p.timestamp)
            
            if insert_pos > 0:
                # insert_pos - 1 이 바로 직전 시간의 사진
                _, target_cluster_idx = valid_photos_map[insert_pos - 1]
                
                # 병합 수행
                valid_clusters[target_cluster_idx].append(noise_p)
                # 병합 후 valid_photos_map 업데이트는 하지 않음 (한 번의 패스로 처리)
                # 만약 노이즈가 연달아 있다면? 
                # 요구사항: "시간상 직전인 사진이 포함된 클러스터" -> 직전이 노이즈였고 걔가 어딘가 병합되었다면?
                # 현재 로직: "이미 2개 이상으로 확정된 클러스터" 기준. 
                # 즉, 노이즈->노이즈->클러스터 병합 체인은 지원 안 함. (복잡도 방지)
            else:
                remaining_noise.append(noise_c)

        return valid_clusters + remaining_noise

    def _adjust_gps_inaccuracy(self, photos: List[PhotoMeta]) -> None:
        """
        도보 이동 기준 속도(5m/s)를 초과하는 GPS 튐 현상을 감지하여,
        이전 위치로 보정합니다. (튀는 점 제거 효과)
        """
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        
        max_speed_mps = 3.0 # 도보 기준 넉넉하게 약 18km/h

        for i in range(1, len(timed_photos)):
            prev = timed_photos[i-1]
            curr = timed_photos[i]
            
            dt = curr.timestamp - prev.timestamp
            if dt <= 0: continue
            
            # 거리 계산 (pyproj Geod 사용)
            # inv(lon1, lat1, lon2, lat2) -> az12, az21, dist
            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
            
            speed = dist / dt
            
            if speed > max_speed_mps:
                logger.info(f"GPS Outlier detected: {curr.original_name} (Speed: {speed:.2f} m/s). Correcting to previous location.")
                # 이전 유효 위치로 강제 보정
                curr.lat = prev.lat
                curr.lon = prev.lon
                # 고도가 있다면 함께 보정
                if prev.alt is not None:
                    curr.alt = prev.alt

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

            # 20초 이내이고, p2(후행 사진)가 유효한 위치 정보를 가지고 있다면
            if 0 <= diff <= 20:
                if p2.lat is not None and p2.lon is not None:
                    p1.lat = p2.lat
                    p1.lon = p2.lon
                    # 고도 정보도 있다면 함께 업데이트 (선택 사항이나 일관성을 위해 권장)
                    if p2.alt is not None:
                        p1.alt = p2.alt

    def _cluster_hdbscan(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        """
        Cluster using HDBSCAN with relative density.
        - min_samples=1: Reduces noise classification (prevents close points from becoming singletons).
        - cluster_selection_epsilon=5m: Ensures points within 5m are merged, fixing over-splitting of close points.
        """
        if len(photos) < 2:
            return [photos]

        coords = np.radians([[p.lat, p.lon] for p in photos])
        
        # Use HDBSCAN defaults for variable density clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_samples, # default 2
            min_samples=1,                     # Reduced to 1 to minimize noise (singletons)
            metric='haversine',
            cluster_selection_method='eom',
            cluster_selection_epsilon=5.0 / 6371000.0, # Merge clusters closer than 5m
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
        if noise:
            result.append(noise)
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
