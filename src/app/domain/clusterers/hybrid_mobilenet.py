# import io
# import logging
# from concurrent.futures import ProcessPoolExecutor
# from typing import Dict, List, Optional

# import numpy as np

# # PyTorch & MobileNet
# import torch
# import torchvision.transforms as T
# from app.domain.clusterers.base import Clusterer
# from app.domain.storage.factory import get_storage_client
# from app.models.photometa import PhotoMeta
# from PIL import Image, ImageFile, ImageFilter, ImageOps
# from pyproj import Geod
# from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# # HDBSCAN
# try:
#     from sklearn.cluster import HDBSCAN
# except ImportError:
#     import hdbscan as HDBSCAN

# logger = logging.getLogger(__name__)
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # --- [전역 모델 로딩] ---
# # 워커 프로세스마다 모델을 로드하면 메모리(4GB)가 터질 수 있습니다.
# # 따라서 'spawn' 방식으로 프로세스를 띄우거나, 필요한 순간에만 로드해야 합니다.
# # 여기서는 2 CPU 제한이므로, 매번 로드하기보다 가벼운 모델을 전역으로 공유하거나
# # 함수 내에서 로드하되 캐싱하는 방식을 씁니다.

# class FeatureExtractor:
#     _instance = None
#     _model = None
#     _preprocess = None

#     @classmethod
#     def get_model(cls):
#         if cls._model is None:
#             # CPU 모드로 경량 모델 로드
#             # pretrained=True: 이미지넷 데이터로 학습된 가중치 사용 (사물의 특징을 잘 앎)
#             weights = MobileNet_V3_Small_Weights.DEFAULT
#             cls._model = mobilenet_v3_small(weights=weights)
#             cls._model.eval() # 평가 모드 (학습 X)

#             # 마지막 분류 레이어(Classifier) 제거 -> 특징 벡터(Embedding)만 추출
#             # MobileNetV3 Small의 마지막 features 출력은 576차원
#             cls._model.classifier = torch.nn.Identity()

#             cls._preprocess = weights.transforms()
#         return cls._model, cls._preprocess

# def _extract_mobilenet_feature(path: str) -> Optional[np.ndarray]:
#     """
#     MobileNetV3를 사용하여 이미지의 의미론적 특징(Semantic Feature) 추출
#     """
#     try:
#         # 1. 이미지 다운로드 (기존 동일)
#         storage = get_storage_client()
#         img_data = None

#         if "storage.googleapis.com" in path or path.startswith("gs://"):
#             try:
#                 if path.startswith("gs://"):
#                     blob_name = path.replace("gs://", "").split("/", 1)[1]
#                 else:
#                     blob_name = path.split("storage.googleapis.com/")[1].split("/", 1)[1]
#                 bucket = storage.bucket
#                 blob = bucket.blob(blob_name)
#                 # 딥러닝 모델은 전체 이미지가 필요할 수 있으나,
#                 # MobileNet은 224x224로 리사이즈 하므로 100KB 정도면 충분
#                 img_data = blob.download_as_bytes(start=0, end=100 * 1024)
#             except: return None
#         else:
#             with open(path, "rb") as f: img_data = f.read()

#         if not img_data: return None

#         # 2. 전처리
#         with Image.open(io.BytesIO(img_data)) as img:
#             img = img.convert("RGB") # PyTorch 모델은 RGB 3채널 필요

#             # 모델 로드 (싱글톤 패턴 활용)
#             model, preprocess = FeatureExtractor.get_model()

#             # 이미지 텐서 변환 및 정규화
#             input_tensor = preprocess(img).unsqueeze(0) # Batch 차원 추가

#             # 3. 추론 (Inference) - CPU
#             with torch.no_grad():
#                 feature_vector = model(input_tensor)

#             # (1, 576) -> (576,) numpy array
#             return feature_vector.squeeze().numpy()

#     except Exception as e:
#         logger.error(f"MobileNet extraction failed: {e}")
#         return None


# class HybridCluster(Clusterer):
#     def __init__(self):
#         self.geod = Geod(ellps="WGS84")

#         # [튜닝 파라미터]
#         self.max_gps_tolerance_m = 30.0

#         # 차원이 커졌으므로(256차원) 유사도 임계값도 조정 필요
#         # Cosine Distance 기준 (0.0: 동일, 1.0: 다름)
#         self.similarity_strict_thresh = 0.25 # 이보다 작으면 아주 강력하게 병합
#         self.similarity_loose_thresh = 0.55  # 이보다 크면 분리

#     async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
#         # 1. GPS 보정
#         self._correct_outliers_by_speed(photos)
#         self._adjust_gps_inaccuracy(photos)

#         valid_photos = [p for p in photos if p.lat is not None and p.lon is not None]
#         no_gps_photos = [p for p in photos if p.lat is None or p.lon is None]

#         if not valid_photos: return [photos]

#         # 2. 2D Grid 특징 추출 (병렬 처리)
#         img_paths = [p.path for p in valid_photos]
#         logger.info(f"Extracting 2D grid features for {len(valid_photos)} photos...")

#         features = []
#         if img_paths:
#             # CPU 바운드 작업이므로 ProcessPoolExecutor 필수
#             with ProcessPoolExecutor(max_workers=2) as executor:
#                 features = list(executor.map(_extract_mobilenet_feature, img_paths))
#         else:
#             features = [None] * len(valid_photos)

#         # 3. 가중치 거리 행렬 계산
#         dist_matrix = self._compute_weighted_distance_matrix(valid_photos, features)

#         # 4. HDBSCAN 적용 (엄격 모드)
#         try:
#             clusterer = HDBSCAN(
#                 min_cluster_size=2,
#                 min_samples=2,      # 노이즈 제거 강화
#                 metric='precomputed',
#                 cluster_selection_epsilon=3.0, # 3.0 (Weighted)Meter 이내 병합
#                 cluster_selection_method='leaf' # 세밀한 분류 선호
#             )
#             labels = clusterer.fit_predict(dist_matrix)
#         except Exception as e:
#             logger.error(f"HDBSCAN failed: {e}")
#             return [valid_photos]

#         # 5. 결과 정리
#         clusters = self._group_by_labels(valid_photos, labels)
#         if no_gps_photos: clusters.append(no_gps_photos)
#         return clusters

#     def _compute_weighted_distance_matrix(self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]]) -> np.ndarray:
#         n = len(photos)
#         dist_matrix = np.zeros((n, n))
#         coords = np.array([[p.lat, p.lon] for p in photos])

#         for i in range(n):
#             for j in range(i + 1, n):
#                 _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])

#                 weight_factor = 1.0

#                 if features[i] is not None and features[j] is not None:
#                     # Cosine Distance
#                     similarity = np.dot(features[i], features[j])
#                     struct_dist = 1.0 - similarity
#                     struct_dist = max(0.0, min(1.0, struct_dist))

#                     # A. 구조 매우 유사 (위치까지 일치)
#                     if struct_dist < self.similarity_strict_thresh:
#                         # GPS 거리 10%~20%로 축소 (강력 병합)
#                         weight_factor = 0.1 + (struct_dist * 0.5)
#                     # B. 구조 다름
#                     elif struct_dist > self.similarity_loose_thresh:
#                         # GPS 거리 3배~8배 확대 (확실한 분리)
#                         weight_factor = 3.0 + (struct_dist - 0.55) * 10.0
#                     # C. 중간
#                     else:
#                         weight_factor = 1.0 + (struct_dist - 0.25) * 3.0
#                 else:
#                     if gps_dist < 10.0: weight_factor = 1.0
#                     else: weight_factor = 2.0

#                 final_dist = gps_dist * weight_factor

#                 # [Safety Cutoff]
#                 # GPS가 물리적으로 너무 먼데 가중치로 인해 가까워진 경우 방지
#                 if gps_dist > self.max_gps_tolerance_m:
#                     if weight_factor > 0.2: # 정말 완벽하게 똑같지 않으면
#                         final_dist = 1000.0

#                 dist_matrix[i][j] = dist_matrix[j][i] = final_dist

#         return dist_matrix

#     def _correct_outliers_by_speed(self, photos: List[PhotoMeta]) -> None:
#         """이전 코드와 동일 (속도 기반 보정)"""
#         timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
#         timed_photos.sort(key=lambda x: x.timestamp)
#         max_speed_mps = 5.0
#         for i in range(1, len(timed_photos)):
#             prev = timed_photos[i-1]
#             curr = timed_photos[i]
#             dt = curr.timestamp - prev.timestamp
#             if dt <= 0: continue
#             _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
#             if (dist / dt) > max_speed_mps:
#                 curr.lat = prev.lat
#                 curr.lon = prev.lon
#                 if prev.alt is not None: curr.alt = prev.alt

#     def _adjust_gps_inaccuracy(self, photos: List[PhotoMeta]) -> None:
#         """이전 코드와 동일 (시간차 미세 보정)"""
#         timed_photos = [p for p in photos if p.timestamp is not None]
#         timed_photos.sort(key=lambda x: x.timestamp)
#         for i in range(len(timed_photos) - 2, -1, -1):
#             p1 = timed_photos[i]
#             p2 = timed_photos[i+1]
#             if 0 <= (p2.timestamp - p1.timestamp) <= 20:
#                 if p2.lat is not None and p2.lon is not None:
#                     p1.lat = p2.lat
#                     p1.lon = p2.lon
#                     if p2.alt is not None: p1.alt = p2.alt

#     def _group_by_labels(self, photos: List[PhotoMeta], labels: np.ndarray) -> List[List[PhotoMeta]]:
#         clusters = {}
#         noise = []
#         for p, label in zip(photos, labels):
#             if label == -1: noise.append(p)
#             else: clusters.setdefault(label, []).append(p)
#         result = list(clusters.values())
#         if noise:
#             for n_photo in noise: result.append([n_photo])
#         return result

#     def condition(self, photos: list[PhotoMeta]):
#         return len(photos) >= 5
