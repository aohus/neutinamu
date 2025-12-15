import asyncio
import logging
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pyproj import Geod
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from app.domain.clusterers.base import Clusterer
from app.domain.storage.gcs import GCSStorageService
from app.models.photometa import PhotoMeta

# Attempt imports for Torch and HDBSCAN
try:
    import torch
    from PIL import Image, ImageFile
    from torchvision import transforms

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as HDBSCAN
    except ImportError:
        HDBSCAN = None

logger = logging.getLogger(__name__)

# Parameters from params_dataset_cosplace_masking_251215_103834.json
PARAMS = {
    "eps": 7.488875486571053,
    "max_gps_tol": 44.35800514729294,
    "min_cluster_size": 2,
    "min_samples": 1,
    "max_cluster_size": 8,
    "strict_thresh": 0.15011476241322744,
    # "loose_thresh": 0.6418121751611363,
    "loose_thresh": 0.5018121751611363,
    "w_merge": 0.10088178479325592,
    "w_split": 3.669884372778316,
}


class CosPlaceExtractor:
    _model = None
    _preprocess = None
    OUTPUT_DIM = 512

    def __init__(self):
        if not HAS_TORCH:
            logger.warning("Torch/Torchvision not installed. CosPlaceExtractor will be disabled.")
            return
        if CosPlaceExtractor._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls):
        logger.info("Loading CosPlace model from Torch Hub...")
        try:
            # 'gmberton/CosPlace' 리포지토리에서 자동으로 모델과 가중치를 가져옵니다.
            cls._model = torch.hub.load(
                "gmberton/CosPlace", "get_trained_model", backbone="ResNet50", fc_output_dim=cls.OUTPUT_DIM
            )
            cls._model.eval()

            if torch.cuda.is_available():
                cls._model = cls._model.cuda()

            cls._preprocess = transforms.Compose(
                [
                    transforms.Resize((480, 640)),  # 장소 인식에 적합한 해상도
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("CosPlace model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CosPlace model: {e}")
            cls._model = None

    def extract(self, image_input) -> Optional[np.ndarray]:
        if not HAS_TORCH or self._model is None:
            return None

        try:
            if isinstance(image_input, str):
                # Ensure file exists
                if not os.path.exists(image_input):
                    logger.warning(f"Image not found: {image_input}")
                    return None
                img = Image.open(image_input)
            else:
                img = image_input

            # Handle alpha channel if present
            img = img.convert("RGB")

            device = next(self._model.parameters()).device
            input_tensor = self._preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = self._model(input_tensor)

            vector = feature.cpu().numpy().flatten()

            # L2 Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm

            return vector
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_input}: {e}")
            return None


class HybridCluster(Clusterer):
    def __init__(self):
        self.geod = Geod(ellps="WGS84")
        self.params = PARAMS

        # 1. GPS Clustering Params (1단계)
        self.gps_eps = self.params.get("eps", 7.488875486571053)
        self.max_gps_tol = self.params.get("max_gps_tol", 44.35800514729294)

        # 2. Visual Split Params (2단계)
        self.visual_split_thresh = self.params.get("loose_thresh", 0.6418121751611363)
        self.split_min_size = 8
        self.min_cluster_size = self.params.get("min_cluster_size", 2)
        self.min_samples = self.params.get("min_samples", 1)

        # 3. Size Enforcement (3단계)
        self.max_cluster_size = self.params.get("max_cluster_size", 4)

        print(f"HybridCluster Initialized. HAS_TORCH={HAS_TORCH}")
        logger.info(f"HybridCluster Initialized. HAS_TORCH={HAS_TORCH}")

        self.extractor = CosPlaceExtractor()
        self.storage = GCSStorageService()

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        if not photos:
            return []

        # 1. GPS Preprocessing (Noise reduction)
        self._adjust_gps_inaccuracy(photos)
        self._correct_outliers_by_speed(photos)

        # Check if any photo is missing GPS
        missing_gps = any(p.lat is None or p.lon is None for p in photos)

        # 2. Extract Features
        logger.info(f"Extracting features for {len(photos)} photos using CosPlace...")
        # Use optimized batch extraction with GCS download
        features = await self._extract_features_optimized(photos)

        # 3. Run Hybrid Clustering Logic
        # If missing_gps is True, skip GPS clustering (Step 1) and use visual split only.
        labels = self._run_clustering_logic(photos, features, skip_gps=True)

        # 4. Group by Labels
        clusters = {}
        noise = []
        for i, label in enumerate(labels):
            if label == -1:
                noise.append(photos[i])
            else:
                clusters.setdefault(label, []).append(photos[i])

        result = list(clusters.values())

        if noise:
            result.append(noise)

        return result

    async def _extract_features_optimized(self, photos: List[PhotoMeta]) -> List[Optional[np.ndarray]]:
        features = [None] * len(photos)
        batch_size = 32
        # Concurrency limit for downloads
        semaphore = asyncio.Semaphore(20)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i in range(0, len(photos), batch_size):
                batch_indices = range(i, min(i + batch_size, len(photos)))
                batch_photos = [photos[idx] for idx in batch_indices]

                download_tasks = []
                local_files = {}  # idx -> path

                for idx, p in zip(batch_indices, batch_photos):
                    # Prioritize thumbnail_url, then url
                    target_path = p.thumbnail_path or p.path
                    logger.info(f"Photo idx={idx}, thumbnail_path={p.thumbnail_path}, path={p.path}")

                    if not target_path:
                        # Fallback: if p.path exists locally, use it
                        if os.path.exists(p.path):
                            local_files[idx] = Path(p.path)
                        continue

                    # Generate temp filename
                    ext = os.path.splitext(target_path)[1]
                    if not ext:
                        ext = ".jpg"
                    if "?" in ext:
                        ext = ext.split("?")[0]

                    dest_path = temp_path / f"{idx}{ext}"
                    local_files[idx] = dest_path

                    # Create download task
                    download_tasks.append(self._download_safe(target_path, dest_path, semaphore))

                # Wait for batch downloads
                if download_tasks:
                    await asyncio.gather(*download_tasks)

                # Extract features for the batch
                for idx in batch_indices:
                    if idx in local_files and local_files[idx].exists():
                        try:
                            features[idx] = self.extractor.extract(str(local_files[idx]))
                        except Exception as e:
                            logger.error(f"Extraction failed for index {idx}: {e}")

                        # Cleanup file immediately to save space
                        # Only delete if it is in our temp dir
                        if temp_path in local_files[idx].parents:
                            local_files[idx].unlink(missing_ok=True)

        return features

    async def _download_safe(self, url: str, dest: Path, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                await self.storage.download_file(url, dest)
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")

    def _run_clustering_logic(
        self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]], skip_gps: bool = True
    ) -> np.ndarray:
        """
        2-Step Clustering with Size Limit:
        Step 1. GPS Distance (eps)로 1차 그룹핑 (skip_gps=True면 생략하고 전체를 하나의 클러스터로 시작)
        Step 2. 그룹 내 사진 수가 split_min_size 넘으면 Visual Distance로 2차 분할 (HDBSCAN)
        Step 3. 여전히 max_cluster_size를 넘는 클러스터는 K-Means로 강제 분할
        """
        n_samples = len(photos)
        if n_samples == 0:
            return np.array([])

        if HDBSCAN is None:
            logger.error("HDBSCAN is not available. Returning all as one cluster (or noise).")
            return np.full(n_samples, -1)

        # --- [Step 1] GPS 기반 1차 클러스터링 ---
        if skip_gps:
            # GPS가 없는 사진이 섞여있으면, GPS 클러스터링을 건너뛰고 전체를 하나로 묶음
            labels = np.zeros(n_samples, dtype=int)
        else:
            gps_matrix = self._compute_gps_matrix(photos)

            try:
                gps_clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric="precomputed",
                    cluster_selection_epsilon=self.gps_eps,
                    cluster_selection_method="eom",
                    allow_single_cluster=True,
                )
                labels = gps_clusterer.fit_predict(gps_matrix)
            except Exception as e:
                logger.error(f"Step 1 GPS HDBSCAN failed: {e}")
                return np.full(n_samples, -1)

        # --- [Step 2] 거대 클러스터 대상 이미지 기반 재분할 (HDBSCAN) ---
        max_label = labels.max()
        next_label_id = max_label + 1

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]

            # 조건: 사진 수가 분할 최소 크기보다 큰가?
            if len(indices) > self.split_min_size:
                sub_features = [features[i] for i in indices]

                # Feature가 하나라도 없으면 분할 불가 (Skip)
                if any(f is None for f in sub_features):
                    continue

                visual_matrix = self._compute_visual_matrix(sub_features)

                try:
                    # 2차 클러스터링 (이미지 유사도 기반)
                    sub_clusterer = HDBSCAN(
                        min_cluster_size=2,
                        min_samples=2,
                        metric="euclidean",  # visual_matrix is Euclidean distance
                        cluster_selection_epsilon=self.visual_split_thresh,
                        allow_single_cluster=False,
                    )
                    sub_labels = sub_clusterer.fit_predict(visual_matrix)

                    found_sub_clusters = set(sub_labels)

                    # -1 (Noise) in sub-clustering usually stays with original label or becomes new noise?
                    # Experimental code:
                    # for sub_id in found_sub_clusters:
                    #     sub_indices = indices[sub_labels == sub_id]
                    #     labels[sub_indices] = next_label_id ...
                    # This implies sub_labels -1 (noise in visual split) are NOT reassigned new IDs,
                    # so they keep the original 'cluster_id'. This effectively "peels off" valid sub-clusters
                    # and leaves the rest in the original group.

                    for sub_id in found_sub_clusters:
                        # Skip noise from sub-clustering if we want to keep them in parent
                        if sub_id == -1:
                            continue

                        sub_indices = indices[sub_labels == sub_id]
                        labels[sub_indices] = next_label_id
                        next_label_id += 1

                except Exception as e:
                    logger.warning(f"Step 2 Visual Split failed for cluster {cluster_id}: {e}")

        # --- [Step 3] Force Split if cluster size > max_cluster_size (K-Means) ---
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            n_members = len(indices)

            if n_members > self.max_cluster_size:
                n_splits = math.ceil(n_members / self.max_cluster_size)
                sub_features = [features[i] for i in indices]

                if any(f is None for f in sub_features):
                    continue

                try:
                    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                    # KMeans requires (n_samples, n_features)
                    # sub_features is list of arrays. stack them.
                    feat_stack = np.stack(sub_features)
                    sub_labels = kmeans.fit_predict(feat_stack)

                    for k in range(n_splits):
                        sub_indices = indices[sub_labels == k]
                        labels[sub_indices] = next_label_id
                        next_label_id += 1

                except Exception as e:
                    logger.warning(f"Step 3 K-Means force split failed for cluster {cluster_id}: {e}")

        return labels

    def _compute_gps_matrix(self, photos: List[PhotoMeta]) -> np.ndarray:
        """순수 GPS 거리 매트릭스 계산"""
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat if p.lat else 0.0, p.lon if p.lon else 0.0] for p in photos])

        for i in range(n):
            for j in range(i + 1, n):
                _, _, dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix

    def _compute_visual_matrix(self, features: List[np.ndarray]) -> np.ndarray:
        """순수 Visual Distance 매트릭스 계산 (L2 Euclidean Distance)"""
        feature_matrix = np.stack(features)
        dist_matrix = squareform(pdist(feature_matrix, metric="euclidean"))
        return dist_matrix

    def _correct_outliers_by_speed(self, photos: List[PhotoMeta]) -> None:
        """
        도보 이동 기준 속도(5m/s)를 초과하는 GPS 튐 현상을 감지하여,
        이전 위치로 보정합니다. (튀는 점 제거 효과)
        """
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
        timed_photos.sort(key=lambda x: x.timestamp)

        max_speed_mps = 4.0  # 도보 기준 넉넉하게 약 18km/h

        for i in range(1, len(timed_photos)):
            prev = timed_photos[i - 1]
            curr = timed_photos[i]

            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                continue

            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)

            speed = dist / dt

            if speed > max_speed_mps:
                # logger.info(f"GPS Outlier detected: {curr.original_name} (Speed: {speed:.2f} m/s). Correcting to previous location.")
                curr.lat = prev.lat
                curr.lon = prev.lon
                if prev.alt is not None:
                    curr.alt = prev.alt

    def _adjust_gps_inaccuracy(self, photos: List[PhotoMeta]) -> None:
        """
        촬영 시간 간격이 짧은(20초 이내) 사진들의 GPS 오차를 보정합니다.
        """
        timed_photos = [p for p in photos if p.timestamp is not None]
        timed_photos.sort(key=lambda x: x.timestamp)

        for i in range(len(timed_photos) - 2, -1, -1):
            p1 = timed_photos[i]
            p2 = timed_photos[i + 1]

            diff = p2.timestamp - p1.timestamp

            if 0 <= diff <= 20:
                if p2.lat is not None and p2.lon is not None:
                    p1.lat = p2.lat
                    p1.lon = p2.lon
                    if p2.alt is not None:
                        p1.alt = p2.alt
