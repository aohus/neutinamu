import asyncio
import logging
import math
import os
import tempfile
import time
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

from .hybrid import PerformanceMonitor

# -------------------------------------------------------------------------
# Optional Dependencies for Monitoring & Legacy Support
# -------------------------------------------------------------------------
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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

# Legacy Parameters
PARAMS = {
    "eps": 7.488875486571053,
    "max_gps_tol": 44.35800514729294,
    "min_cluster_size": 2,
    "min_samples": 1,
    "max_cluster_size": 8,
    "strict_thresh": 0.15011476241322744,
    "loose_thresh": 0.5018121751611363,
    "w_merge": 0.10088178479325592,
    "w_split": 3.669884372778316,
}


class CosPlaceExtractorLegacy:
    _model = None
    _preprocess = None
    OUTPUT_DIM = 512

    def __init__(self):
        if not HAS_TORCH:
            logger.warning("Torch/Torchvision not installed. CosPlaceExtractorLegacy disabled.")
            return
        if CosPlaceExtractorLegacy._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls):
        logger.info("[Legacy] Loading CosPlace model from Torch Hub...")
        try:
            cls._model = torch.hub.load(
                "gmberton/CosPlace", "get_trained_model", backbone="ResNet50", fc_output_dim=cls.OUTPUT_DIM
            )
            cls._model.eval()

            if torch.cuda.is_available():
                cls._model = cls._model.cuda()

            cls._preprocess = transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("[Legacy] CosPlace model loaded successfully.")
        except Exception as e:
            logger.error(f"[Legacy] Failed to load CosPlace model: {e}")
            cls._model = None

    def extract(self, image_input) -> Optional[np.ndarray]:
        if not HAS_TORCH or self._model is None:
            return None

        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return None
                img = Image.open(image_input)
            else:
                img = image_input

            img = img.convert("RGB")
            device = next(self._model.parameters()).device
            input_tensor = self._preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = self._model(input_tensor)

            vector = feature.cpu().numpy().flatten()
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm

            return vector
        except Exception as e:
            logger.error(f"[Legacy] Feature extraction failed: {e}")
            return None


class HybridClusterLegacy(Clusterer):
    def __init__(self):
        self.geod = Geod(ellps="WGS84")
        self.params = PARAMS

        self.gps_eps = self.params.get("eps", 7.488875486571053)
        self.max_gps_tol = self.params.get("max_gps_tol", 44.35800514729294)
        self.visual_split_thresh = self.params.get("loose_thresh", 0.6418121751611363)
        self.split_min_size = 8
        self.min_cluster_size = self.params.get("min_cluster_size", 2)
        self.min_samples = self.params.get("min_samples", 1)
        self.max_cluster_size = self.params.get("max_cluster_size", 4)

        self.extractor = CosPlaceExtractorLegacy()
        self.storage = GCSStorageService()
        self.performance_monitor = PerformanceMonitor()

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        if not photos:
            return []

        self.performance_monitor.start()

        self._adjust_gps_inaccuracy(photos)
        self._correct_outliers_by_speed(photos)

        features = await self._extract_features_optimized(photos)
        labels = self._run_clustering_logic(photos, features, skip_gps=True)

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

        self.performance_monitor.stop()
        self.performance_monitor.report("HybridClusterLegacy", count=len(photos))
        return result

    async def _extract_features_optimized(self, photos: List[PhotoMeta]) -> List[Optional[np.ndarray]]:
        features = [None] * len(photos)
        batch_size = 32
        semaphore = asyncio.Semaphore(20)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i in range(0, len(photos), batch_size):
                batch_indices = range(i, min(i + batch_size, len(photos)))
                batch_photos = [photos[idx] for idx in batch_indices]

                download_tasks = []
                local_files = {}

                for idx, p in zip(batch_indices, batch_photos):
                    target_path = p.thumbnail_path or p.path
                    if not target_path:
                        if os.path.exists(p.path):
                            local_files[idx] = Path(p.path)
                        continue

                    ext = os.path.splitext(target_path)[1] or ".jpg"
                    if "?" in ext:
                        ext = ext.split("?")[0]

                    dest_path = temp_path / f"{idx}{ext}"
                    local_files[idx] = dest_path
                    download_tasks.append(self._download_safe(target_path, dest_path, semaphore))

                if download_tasks:
                    await asyncio.gather(*download_tasks)

                for idx in batch_indices:
                    if idx in local_files and local_files[idx].exists():
                        try:
                            features[idx] = self.extractor.extract(str(local_files[idx]))
                        except Exception as e:
                            pass

                        if temp_path in local_files[idx].parents:
                            local_files[idx].unlink(missing_ok=True)
        return features

    async def _download_safe(self, url: str, dest: Path, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                await self.storage.download_file(url, dest)
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")

    def _run_clustering_logic(self, photos, features, skip_gps=True):
        n_samples = len(photos)
        if n_samples == 0:
            return np.array([])
        if HDBSCAN is None:
            return np.full(n_samples, -1)

        if skip_gps:
            labels = np.zeros(n_samples, dtype=int)
        else:
            gps_matrix = self._compute_gps_matrix(photos)
            try:
                gps_clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric="precomputed",
                    cluster_selection_epsilon=self.gps_eps,
                    allow_single_cluster=True,
                )
                labels = gps_clusterer.fit_predict(gps_matrix)
            except Exception:
                return np.full(n_samples, -1)

        unique_labels = set(labels) - {-1}
        max_label = labels.max()
        next_label_id = max_label + 1

        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) > self.split_min_size:
                sub_features = [features[i] for i in indices]
                if any(f is None for f in sub_features):
                    continue
                visual_matrix = self._compute_visual_matrix(sub_features)
                try:
                    sub_clusterer = HDBSCAN(
                        min_cluster_size=2,
                        min_samples=2,
                        metric="euclidean",
                        cluster_selection_epsilon=self.visual_split_thresh,
                        allow_single_cluster=False,
                    )
                    sub_labels = sub_clusterer.fit_predict(visual_matrix)
                    for sub_id in set(sub_labels):
                        if sub_id == -1:
                            continue
                        sub_indices = indices[sub_labels == sub_id]
                        labels[sub_indices] = next_label_id
                        next_label_id += 1
                except Exception:
                    pass

        unique_labels = set(labels) - {-1}
        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) > self.max_cluster_size:
                n_splits = math.ceil(len(indices) / self.max_cluster_size)
                sub_features = [features[i] for i in indices]
                if any(f is None for f in sub_features):
                    continue
                try:
                    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
                    feat_stack = np.stack(sub_features)
                    sub_labels = kmeans.fit_predict(feat_stack)
                    for k in range(n_splits):
                        sub_indices = indices[sub_labels == k]
                        labels[sub_indices] = next_label_id
                        next_label_id += 1
                except Exception:
                    pass
        return labels

    def _compute_gps_matrix(self, photos):
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat if p.lat else 0.0, p.lon if p.lon else 0.0] for p in photos])
        for i in range(n):
            for j in range(i + 1, n):
                _, _, dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix

    def _compute_visual_matrix(self, features):
        feature_matrix = np.stack(features)
        return squareform(pdist(feature_matrix, metric="euclidean"))

    def _correct_outliers_by_speed(self, photos: List[PhotoMeta]) -> None:
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        max_speed_mps = 4.0
        for i in range(1, len(timed_photos)):
            prev, curr = timed_photos[i - 1], timed_photos[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                continue
            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
            if (dist / dt) > max_speed_mps:
                curr.lat, curr.lon = prev.lat, prev.lon
                if prev.alt is not None:
                    curr.alt = prev.alt

    def _adjust_gps_inaccuracy(self, photos: List[PhotoMeta]) -> None:
        timed_photos = [p for p in photos if p.timestamp is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        for i in range(len(timed_photos) - 2, -1, -1):
            p1, p2 = timed_photos[i], timed_photos[i + 1]
            if 0 <= (p2.timestamp - p1.timestamp) <= 20:
                if p2.lat is not None:
                    p1.lat, p1.lon = p2.lat, p2.lon
                    if p2.alt is not None:
                        p1.alt = p2.alt
