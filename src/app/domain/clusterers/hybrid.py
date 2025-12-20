import asyncio
import gc
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from pyproj import Geod
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# 도메인 의존성 (환경에 맞게 유지)
from app.domain.clusterers.base import Clusterer
from app.domain.storage.gcs import GCSStorageService
from app.models.photometa import PhotoMeta

# --- [Optional Imports for HDBSCAN] ---
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as HDBSCAN
    except ImportError:
        HDBSCAN = None

# --- [Optional Imports for ONNX/Torch] ---
try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    from torchvision import transforms

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

# Parameters
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


class PerformanceMonitor:
    """Helper to measure Time, CPU, and Memory usage."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.start_cpu = 0.0
        self.end_cpu = 0.0
        self.start_mem = 0
        self.end_mem = 0
        self.process = psutil.Process() if HAS_PSUTIL else None

    def start(self):
        gc.collect()  # Clean up before starting measurement
        self.start_time = time.time()
        if self.process:
            self.process.cpu_percent(interval=None)  # Set baseline for next call
            self.start_mem = self.process.memory_info().rss

    def stop(self):
        self.end_time = time.time()
        if self.process:
            self.end_cpu = self.process.cpu_percent(interval=None)  # Get average CPU usage since last call
            self.end_mem = self.process.memory_info().rss

    @property
    def duration(self):
        return self.end_time - self.start_time

    def report(self, label: str, count: Optional[int] = None) -> str:
        count_str = f" (N={count})" if count is not None else ""
        msg = f"[{label}]{count_str} Time: {self.duration:.4f}s"
        if self.process:
            mem_diff_mb = (self.end_mem - self.start_mem) / (1024 * 1024)
            end_mem_mb = self.end_mem / (1024 * 1024)
            msg += f" | CPU: {self.end_cpu:.1f}% | Mem: {end_mem_mb:.1f}MB (Delta: {mem_diff_mb:+.2f}MB)"

        # Write to log file
        try:
            log_dir = Path("logs/performance")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "performance_log.txt"

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {msg}\n")
        except Exception as e:
            # Just log error but don't fail main flow
            logging.getLogger(__name__).error(f"Failed to write performance log: {e}")

        return msg


class CosPlaceExtractor:
    """
    ONNX Runtime을 우선 사용하여 CPU 추론 속도를 최적화한 Extractor.
    ONNX 모델이 없거나 로드 실패 시 PyTorch로 Fallback 합니다.
    """

    _session = None
    _torch_model = None
    _torch_preprocess = None

    # CosPlace ResNet50 Output Dim
    OUTPUT_DIM = 512
    # ImageNet Mean/Std for Numpy Preprocessing
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, onnx_path="img_models/cosplace_resnet50_int8.onnx"):
        self.use_onnx = False

        # 1. Try Loading ONNX
        if HAS_ONNX and os.path.exists(onnx_path):
            if CosPlaceExtractor._session is None:
                self._load_onnx_model(onnx_path)
            if CosPlaceExtractor._session is not None:
                self.use_onnx = True

        # 2. Fallback to Torch if ONNX is not available
        if not self.use_onnx:
            if HAS_TORCH:
                if CosPlaceExtractor._torch_model is None:
                    self._load_torch_model()
            else:
                logger.warning("Neither ONNX nor Torch is available. Feature extraction disabled.")

    @classmethod
    def _load_onnx_model(cls, path):
        logger.info(f"Loading ONNX model from {path}...")
        try:
            # CPU 전용 설정 (필요시 병렬 실행 옵션 조정 가능)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2  # 코어 수에 맞게 조정
            sess_options.log_severity_level = 3  # Suppress warnings (like Unknown CPU vendor)
            
            # [Memory Optimization] Disable Memory Arena
            # This releases memory back to OS immediately after inference,
            # preventing huge memory retention when processing large batches.
            sess_options.enable_cpu_mem_arena = False
            
            cls._session = ort.InferenceSession(path, sess_options, providers=["CPUExecutionProvider"])
            logger.info("CosPlace ONNX model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            cls._session = None

    @classmethod
    def _load_torch_model(cls):
        logger.info("Loading CosPlace model from Torch Hub (Fallback)...")
        try:
            cls._torch_model = torch.hub.load(
                "gmberton/CosPlace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=cls.OUTPUT_DIM,
                trust_repo=True,
            )
            cls._torch_model.eval()
            cls._torch_preprocess = transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("CosPlace Torch model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Torch model: {e}")

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        [핵심 최적화] 이미지 배치를 한 번에 처리하여 Feature Matrix 반환
        Return: (Batch_Size, 512) Numpy Array
        """
        if not images:
            return np.array([])

        # A. ONNX Inference (Fastest on CPU)
        if self.use_onnx and self._session:
            try:
                # 1. Preprocess (Numpy Vectorization)
                # [Memory Optimization] Pre-allocate input tensor
                batch_size = len(images)
                input_tensor = np.zeros((batch_size, 3, 480, 640), dtype=np.float32)

                for i, img in enumerate(images):
                    img = img.convert("RGB").resize((640, 480))  # PIL uses (W, H)
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_np = (img_np - self.MEAN) / self.STD
                    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
                    input_tensor[i] = img_np

                # 2. Run Inference
                input_name = self._session.get_inputs()[0].name
                features = self._session.run(None, {input_name: input_tensor})[0]

                # 3. L2 Normalize
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                return features / (norms + 1e-6)

            except Exception as e:
                logger.error(f"ONNX Batch extraction failed: {e}")
                return np.zeros((len(images), self.OUTPUT_DIM))

        # B. Torch Inference (Fallback)
        elif HAS_TORCH and self._torch_model:
            try:
                batch_tensors = []
                device = next(self._torch_model.parameters()).device
                for img in images:
                    img = img.convert("RGB")
                    batch_tensors.append(self._torch_preprocess(img))

                input_batch = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    features = self._torch_model(input_batch)
                    features = features.cpu().numpy()

                # L2 Normalize
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                return features / (norms + 1e-6)

            except Exception as e:
                logger.error(f"Torch Batch extraction failed: {e}")
                return np.zeros((len(images), self.OUTPUT_DIM))

        else:
            return np.zeros((len(images), self.OUTPUT_DIM))


class HybridCluster(Clusterer):
    def __init__(self):
        self.geod = Geod(ellps="WGS84")
        self.params = PARAMS

        # Clustering Params
        self.gps_eps = self.params.get("eps", 7.48)
        self.visual_split_thresh = self.params.get("loose_thresh", 0.64)
        self.split_min_size = 8
        self.min_cluster_size = self.params.get("min_cluster_size", 2)
        self.min_samples = self.params.get("min_samples", 1)
        self.max_cluster_size = self.params.get("max_cluster_size", 4)

        # Initialize Optimized Extractor
        self.extractor = CosPlaceExtractor(onnx_path="img_models/cosplace_resnet50_int8.onnx")
        self.storage = GCSStorageService()
        self.performance_monitor = PerformanceMonitor()

        logger.info(f"HybridCluster Ready. Mode: {'ONNX' if self.extractor.use_onnx else 'Torch/None'}")

    async def cluster(self, photos: List[PhotoMeta]) -> List[List[PhotoMeta]]:
        if not photos:
            return []

        logger.info(f"HybridCluster Mode: {'ONNX' if self.extractor.use_onnx else 'Torch/None'}")
        self.performance_monitor.start()

        # 1. Preprocessing (GPS Correction)
        self._adjust_gps_inaccuracy(photos)
        self._correct_outliers_by_speed(photos)

        # 2. Extract Features (Optimized Pipeline)
        logger.info(f"Start feature extraction for {len(photos)} photos.")
        features_matrix = await self._extract_features_optimized(photos)

        # 3. Clustering Logic
        labels = self._run_clustering_logic(photos, features_matrix)

        # 4. Result Grouping
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
        self.performance_monitor.report("HybridCluster", count=len(photos))

        return result

    async def _extract_features_optimized(self, photos: List[PhotoMeta]) -> List[Optional[np.ndarray]]:
        """
        Batch Size만큼 다운로드 -> 메모리 로드 -> Batch 추론 -> 메모리 해제
        반환값: 각 사진에 해당하는 Feature Vector (List 순서 보장)
        """
        n_photos = len(photos)
        # 최종 결과를 담을 리스트 (크기 미리 할당)
        final_features = [None] * n_photos

        # OOM 방지를 위한 배치 설정
        BATCH_SIZE = 16
        semaphore = asyncio.Semaphore(10)  # 동시 다운로드 수 제한

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i in range(0, n_photos, BATCH_SIZE):
                batch_indices = range(i, min(i + BATCH_SIZE, n_photos))
                batch_photos = [photos[idx] for idx in batch_indices]

                # A. 병렬 다운로드
                download_tasks = []
                local_files_map = {}  # idx -> file_path

                for idx, p in zip(batch_indices, batch_photos):
                    target_path = p.thumbnail_path or p.path
                    if not target_path:
                        # 로컬 경로가 있다면 사용
                        if os.path.exists(p.path):
                            local_files_map[idx] = Path(p.path)
                        continue

                    # 확장자 추출 및 임시 파일명
                    ext = os.path.splitext(target_path)[1].split("?")[0] or ".jpg"
                    dest_path = temp_path / f"{idx}{ext}"
                    local_files_map[idx] = dest_path

                    download_tasks.append(self._download_safe(target_path, dest_path, semaphore))

                if download_tasks:
                    await asyncio.gather(*download_tasks)

                # B. 메모리에 이미지 로드 (Batch 준비)
                valid_indices = []
                pil_images = []

                for idx in batch_indices:
                    if idx in local_files_map and local_files_map[idx].exists():
                        try:
                            with Image.open(local_files_map[idx]) as img:
                                # [Memory Optimization] Resize immediately to (640, 480)
                                # This prevents OOM when holding multiple high-res photos in 'pil_images'
                                img_resized = img.convert("RGB").resize((640, 480))
                                pil_images.append(img_resized)
                                valid_indices.append(idx)
                        except Exception as e:
                            logger.warning(f"Failed to open image {idx}: {e}")

                # C. 배치 추론 (Vectorized Inference)
                if pil_images:
                    batch_features = self.extractor.extract_batch(pil_images)

                    # 결과 매핑
                    for sub_idx, feature in enumerate(batch_features):
                        original_idx = valid_indices[sub_idx]
                        final_features[original_idx] = feature

                # D. 중요: 메모리 및 디스크 정리
                del pil_images
                del batch_photos
                # 임시 파일 삭제
                for fpath in local_files_map.values():
                    if fpath.exists():
                        fpath.unlink(missing_ok=True)
                
                gc.collect()

        return final_features

    async def _download_safe(self, url: str, dest: Path, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                await self.storage.download_file(url, dest)
            except Exception as e:
                logger.warning(f"Download error {url}: {e}")

    def _run_clustering_logic(
        self, photos: List[PhotoMeta], features: List[Optional[np.ndarray]], skip_gps: bool = False
    ) -> np.ndarray:

        n_samples = len(photos)
        # None Feature가 있는지 확인하고 필터링하거나 0벡터 처리 (여기서는 인덱스 유지를 위해 None 체크)
        valid_mask = [f is not None for f in features]
        if not any(valid_mask):
            return np.full(n_samples, -1)

        # HDBSCAN 사용 불가 시
        if HDBSCAN is None:
            return np.full(n_samples, -1)

        # GPS 결측치 확인
        has_full_gps = all(p.lat is not None and p.lon is not None for p in photos)
        if not has_full_gps:
            skip_gps = True

        # --- [Step 1] GPS Clustering ---
        labels = np.zeros(n_samples, dtype=int)

        if not skip_gps:
            gps_matrix = self._compute_gps_matrix(photos)
            try:
                # precomputed metric을 사용할 때는 거리 행렬이 유효해야 함
                gps_clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric="precomputed",
                    cluster_selection_epsilon=self.gps_eps,
                    allow_single_cluster=True,
                )
                labels = gps_clusterer.fit_predict(gps_matrix)
            except Exception as e:
                logger.error(f"GPS Clustering failed: {e}")
                labels = np.full(n_samples, -1)

        # --- [Step 2] Visual Split (Recursive HDBSCAN) ---
        # Feature List를 Numpy Array로 변환 (결측치는 0으로 채우거나 제외해야 함)
        # 여기서는 편의상 전체 Matrix를 만들지 않고 클러스터별로 slice해서 처리

        max_label = labels.max()
        next_label_id = max_label + 1

        unique_labels = set(labels) - {-1}

        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]

            # 클러스터 크기가 너무 작으면 스킵
            if len(indices) <= self.split_min_size:
                continue

            # 해당 클러스터의 Feature 추출
            sub_features = []
            valid_sub_indices = []

            for idx in indices:
                if features[idx] is not None:
                    sub_features.append(features[idx])
                    valid_sub_indices.append(idx)

            if len(sub_features) < 2:
                continue

            sub_features_arr = np.array(sub_features)  # (N, 512)

            # 거리 행렬 계산 (Euclidean)
            # HDBSCAN에 feature matrix 직접 넣어도 되지만, metric='euclidean' 명시
            try:
                sub_clusterer = HDBSCAN(
                    min_cluster_size=2,
                    min_samples=2,  # 민감도 조정
                    metric="euclidean",
                    cluster_selection_epsilon=self.visual_split_thresh,
                    allow_single_cluster=False,
                )
                sub_labels = sub_clusterer.fit_predict(sub_features_arr)

                # 서브 클러스터 라벨링 적용
                found_subs = set(sub_labels) - {-1}
                for sub_id in found_subs:
                    # 원본 indices 중, 현재 sub_id에 해당하는 것들만 필터링
                    # 주의: sub_labels 길이는 valid_sub_indices 길이와 같음
                    mask = sub_labels == sub_id
                    target_real_indices = np.array(valid_sub_indices)[mask]

                    labels[target_real_indices] = next_label_id
                    next_label_id += 1

            except Exception as e:
                logger.warning(f"Visual split failed for cluster {cluster_id}: {e}")

        # --- [Step 3] Force Split (K-Means) ---
        # 최대 크기 초과시 강제 분할
        unique_labels = set(labels) - {-1}
        for cluster_id in list(unique_labels):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) > self.max_cluster_size:

                # Feature 수집
                sub_features = []
                valid_sub_indices = []
                for idx in indices:
                    if features[idx] is not None:
                        sub_features.append(features[idx])
                        valid_sub_indices.append(idx)

                if len(sub_features) < self.max_cluster_size:
                    continue

                n_splits = math.ceil(len(valid_sub_indices) / self.max_cluster_size)

                try:
                    kmeans = KMeans(n_clusters=n_splits, n_init=10, random_state=42)
                    sub_labels = kmeans.fit_predict(np.stack(sub_features))

                    for k in range(n_splits):
                        mask = sub_labels == k
                        target_real_indices = np.array(valid_sub_indices)[mask]
                        labels[target_real_indices] = next_label_id
                        next_label_id += 1

                except Exception as e:
                    logger.warning(f"KMeans split failed for cluster {cluster_id}: {e}")

        return labels

    def _compute_gps_matrix(self, photos: List[PhotoMeta]) -> np.ndarray:
        n = len(photos)
        coords = np.array([[p.lat or 0.0, p.lon or 0.0] for p in photos])

        # Geod는 벡터화 연산 지원이 제한적이므로 loop 유지하되, 메모리 최적화를 위해 필요시 chunking
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                _, _, dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix

    def _correct_outliers_by_speed(self, photos: List[PhotoMeta]) -> None:
        """GPS 튀는 현상 보정"""
        # (기존 로직 유지)
        timed_photos = [p for p in photos if p.timestamp is not None and p.lat is not None]
        timed_photos.sort(key=lambda x: x.timestamp)
        max_speed_mps = 5.0  # 조금 더 완화

        for i in range(1, len(timed_photos)):
            prev, curr = timed_photos[i - 1], timed_photos[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                continue

            _, _, dist = self.geod.inv(prev.lon, prev.lat, curr.lon, curr.lat)
            if (dist / dt) > max_speed_mps:
                curr.lat, curr.lon = prev.lat, prev.lon
                if prev.alt:
                    curr.alt = prev.alt

    def _adjust_gps_inaccuracy(self, photos: List[PhotoMeta]) -> None:
        """짧은 시간 내 이동 보정"""
        # (기존 로직 유지)
        timed_photos = [p for p in photos if p.timestamp is not None]
        timed_photos.sort(key=lambda x: x.timestamp)

        for i in range(len(timed_photos) - 2, -1, -1):
            p1, p2 = timed_photos[i], timed_photos[i + 1]
            if 0 <= (p2.timestamp - p1.timestamp) <= 20:
                if p2.lat is not None:
                    p1.lat, p1.lon = p2.lat, p2.lon
