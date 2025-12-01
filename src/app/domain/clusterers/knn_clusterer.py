#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new_deep_clusterer.py

공원 사진 자동 분류기 (고정밀 딥러닝 + 기하 검증 버전)

기존 deep_clusterer.DeepClusterer 와 완전히 호환되는 인터페이스를 유지하면서,
다음과 같은 점을 강화한 버전입니다.

- CLIP + EfficientNet + ViT + 전통 특징을 결합한 고차원 특징 벡터
- k-NN 기반 전역 임베딩 유사도
- SIFT + RANSAC 를 이용한 기하학적 검증(같은 장소 여부 판별 강화)
- 연결 요소 기반 클러스터링 + 품질 스코어(유사도 × 개수)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 유틸: 로컬 특징 + RANSAC 기반 기하학적 유사도
# ---------------------------------------------------------------------------

try:
    import cv2
except Exception:  # pragma: no cover - 환경에 따라 cv2가 없을 수도 있음
    cv2 = None  # type: ignore


class LocalGeometryMatcher:
    """
    두 이미지 사이의 기하학적 일관성을 SIFT + RANSAC 으로 평가하는 도우미 클래스.
    """

    def __init__(
        self,
        max_features: int = 2000,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 5.0,
        min_good_matches: int = 10,
    ) -> None:
        self.enabled = cv2 is not None
        if not self.enabled:
            logger.warning(
                "⚠️ OpenCV(cv2)를 찾을 수 없어 기하학적 검증 단계는 비활성화됩니다. "
                "pip install opencv-python-headless 로 설치할 수 있습니다."
            )
            self.detector = None
        else:
            # SIFT는 opencv-contrib-python 필요
            try:
                self.detector = cv2.SIFT_create(nfeatures=max_features)  # type: ignore[attr-defined]
            except Exception:
                self.detector = None
                self.enabled = False
                logger.warning(
                    "⚠️ SIFT 생성 실패. opencv-contrib-python 이 설치되어 있는지 확인하세요. "
                    "기하학적 검증 단계는 비활성화됩니다."
                )

        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_good_matches = min_good_matches

    def _load_gray(self, path: Path) -> Optional[np.ndarray]:
        if not self.enabled or self.detector is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # type: ignore[operator]
        if img is None:
            return None
        return img

    def geo_score(self, path1: Path, path2: Path) -> float:
        """
        0.0 ~ 1.0 사이의 기하학적 일관성 점수.
        - 0.0 에 가까울수록 구조가 다르거나 매칭 실패
        - 1.0 에 가까울수록 구조가 상당히 일치
        """
        if not self.enabled or self.detector is None:
            return 1.0  # 기하 검증을 사용할 수 없는 환경에서는 항상 통과

        img1 = self._load_gray(path1)
        img2 = self._load_gray(path2)
        if img1 is None or img2 is None:
            return 0.0

        keypoints1, desc1 = self.detector.detectAndCompute(img1, None)
        keypoints2, desc2 = self.detector.detectAndCompute(img2, None)
        if desc1 is None or desc2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # type: ignore[attr-defined]
        matches = bf.knnMatch(desc1, desc2, k=2)  # type: ignore[attr-defined]

        good = []
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return 0.0

        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(  # type: ignore[attr-defined]
            pts1, pts2, cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_thresh  # type: ignore[attr-defined]
        )
        if H is None or mask is None:
            return 0.0

        inliers = mask.ravel().sum()
        total = len(good)
        if total == 0:
            return 0.0
        score = float(inliers) / float(total)
        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 사람 감지용 DETR 래퍼 (기존 deep_clusterer 와 동일한 인터페이스 유지)
# ---------------------------------------------------------------------------

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
except Exception:  # pragma: no cover
    DetrForObjectDetection = None  # type: ignore
    DetrImageProcessor = None  # type: ignore


class PeopleDetector:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device

    def __call__(self, images, return_tensors: str = "pt"):
        inputs = self.processor(images=images, return_tensors=return_tensors)
        import torch  # 지연 임포트

        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    def post_process_object_detection(self, outputs, **kwargs):
        return self.processor.post_process_object_detection(outputs, **kwargs)


def create_people_detector(device):
    """사람 감지 모델 초기화 (없으면 None 반환)."""
    if DetrForObjectDetection is None or DetrImageProcessor is None:
        logger.warning(
            "⚠️ transformers / DETR 를 찾을 수 없습니다. 사람 제거 기능이 비활성화됩니다."
        )
        return None

    try:
        logger.info("🧍 사람 감지 모델 로딩 중 (DETR ResNet-50)...")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        import torch  # 지연 임포트

        model = model.to(device)
        model.eval()
        return PeopleDetector(processor, model, device)
    except Exception as e:  # pragma: no cover - 환경 의존
        logger.warning(f"⚠️ 사람 감지 모델 로드 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# 딥러닝 기반 클러스터러 (기존 DeepClusterer 와 동일한 이름 / 메서드 시그니처)
# ---------------------------------------------------------------------------


class DeepClusterer:
    def __init__(
        self,
        input_path,
        similarity_threshold: float = 0.6,
        use_cache: bool = True,
        remove_people: bool = True,
    ):
        """
        Args:
            input_path: 이미지들이 들어 있는 디렉터리 경로
            similarity_threshold: 전역 임베딩 코사인 유사도 임계값 (0~1)
            use_cache: 특징 벡터 캐시 사용 여부
            remove_people: 사람 영역 마스킹 여부
        """
        from pathlib import Path as _Path

        import torch

        self.input_path = _Path(input_path)
        self.output_path = self.input_path / "advanced"
        self.cache_dir = self.input_path / ".photo_cache"
        self.similarity_threshold = similarity_threshold
        self.use_cache = use_cache
        self.remove_people = remove_people

        self.photos: List[_Path] = []
        self.groups: List[Dict] = []

        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"💾 Cache directory: {self.cache_dir}")

        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Mac (Apple Silicon) + Metal GPU
            self.device = torch.device("mps")
            logger.info("🔧 Using device: mps (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            # NVIDIA GPU (Linux/Windows 등)
            self.device = torch.device("cuda")
            logger.info("🔧 Using device: cuda (NVIDIA GPU)")
        else:
            self.device = torch.device("cpu")
            logger.info("🔧 Using device: cpu")
            logger.info(f"🔧 Using device: {self.device}")

        # 캐시 통계
        if self.use_cache:
            self.cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}

        # 모델들
        self.clip_model = None
        self.clip_preprocess = None
        self.efficientnet = None
        self.vit = None
        self.people_detector: Optional[PeopleDetector] = None

        # 기하 검증기
        self.geo_matcher = LocalGeometryMatcher()

        self.setup_models()

    # ------------------------------------------------------------------
    # 모델 로딩
    # ------------------------------------------------------------------
    def setup_models(self):
        """CLIP + EfficientNet + ViT 모델들을 초기화한다."""
        logger.info("🤖 Loading vision models...")

        try:
            import open_clip
            import timm
            import torch

            # 사람 제거 모델
            if self.remove_people:
                self.people_detector = create_people_detector(self.device)

            # OpenCLIP (ViT-B-32)
            logger.info("   📥 Loading OpenCLIP ViT-B-32...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.clip_model.to(self.device)
            self.clip_model.eval()

            # EfficientNet (feature extractor)
            logger.info("   📥 Loading EfficientNet-B4...")
            self.efficientnet = timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=0
            ).to(self.device)
            self.efficientnet.eval()

            # ViT (feature extractor)
            logger.info("   📥 Loading ViT-B-16...")
            self.vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            ).to(self.device)
            self.vit.eval()

            logger.info("✅ All models loaded successfully!")

        except Exception as e:  # pragma: no cover - 설치 환경에 따라
            logger.error(f"❌ Model loading failed: {e}")
            logger.info(
                "💡 Please install required libraries with:\n"
                "   pip install torch torchvision timm open_clip_torch transformers opencv-python-headless"
            )
            raise

    # ------------------------------------------------------------------
    # 캐시 관련
    # ------------------------------------------------------------------
    def get_file_hash(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_cache_path(self, file_path: Path, feature_type: str) -> Path:
        file_hash = self.get_file_hash(file_path)
        cache_filename = f"{file_hash}_{feature_type}.pkl"
        return self.cache_dir / cache_filename

    def load_from_cache(self, file_path: Path, feature_type: str):
        if not self.use_cache:
            return None

        cache_path = self.get_cache_path(file_path, feature_type)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    features = pickle.load(f)
                if self.use_cache:
                    self.cache_stats["hits"] += 1
                return features
            except Exception as e:
                logger.warning(f"⚠️ Cache load failed ({cache_path}): {e}")
        if self.use_cache:
            self.cache_stats["misses"] += 1
        return None

    def save_to_cache(self, file_path: Path, feature_type: str, features):
        if not self.use_cache:
            return
        cache_path = self.get_cache_path(file_path, feature_type)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.warning(f"⚠️ Cache save failed: {e}")

    def clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("🗑️ Cache cleared.")

    # ------------------------------------------------------------------
    # 이미지 로딩 및 전처리
    # ------------------------------------------------------------------
    def load_photos(self) -> List[Path]:
        """입력 폴더에서 이미지 파일 목록을 수집한다."""
        if self.photos:
            # cluster(photo_paths) 로 전달된 리스트가 있으면 그것을 사용
            logger.info(f"📷 Using {len(self.photos)} pre-specified photo paths.")
            return self.photos

        photo_files: List[Path] = []
        for root, _, files in os.walk(self.input_path):
            root_path = Path(root)
            # 결과 폴더/캐시 폴더는 제외
            if root_path == self.output_path or self.cache_dir in root_path.parents:
                continue
            for name in files:
                ext = Path(name).suffix.lower()
                if ext in self.image_extensions:
                    photo_files.append(root_path / name)

        photo_files.sort()
        logger.info(f"📷 Found {len(photo_files)} image files.")
        return photo_files

    # ------------------------------------------------------------------
    # 사람 제거 (선택적)
    # ------------------------------------------------------------------
    def mask_people(self, image: Image.Image) -> Image.Image:
        """
        DETR 기반으로 사람 박스를 찾아 흐리게 처리.
        DETR 모델이 없으면 원본 이미지를 그대로 반환.
        """
        if not self.remove_people or self.people_detector is None:
            return image

        try:
            import torch

            inputs = self.people_detector([image], return_tensors="pt")
            outputs = self.people_detector.model(**inputs)
            target_sizes = torch.tensor([[image.height, image.width]]).to(self.device)
            results = self.people_detector.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]

            person_boxes: List[List[float]] = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if float(score) < 0.8:
                    continue
                # COCO dataset 에서 person class id == 1
                if int(label) == 1:
                    person_boxes.append([float(x) for x in box.tolist()])

            if not person_boxes:
                return image

            img_np = np.array(image).copy()
            for x_min, y_min, x_max, y_max in person_boxes:
                x_min_i = max(0, int(x_min))
                y_min_i = max(0, int(y_min))
                x_max_i = min(img_np.shape[1], int(x_max))
                y_max_i = min(img_np.shape[0], int(y_max))

                roi = img_np[y_min_i:y_max_i, x_min_i:x_max_i]
                if roi.size == 0:
                    continue
                roi_blur = cv2.GaussianBlur(roi, (21, 21), 0) if cv2 is not None else roi
                img_np[y_min_i:y_max_i, x_min_i:x_max_i] = roi_blur

            return Image.fromarray(img_np)
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ Failed to mask people: {e}")
            return image

    # ------------------------------------------------------------------
    # 특징 추출
    # ------------------------------------------------------------------
    def extract_clip_features(self, image: Image.Image, file_path: Path) -> Optional[np.ndarray]:
        cached = self.load_from_cache(file_path, "clip")
        if cached is not None:
            return cached

        if self.clip_model is None or self.clip_preprocess is None:
            return None

        import torch

        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            feat_np: np.ndarray = features.cpu().numpy().flatten()
            self.save_to_cache(file_path, "clip", feat_np)
            return feat_np

    def extract_efficientnet_features(
        self, image: Image.Image, file_path: Path
    ) -> Optional[np.ndarray]:
        cached = self.load_from_cache(file_path, "efficientnet")
        if cached is not None:
            return cached
        if self.efficientnet is None:
            return None

        import torch
        import torchvision.transforms as T

        preprocess = T.Compose(
            [
                T.Resize((380, 380)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        img_tensor = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.efficientnet(img_tensor)
            feat_np: np.ndarray = feats.cpu().numpy().flatten()
            self.save_to_cache(file_path, "efficientnet", feat_np)
            return feat_np

    def extract_vit_features(self, image: Image.Image, file_path: Path) -> Optional[np.ndarray]:
        cached = self.load_from_cache(file_path, "vit")
        if cached is not None:
            return cached
        if self.vit is None:
            return None

        import torch
        import torchvision.transforms as T

        preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        img_tensor = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.vit(img_tensor)
            feat_np: np.ndarray = feats.cpu().numpy().flatten()
            self.save_to_cache(file_path, "vit", feat_np)
            return feat_np

    def extract_traditional_features(self, image_path: Path) -> Optional[np.ndarray]:
        """
        색 히스토그램 + LBP 기반 간단한 전통 특징.
        """
        cached = self.load_from_cache(image_path, "traditional")
        if cached is not None:
            return cached

        try:
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)

            # 색 히스토그램 (각 채널 32-bin)
            hist_features: List[float] = []
            for ch in range(3):
                hist = cv2.calcHist(  # type: ignore[operator]
                    [img_np], [ch], None, [32], [0, 256]
                )
                hist = cv2.normalize(hist, hist).flatten()  # type: ignore[operator]
                hist_features.extend(hist.tolist())

            # LBP
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if cv2 is not None else np.array(
                img.convert("L")
            )  # type: ignore[operator]
            lbp = self.calculate_lbp(gray, radius=1, n_points=8)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256), density=True)
            hist_features.extend(lbp_hist.flatten().tolist())

            feat_np = np.array(hist_features, dtype=np.float32)
            self.save_to_cache(image_path, "traditional", feat_np)
            return feat_np
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ Failed to extract traditional features ({image_path}): {e}")
            return None

    def calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        rows, cols = image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                code = 0
                for p in range(n_points):
                    theta = 2.0 * np.pi * p / n_points
                    y = i + int(round(radius * np.sin(theta)))
                    x = j + int(round(radius * np.cos(theta)))
                    neighbor = image[y, x]
                    code |= (1 << p) if neighbor > center else 0
                lbp[i, j] = code

        return lbp

    # ------------------------------------------------------------------
    # 딥 특징 결합
    # ------------------------------------------------------------------
    def extract_deep_features(
        self, image_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        하나의 이미지에 대해 CLIP / EfficientNet / ViT / 전통 특징을 모두 추출하고
        가중 합한 하나의 벡터를 반환한다.
        """
        cached = self.load_from_cache(image_path, "combined")
        if cached is not None:
            # combined 만 캐시된 경우를 대비해 dict 는 None 으로 둔다
            return cached, None

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"⚠️ Failed to open image {image_path}: {e}")
            return None, None

        # 사람 제거 (선택)
        img_for_features = self.mask_people(img)

        clip_f = self.extract_clip_features(img_for_features, image_path)
        eff_f = self.extract_efficientnet_features(img_for_features, image_path)
        vit_f = self.extract_vit_features(img_for_features, image_path)
        trad_f = self.extract_traditional_features(image_path)

        # 어떤 특징이라도 하나는 있어야 함
        if all(f is None for f in (clip_f, eff_f, vit_f, trad_f)):
            return None, None

        parts = []
        features_dict: Dict[str, np.ndarray] = {}

        def _norm_or_none(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if x is None:
                return None
            norm = np.linalg.norm(x)
            if norm == 0:
                return None
            return x / norm

        clip_n = _norm_or_none(clip_f)
        eff_n = _norm_or_none(eff_f)
        vit_n = _norm_or_none(vit_f)
        trad_n = _norm_or_none(trad_f)

        if clip_n is not None:
            parts.append(clip_n * 0.45)
            features_dict["clip"] = clip_n
        if eff_n is not None:
            parts.append(eff_n * 0.30)
            features_dict["efficientnet"] = eff_n
        if vit_n is not None:
            parts.append(vit_n * 0.20)
            features_dict["vit"] = vit_n
        if trad_n is not None:
            parts.append(trad_n * 0.05)
            features_dict["traditional"] = trad_n

        combined = np.concatenate(parts).astype(np.float32)
        self.save_to_cache(image_path, "combined", combined)
        return combined, features_dict

    # ------------------------------------------------------------------
    # 고급 클러스터링 (전역 임베딩 + 기하 검증 + 그래프 연결요소)
    # ------------------------------------------------------------------
    def _connected_components(self, n: int, edges: List[Tuple[int, int]]) -> List[int]:
        if n == 0:
            return []
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in edges:
            union(i, j)

        roots = [find(i) for i in range(n)]
        root_to_label: Dict[int, int] = {}
        next_label = 0
        labels: List[int] = []
        for r in roots:
            if r not in root_to_label:
                root_to_label[r] = next_label
                next_label += 1
            labels.append(root_to_label[r])
        return labels

    def advanced_clustering(self, features_array: np.ndarray, photo_files: List[Path]) -> List[Dict]:
        """
        기존 deep_clusterer 의 advanced_clustering 과 동일한 역할을 수행하되,
        - 전역 임베딩 기반 k-NN 그래프
        - SIFT + RANSAC 기하 검증
        - 연결 요소 기반 클러스터링
        을 사용하여 같은 장소 사진을 더욱 정밀하게 묶는다.
        """
        n, d = features_array.shape
        if n == 0:
            return []

        logger.info("📌 Building k-NN graph in feature space...")
        k = min(max(10, int(np.sqrt(n)) + 1), n)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(features_array)
        distances, indices = nn.kneighbors(features_array)

        edges: List[Tuple[int, int]] = []
        geo_threshold = 0.2  # 기하학 일관성 최소 비율

        for i in range(n):
            for dist, j in zip(distances[i][1:], indices[i][1:]):  # [0] 은 자기 자신
                sim = 1.0 - float(dist)  # cosine distance -> similarity
                if sim < self.similarity_threshold:
                    continue

                # 기하 검증
                score_geo = self.geo_matcher.geo_score(photo_files[i], photo_files[j])
                if score_geo < geo_threshold:
                    continue

                edges.append((i, j))

        logger.info(f"🔗 Retained {len(edges)} edges after geometric verification.")

        if not edges:
            # 엣지가 하나도 없으면 각 이미지를 별도 그룹으로
            groups: List[Dict] = []
            for idx, p in enumerate(photo_files):
                groups.append(
                    {
                        "id": idx,
                        "photos": [p],
                        "count": 1,
                        "avg_similarity": 1.0,
                        "quality_score": 1.0,
                    }
                )
            return groups

        labels = self._connected_components(n, edges)

        # 라벨 -> 인덱스 리스트
        label_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            label_to_indices.setdefault(lbl, []).append(idx)

        groups: List[Dict] = []
        for label, idxs in label_to_indices.items():
            group_photos = [photo_files[i] for i in idxs]
            group_feats = features_array[idxs]

            # 클러스터 중심과의 평균 코사인 유사도
            centroid = group_feats.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            sims = group_feats @ centroid
            avg_similarity = float(np.mean(sims))

            groups.append(
                {
                    "id": label,
                    "photos": group_photos,
                    "count": len(group_photos),
                    "avg_similarity": avg_similarity,
                    "quality_score": avg_similarity * len(group_photos),
                }
            )

        # 품질 점수 기준 내림차순 정렬
        groups.sort(key=lambda g: g["quality_score"], reverse=True)
        return groups

    # ------------------------------------------------------------------
    # 파이프라인 엔트리 포인트
    # ------------------------------------------------------------------
    def cluster_photos(self):
        logger.info("🔍 Starting deep learning-based photo analysis...")
        photo_files = self.load_photos()
        if len(photo_files) < 2:
            logger.warning("❌ Not enough images to analyze.")
            return

        features_list: List[np.ndarray] = []
        valid_photos: List[Path] = []

        logger.info("🚀 Extracting high-dimensional features... (caching may speed this up)")
        start_time = time.time()
        for photo_file in tqdm(photo_files, desc="Extracting deep features"):
            combined_features, _features_dict = self.extract_deep_features(photo_file)
            if combined_features is not None:
                features_list.append(combined_features)
                valid_photos.append(photo_file)

        extraction_time = time.time() - start_time
        logger.info(
            f"✅ Extracted features from {len(valid_photos)} images in {extraction_time:.1f}s"
        )

        if self.use_cache and hasattr(self, "cache_stats"):
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = (
                self.cache_stats["hits"] / total_requests * 100 if total_requests > 0 else 0.0
            )
            logger.info(
                f"💾 Cache stats: {self.cache_stats['hits']} hits, "
                f"{self.cache_stats['misses']} misses ({hit_rate:.1f}% hit rate)"
            )

        if not features_list:
            logger.warning("❌ No valid features extracted; aborting clustering.")
            return

        features_array = np.stack(features_list, axis=0)
        logger.info(f"📊 Feature vector dimensions: {features_array.shape}")

        self.groups = self.advanced_clustering(features_array, valid_photos)
        logger.info(f"✅ Clustered into {len(self.groups)} groups with high precision.")
        for group in self.groups:
            quality_desc = (
                "High"
                if group["quality_score"] > 2
                else "Medium"
                if group["quality_score"] > 1
                else "Low"
            )
            logger.info(
                f"   📍 Group {group['id']}: {group['count']} photos, "
                f"avg similarity: {group['avg_similarity']:.3f}, "
                f"Quality: {quality_desc}"
            )

    def cluster(self, photo_paths: List[str]) -> List[List[str]]:
        """
        Clusters a given list of photo paths.
        This is the main entry point when used as part of a pipeline.
        (기존 deep_clusterer 와 동일한 시그니처 유지)
        """
        self.photos = [Path(p) for p in photo_paths]
        self.cluster_photos()

        sub_clusters: List[List[str]] = []
        if self.groups:
            for group in self.groups:
                sub_clusters.append([str(photo) for photo in group["photos"]])
        return sub_clusters

    def run(self):
        """
        입력 폴더(self.input_path) 전체를 대상으로 클러스터링하는 편의 메서드.
        """
        self.photos = []  # 폴더 전체 사용
        self.create_output_folders()
        self.cluster_photos()
        self.copy_photos_to_groups()
        self.create_master_result_image()
        self.create_detailed_report()

    # ------------------------------------------------------------------
    # 결과 저장 / 시각화
    # ------------------------------------------------------------------
    def create_output_folders(self):
        if self.output_path.exists():
            logger.info("🗑️ Deleting existing output folder...")
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created output folder: {self.output_path}")

    def copy_photos_to_groups(self):
        logger.info("📋 Copying photos to group folders...")
        for group in tqdm(self.groups, desc="Creating folders"):
            group_folder = self.output_path / f"location_{group['id']}"
            group_folder.mkdir(exist_ok=True)
            for photo_path in group["photos"]:
                shutil.copy2(photo_path, group_folder / photo_path.name)

    def create_master_result_image(self):
        logger.info("🎨 Creating master result image...")
        if not self.groups:
            return

        cell_width, cell_height, header_height, padding, cols = 300, 250, 40, 10, 3
        rows = (len(self.groups) + cols - 1) // cols
        canvas_width = cols * (cell_width + padding) - padding
        canvas_height = rows * (cell_height + header_height + padding) - padding
        master_image = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(master_image)

        try:
            from PIL import ImageFont

            try:
                font_dir = Path("/app/fonts")
                font = ImageFont.truetype(str(font_dir / "AppleGothic.ttf"), 16)
            except Exception:
                font = ImageFont.load_default()
        except Exception:
            ImageFont = None  # type: ignore
            font = None  # type: ignore

        for idx, group in enumerate(self.groups):
            row = idx // cols
            col = idx % cols
            group_x = col * (cell_width + padding)
            group_y = row * (cell_height + header_height + padding)

            # 헤더 텍스트
            header_text = (
                f"Group {group['id']}  "
                f"(n={group['count']}, "
                f"sim={group['avg_similarity']:.2f})"
            )
            if font is not None:
                draw.text((group_x + 5, group_y + 5), header_text, fill="black", font=font)
            else:
                draw.text((group_x + 5, group_y + 5), header_text, fill="black")

            # 사진들(최대 3장)
            photos_to_show = group["photos"][:3]
            photo_width = cell_width // 3
            for photo_idx, photo_path in enumerate(photos_to_show):
                try:
                    img = Image.open(photo_path).resize(
                        (photo_width - 2, cell_height - 2), Image.Resampling.LANCZOS
                    )
                    photo_x = group_x + photo_idx * photo_width + 1
                    photo_y = group_y + header_height + 1
                    master_image.paste(img, (photo_x, photo_y))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process photo {photo_path}: {e}")

        master_path = self.output_path / "classification_result.jpg"
        master_image.save(master_path, quality=95, optimize=True)
        logger.info(f"✅ Master result image saved: {master_path}")
        return master_path

    def create_detailed_report(self):
        summary = {
            "analysis_info": {
                "model_used": ["OpenCLIP ViT-B-32", "EfficientNet-B4", "Vision Transformer"],
                "device": str(self.device),
                "similarity_threshold": self.similarity_threshold,
            },
            "total_photos": sum(g["count"] for g in self.groups),
            "total_groups": len(self.groups),
            "groups": [
                {
                    "id": str(g["id"]),
                    "photo_count": g["count"],
                    "average_similarity": float(g["avg_similarity"]),
                    "quality_score": float(g["quality_score"]),
                    "photos": [p.name for p in g["photos"]],
                }
                for g in self.groups
            ],
        }

        self.output_path.mkdir(parents=True, exist_ok=True)
        report_path = self.output_path / "analysis_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"📊 Detailed report saved: {report_path}")