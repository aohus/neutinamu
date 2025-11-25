#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공원 사진 자동 분류기 (딥러닝 버전)
- CLIP + EfficientNet을 활용한 고정밀 이미지 특징 추출
- 같은 장소의 사진들을 자동으로 그룹핑
- 조도, 밝기, 계절 변화, 구도 변화 모두 고려
"""

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
except ImportError:
    print("DETR 모델을 위한 transformers 라이브러리가 필요합니다.")
import hashlib
import json
import logging
import os
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open_clip as clip
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DeepClusterer:
    def __init__(self, input_path, similarity_threshold=0.1, use_cache=True, remove_people=True):
        self.input_path = Path(input_path)
        self.output_path = self.input_path / "advanced"
        self.cache_dir = self.input_path / ".photo_cache"
        self.similarity_threshold = similarity_threshold
        self.use_cache = use_cache
        self.remove_people = remove_people
        self.people_detector = None
        self.photos: List[Path] = []
        self.groups = []
        
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            logger.info(f"💾 Cache directory: {self.cache_dir}")

        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {self.device}")

        self.setup_models()

    def setup_models(self):
        """딥러닝 모델들 초기화"""
        logger.info("🤖 Loading deep learning models...")

        try:
            if self.remove_people:
                self.people_detector = create_people_detector(self.device)

            logger.info("   📥 Loading OpenCLIP model...")
            self.clip_model, _, self.clip_preprocess = clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.device
            )
            self.clip_model.eval()

            logger.info("   📥 Loading EfficientNet model...")
            self.efficientnet = timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=0
            )
            self.efficientnet = self.efficientnet.to(self.device)
            self.efficientnet.eval()

            self.efficientnet_transform = transforms.Compose(
                [
                    transforms.Resize((380, 380)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info("   📥 Loading Vision Transformer...")
            self.vit_model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
            self.vit_model = self.vit_model.to(self.device)
            self.vit_model.eval()

            self.vit_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )


        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.info("💡 Please install required libraries with:")
            logger.info("pip install torch torchvision timm open_clip_torch")
            raise

        logger.info("✅ All models loaded successfully!")

        if self.use_cache:
            self.cache_stats = {"hits": 0, "misses": 0}

    def get_file_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
            hash_md5.update(chunk)
            stat = os.stat(file_path)
            hash_md5.update(str(stat.st_size).encode())
            hash_md5.update(str(stat.st_mtime).encode())
        return hash_md5.hexdigest()

    def get_cache_path(self, file_path, feature_type):
        file_hash = self.get_file_hash(file_path)
        cache_filename = f"{file_hash}_{feature_type}.pkl"
        return self.cache_dir / cache_filename

    def load_from_cache(self, file_path, feature_type):
        if not self.use_cache:
            return None

        cache_path = self.get_cache_path(file_path, feature_type)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    features = pickle.load(f)
                self.cache_stats["hits"] += 1
                return features
            except Exception:
                cache_path.unlink(missing_ok=True)

        self.cache_stats["misses"] += 1
        return None

    def save_to_cache(self, file_path, feature_type, features):
        if not self.use_cache or features is None:
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

    def load_photos(self) -> List[Path]:
        logger.info("📁 Searching for image files...")

        photo_files: List[Path] = []
        if self.photos:
            photo_files.extend(self.photos)
        else:
            for ext in self.image_extensions:
                photo_files.extend(self.input_path.glob(f"*{ext}"))
                photo_files.extend(self.input_path.glob(f"*{ext.upper()}"))

        logger.info(f"✅ Found {len(photo_files)} image files.")
        return photo_files

    def extract_clip_features(self, image, file_path):
        cached_features = self.load_from_cache(file_path, "clip")
        if cached_features is not None:
            return cached_features

        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
            features_np = features.cpu().numpy().flatten()

            self.save_to_cache(file_path, "clip", features_np)
            return features_np

    def extract_efficientnet_features(self, image, file_path):
        cached_features = self.load_from_cache(file_path, "efficientnet")
        if cached_features is not None:
            return cached_features

        with torch.no_grad():
            image_tensor = (
                self.efficientnet_transform(image).unsqueeze(0).to(self.device)
            )
            features = self.efficientnet(image_tensor)
            features_np = features.cpu().numpy().flatten()

            self.save_to_cache(file_path, "efficientnet", features_np)
            return features_np

    def extract_vit_features(self, image, file_path):
        cached_features = self.load_from_cache(file_path, "vit")
        if cached_features is not None:
            return cached_features

        with torch.no_grad():
            image_tensor = self.vit_transform(image).unsqueeze(0).to(self.device)
            features = self.vit_model(image_tensor)
            features_np = features.cpu().numpy().flatten()

            self.save_to_cache(file_path, "vit", features_np)
            return features_np

    def extract_traditional_features(self, image_path):
        cached_features = self.load_from_cache(image_path, "traditional")
        if cached_features is not None:
            return cached_features

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            img_resized = cv2.resize(img, (224, 224))
            features = []

            img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            for i in range(3):
                hist = cv2.calcHist(
                    [img_hsv], [i], None, [32], [0, 256 if i > 0 else 180]
                )
                features.extend(hist.flatten())

            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)

            lbp = self.calculate_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
            features.extend(lbp_hist.flatten())

            features_np = np.array(features)
            self.save_to_cache(image_path, "traditional", features_np)
            return features_np

        except Exception as e:
            logger.error(f"❌ Failed to extract traditional features ({image_path}): {e}")
            return None

    def calculate_lbp(self, image, radius=1, n_points=8):
        rows, cols = image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                code = 0
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j + radius * np.sin(angle)))
                    if 0 <= x < rows and 0 <= y < cols and image[x, y] >= center:
                        code += 2**p
                lbp[i, j] = code
        return lbp

    def mask_people(self, image):
        """이미지에서 사람을 감지하고 마스킹"""
        try:
            # DETR 모델을 사용하여 사람 감지
            inputs = self.people_detector(images=image, return_tensors="pt")
            outputs = self.people_detector.model(**inputs)
            
            # DETR 모델 결과 처리
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.people_detector.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            # NumPy 배열로 변환
            img_np = np.array(image)
            
            # 사람으로 감지된 영역 마스킹 (평균 색상 또는 가우시안 블러로 대체)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label == 1:  # COCO 데이터셋에서 '사람' 클래스는 인덱스 1
                    # 박스 좌표 추출
                    xmin, ymin, xmax, ymax = box.int().tolist()
                    
                    # 영역 마스킹 (다양한 방법 중 하나 선택)
                    # 1. 평균 색상으로 대체
                    # mean_color = np.mean(img_np, axis=(0, 1))
                    # img_np[ymin:ymax, xmin:xmax] = mean_color
                    
                    # 2. 가우시안 블러 적용
                    roi = img_np[ymin:ymax, xmin:xmax]
                    if roi.size > 0:  # 유효한 영역인 경우
                        blurred = cv2.GaussianBlur(roi, (51, 51), 0)
                        img_np[ymin:ymax, xmin:xmax] = blurred
            
            # 이미지로 변환하여 반환
            masked_image = Image.fromarray(img_np)
            return masked_image
        
        except Exception as e:
            print(f"사람 마스킹 중 오류 발생: {e}")
            return image  # 오류 시 원본 이미지 반환
        
    def convert_image_file(self, path):
        image = Image.open(path).convert('RGB')
        if self.remove_people and self.people_detector is not None:
            image = self.mask_people(image, self.people_detector)
        return image

    def extract_deep_features(self, image_path):
        cached_combined = self.load_from_cache(image_path, "combined")
        if cached_combined is not None:
            return cached_combined, None

        try:
            image = self.convert_image_file(image_path)
            features_dict = {}

            clip_features = self.extract_clip_features(image, image_path)
            features_dict["clip"] = clip_features

            efficientnet_features = self.extract_efficientnet_features(image, image_path)
            features_dict["efficientnet"] = efficientnet_features

            vit_features = self.extract_vit_features(image, image_path)
            features_dict["vit"] = vit_features

            traditional_features = self.extract_traditional_features(image_path)
            if traditional_features is not None:
                features_dict["traditional"] = traditional_features

            combined_features = []
            clip_weighted = clip_features * 0.4
            combined_features.extend(clip_weighted)

            efficientnet_norm = efficientnet_features / (np.linalg.norm(efficientnet_features) + 1e-8)
            efficientnet_weighted = efficientnet_norm * 0.3
            combined_features.extend(efficientnet_weighted)

            vit_norm = vit_features / (np.linalg.norm(vit_features) + 1e-8)
            vit_weighted = vit_norm * 0.25
            combined_features.extend(vit_weighted)

            if traditional_features is not None:
                traditional_norm = traditional_features / (np.linalg.norm(traditional_features) + 1e-8)
                traditional_weighted = traditional_norm * 0.05
                combined_features.extend(traditional_weighted)

            combined_features_np = np.array(combined_features)
            self.save_to_cache(image_path, "combined", combined_features_np)
            return combined_features_np, features_dict

        except Exception as e:
            logger.error(f"❌ Deep feature extraction failed ({image_path}): {e}")
            return None, None

    def advanced_clustering(self, features_array, photo_files):
        logger.info("🧠 Performing advanced clustering...")
        from sklearn.preprocessing import normalize

        features_normalized = normalize(features_array, norm="l2")
        similarity_matrix = cosine_similarity(features_normalized)
        distance_matrix = np.clip(1 - similarity_matrix, 0, 2)

        clustering = DBSCAN(eps=self.similarity_threshold, min_samples=2, metric="euclidean")
        cluster_labels = clustering.fit_predict(features_normalized)

        unique_labels = set(cluster_labels)
        refined_groups = []
        similarity_matrix = cosine_similarity(features_normalized)

        for label in unique_labels:
            if label == -1:
                continue
            group_indices = [i for i, l in enumerate(cluster_labels) if l == label]
            group_photos = [photo_files[i] for i in group_indices]

            if len(group_photos) >= 2:
                group_similarities = [
                    similarity_matrix[i][j]
                    for i_idx, i in enumerate(group_indices)
                    for j_idx, j in enumerate(group_indices)
                    if i_idx < j_idx
                ]
                avg_similarity = np.mean(group_similarities) if group_similarities else 0
                refined_groups.append({
                    "id": label,
                    "photos": group_photos,
                    "count": len(group_photos),
                    "avg_similarity": avg_similarity,
                    "quality_score": avg_similarity * len(group_photos),
                })

        noise_indices = [i for i, l in enumerate(cluster_labels) if l == -1]
        for noise_idx in noise_indices:
            noise_photo = photo_files[noise_idx]
            best_group, best_similarity = None, 0
            for group in refined_groups:
                group_indices = [i for i, p in enumerate(photo_files) if p in group["photos"]]
                similarities = [similarity_matrix[noise_idx][i] for i in group_indices]
                max_sim = max(similarities) if similarities else 0
                if max_sim > best_similarity and max_sim > (1 - self.similarity_threshold * 1.5):
                    best_similarity, best_group = max_sim, group
            
            if best_group:
                best_group["photos"].append(noise_photo)
                best_group["count"] += 1
            else:
                refined_groups.append({
                    "id": f"unique_{len(refined_groups)}",
                    "photos": [noise_photo],
                    "count": 1,
                    "avg_similarity": 1.0,
                    "quality_score": 0.5,
                })

        refined_groups.sort(key=lambda x: x["quality_score"], reverse=True)
        return refined_groups

    def cluster_photos(self):
        logger.info("🔍 Starting deep learning-based photo analysis...")
        photo_files = self.load_photos()
        if len(photo_files) < 2:
            logger.warning("❌ Not enough images to analyze.")
            return

        features_list, valid_photos, all_features_dict = [], [], []
        logger.info("🚀 Extracting high-dimensional features... (caching may speed this up)")
        start_time = time.time()
        for photo_file in tqdm(photo_files, desc="Extracting deep features"):
            combined_features, features_dict = self.extract_deep_features(photo_file)
            if combined_features is not None:
                features_list.append(combined_features)
                valid_photos.append(photo_file)
                all_features_dict.append(features_dict)
        
        extraction_time = time.time() - start_time
        if len(features_list) < 2:
            logger.warning("❌ Not enough images with successfully extracted features.")
            return

        logger.info(f"✅ Extracted features from {len(valid_photos)} images in {extraction_time:.1f}s")
        if self.use_cache:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            logger.info(f"💾 Cache stats: {self.cache_stats['hits']} hits, {self.cache_stats['misses']} misses ({hit_rate:.1f}% hit rate)")
            if hit_rate > 50:
                logger.info("🚀 Cache significantly improved processing speed!")

        features_array = np.array(features_list)
        logger.info(f"📊 Feature vector dimensions: {features_array.shape}")
        self.groups = self.advanced_clustering(features_array, valid_photos)
        logger.info(f"✅ Clustered into {len(self.groups)} groups with high precision.")
        for i, group in enumerate(self.groups):
            quality_desc = "High" if group["quality_score"] > 2 else "Medium" if group["quality_score"] > 1 else "Low"
            logger.info(f"   📍 Group {group['id']}: {group['count']} photos (Similarity: {group['avg_similarity']:.3f}, Quality: {quality_desc})")

    def cluster(self, photo_paths: List[str]) -> List[List[str]]:
        """
        Clusters a given list of photo paths.
        This is the main entry point when used as part of a pipeline.
        """
        self.photos = [Path(p) for p in photo_paths]
        self.cluster_photos()
        
        sub_clusters = []
        if self.groups:
            for group in self.groups:
                sub_clusters.append([str(photo) for photo in group["photos"]])
        return sub_clusters

    def run(self):
        """Standalone execution for AI classification."""
        logger.info("🚀 Starting AI-based park photo auto-classifier!")
        logger.info(f"📂 Input folder: {self.input_path}")

        if not self.input_path.exists():
            logger.error(f"❌ Input folder does not exist: {self.input_path}")
            return

        try:
            self.cluster_photos()
            if not self.groups:
                logger.warning("❌ No groups could be classified.")
                return

            self.create_output_folders()
            self.copy_photos_to_groups()
            self.create_master_result_image()
            self.create_detailed_report()

            logger.info("🎉 AI classification complete!")
            logger.info(f"📁 Results in: {self.output_path}")
            logger.info("📊 Check classification_result.jpg for an overview.")
            logger.info("💡 Check detailed_report.txt for a detailed analysis.")

        except Exception as e:
            logger.error(f"❌ Error during processing: {e}", exc_info=True)

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
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        for idx, group in enumerate(self.groups):
            row, col = idx // cols, idx % cols
            group_x = col * (cell_width + padding)
            group_y = row * (cell_height + header_height + padding)
            draw.rectangle([group_x, group_y, group_x + cell_width, group_y + header_height], fill="lightgray", outline="gray")
            draw.text((group_x + 10, group_y + 5), f"Cluster {group['id']}", fill="black", font=font)
            draw.text((group_x + 10, group_y + 22), f"{group['count']} photos", fill="gray", font=small_font)

            photos_to_show = group["photos"][:3]
            photo_width = cell_width // len(photos_to_show)
            for photo_idx, photo_path in enumerate(photos_to_show):
                try:
                    img = Image.open(photo_path).resize((photo_width - 2, cell_height - 2), Image.Resampling.LANCZOS)
                    photo_x = group_x + photo_idx * photo_width + 1
                    photo_y = group_y + header_height + 1
                    master_image.paste(img, (photo_x, photo_y))
                    if photo_idx < len(photos_to_show) - 1:
                        draw.line([(photo_x + photo_width - 1, photo_y), (photo_x + photo_width - 1, photo_y + cell_height - 2)], fill="white", width=2)
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
            "groups": [{
                "id": str(g["id"]),
                "photo_count": g["count"],
                "average_similarity": float(g["avg_similarity"]),
                "quality_score": float(g["quality_score"]),
                "photos": [p.name for p in g["photos"]],
            } for g in self.groups],
        }

        report_path = self.output_path / "analysis_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"📊 Detailed report saved: {report_path}")


def create_people_detector(device):
    """사람 감지 모델 초기화"""
    try:
        print("사람 감지 모델 로딩 중...")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # 감지 모델을 GPU로 이동 (가능한 경우)
        model = model.to(device)
        model.eval()
        return PeopleDetector(processor, model, device)
    except Exception as e:
        print(f"사람 감지 모델 로드 실패: {e}")
        print("사람 마스킹 비활성화 중...")
        return None


# 사용자 정의 감지기 클래스 생성
class PeopleDetector:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device
    
    def __call__(self, images, return_tensors="pt"):
        inputs = self.processor(images=images, return_tensors=return_tensors)
        # 입력을 디바이스로 이동
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        return inputs
    
    def post_process_object_detection(self, outputs, **kwargs):
        return self.processor.post_process_object_detection(outputs, **kwargs)