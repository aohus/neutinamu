import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image
from PIL.ExifTags import TAGS
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

# 고정된 입력 및 출력 디렉토리 설정
IMAGE_DIR = "./1차"
OUTPUT_DIR = IMAGE_DIR + "_claude"


class ImageDataset(Dataset):
    """DINOv2 모델용 이미지 데이터셋"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # PIL로 이미지 읽기
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return {"path": image_path, "image": image}
        except Exception as e:
            print(f"이미지 {image_path} 로딩 중 오류: {e}")
            # 오류 발생 시 검은색 더미 이미지 반환
            dummy = Image.new("RGB", (224, 224), color="black")
            if self.transform:
                dummy = self.transform(dummy)
            return {"path": image_path, "image": dummy, "error": True}


def extract_datetime(image_path):
    """이미지의 EXIF 데이터에서 촬영 시간 추출"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"Error extracting datetime from {image_path}: {e}")
    return None


def extract_features_with_dinov2(image_paths, batch_size=8):
    """DINOv2-giant 모델을 사용하여 이미지 특징 추출"""
    print("DINOv2-giant 모델 로딩 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    try:
        # DINOv2 모델 및 프로세서 로드
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        model = AutoModel.from_pretrained("facebook/dinov2-giant")
        model = model.to(device)
        model.eval()

        # 이미지 전처리 변환
        transform = transforms.Compose(
            [
                transforms.Resize((518, 518)),  # DINOv2 권장 사이즈
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 데이터셋 및 데이터로더 생성
        dataset = ImageDataset(image_paths, transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        features = []
        valid_paths = []
        datetime_list = []

        print("이미지 특징 추출 중...")
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="특징 추출"):
                images = batch["image"].to(device)
                paths = batch["path"]
                errors = batch.get("error", [False] * len(paths))

                # 유효한 이미지만 처리
                valid_indices = [i for i, error in enumerate(errors) if not error]

                if valid_indices:
                    # DINOv2 모델로 특징 추출
                    outputs = model(images[valid_indices])
                    # [CLS] 토큰의 특징 벡터 사용
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                    for i, idx in enumerate(valid_indices):
                        features.append(batch_features[i])
                        valid_paths.append(paths[idx])

                        # 시간 정보 추출
                        dt = extract_datetime(paths[idx])
                        datetime_list.append(dt)

        print(f"총 {len(valid_paths)}개 이미지에서 특징 추출 완료")
        return features, valid_paths, datetime_list

    except Exception as e:
        print(f"DINOv2 모델 사용 중 오류 발생: {e}")
        # 오류 시 기본 특징 추출 방식으로 대체
        print("기본 특징 추출 방식으로 대체합니다...")
        return extract_features_with_opencv(image_paths)


def extract_features_with_opencv(image_paths):
    """OpenCV를 사용한 백업 특징 추출 방식"""
    print("OpenCV를 사용하여 특징 추출 중...")
    features_list = []
    valid_paths = []
    datetime_list = []

    for i, path in enumerate(image_paths):
        if i % 20 == 0:
            print(f"  {i}/{len(image_paths)} 처리 중...")

        try:
            # 이미지 로드 및 리사이징
            img = cv2.imread(path)
            if img is None:
                continue

            # 이미지 크기 조정
            img = cv2.resize(img, (224, 224))

            # 색상 히스토그램 특징 추출
            hist_features = []
            for i in range(3):  # BGR 채널
                hist = cv2.calcHist([img], [i], None, [64], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)

            # 간단한 전역 특징 추가
            for i in range(3):
                hist_features.append(float(np.mean(img[:, :, i])))
                hist_features.append(float(np.std(img[:, :, i])))

            features = np.array(hist_features, dtype=float)

            # 특징 벡터가 유효한지 확인
            if not np.all(np.isfinite(features)):
                continue

            features_list.append(features)
            valid_paths.append(path)

            # 시간 정보 추출
            dt = extract_datetime(path)
            datetime_list.append(dt)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return features_list, valid_paths, datetime_list


def cluster_images(features, valid_paths, datetime_list, n_clusters=None):
    """이미지 클러스터링"""
    if not features or not valid_paths:
        print("유효한 특징이 추출되지 않았습니다.")
        return None, None

    # 특징 데이터 정규화
    print("특징 데이터 정규화 중...")
    X = np.array(features)
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X)

    # 시간 특징 추가 (있는 경우)
    # has_time_data = all(dt is not None for dt in datetime_list)
    # if has_time_data:
    #     print("시간 정보를 활용하여 클러스터링합니다...")
    #     # 시간을 타임스탬프로 변환하고 정규화
    #     timestamps = np.array([(dt - datetime(1970, 1, 1)).total_seconds() for dt in datetime_list]).reshape(-1, 1)
    #     time_scaler = StandardScaler()
    #     timestamps_scaled = time_scaler.fit_transform(timestamps)

    #     # 가중치 적용 (시간 정보에 가중치 부여)
    #     time_weight = 0.3
    #     feature_weight = 1.0 - time_weight

    #     # 특징과 시간 정보 결합
    #     X_combined = np.hstack([X_scaled * feature_weight, timestamps_scaled * time_weight])
    # else:
    #     print("시간 정보가 없어 이미지 특징만으로 클러스터링합니다...")
    #     X_combined = X_scaled

    # 클러스터링 알고리즘 선택 및 적용
    print("클러스터링 중...")
    if n_clusters and n_clusters > 0:
        print(f"K-means 클러스터링으로 {n_clusters}개 구역으로 분류합니다.")
        # K-means 클러스터링
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        print("DBSCAN 클러스터링으로 자동 구역 분류를 수행합니다.")
        # DBSCAN 클러스터링 (클러스터 수를 자동으로 결정)
        eps = 0.5  # 밀도 기반 클러스터링의 이웃 거리 임계값
        min_samples = 5  # 핵심 포인트 기준 최소 샘플 수
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    labels = clusterer.fit_predict(X_combined)

    return valid_paths, labels


def organize_images_by_cluster(image_paths, labels, output_dir):
    """클러스터링 결과에 따라 이미지 정리"""
    # 클러스터별 디렉토리 생성
    unique_labels = set(labels)

    for label in unique_labels:
        # -1은 DBSCAN에서 노이즈를 나타냄
        if label == -1:
            cluster_dir = os.path.join(output_dir, "uncategorized")
        else:
            cluster_dir = os.path.join(output_dir, f"zone_{label+1}")

        os.makedirs(cluster_dir, exist_ok=True)

    # 각 이미지를 해당 클러스터 디렉토리로 복사
    print("이미지를 구역별 폴더로 복사하는 중...")
    for path, label in zip(image_paths, labels):
        if label == -1:
            dest_dir = os.path.join(output_dir, "uncategorized")
        else:
            dest_dir = os.path.join(output_dir, f"zone_{label+1}")

        # 원본 파일명 유지하면서 복사
        filename = os.path.basename(path)
        shutil.copy2(path, os.path.join(dest_dir, filename))

    # 각 클러스터에 몇 개의 이미지가 있는지 출력
    print("\n분류 결과:")
    for label in unique_labels:
        if label == -1:
            dir_name = "uncategorized"
        else:
            dir_name = f"zone_{label+1}"

        cluster_dir = os.path.join(output_dir, dir_name)
        num_images = len(os.listdir(cluster_dir))
        print(f"{dir_name}: {num_images}개 이미지")


def visualize_clusters(image_paths, labels, output_dir):
    """클러스터링 결과 시각화"""
    # 각 클러스터에서 3개의 샘플 이미지를 보여줌
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:  # uncategorized를 마지막에 표시
        unique_labels.remove(-1)
        unique_labels.append(-1)

    # 시각화를 위한 그리드 설정
    n_clusters = len(unique_labels)

    # 클러스터가 없으면 시각화를 건너뜁니다
    if n_clusters == 0:
        print("시각화할 클러스터가 없습니다.")
        return

    # 각 클러스터마다 최대 3개의 이미지를 표시
    max_samples = 3
    fig, axes = plt.subplots(n_clusters, max_samples, figsize=(15, 5 * n_clusters))

    # 단일 클러스터이거나 단일 샘플인 경우 축 처리
    if n_clusters == 1:
        axes = np.array([axes])  # 2D 배열로 만들기

    print("클러스터 시각화 생성 중...")
    for i, label in enumerate(unique_labels):
        # 해당 클러스터의 이미지 경로 가져오기
        cluster_images = [
            path for path, lbl in zip(image_paths, labels) if lbl == label
        ]

        # 최대 3개의 샘플 선택
        samples = cluster_images[:max_samples]

        # 샘플 이미지 표시
        for j in range(max_samples):
            ax = axes[i, j] if n_clusters > 1 else axes[j]

            if j < len(samples):  # 샘플이 있는 경우
                try:
                    sample = samples[j]
                    img = cv2.imread(sample)
                    if img is not None:
                        img = cv2.cvtColor(
                            img, cv2.COLOR_BGR2RGB
                        )  # OpenCV는 BGR, matplotlib은 RGB
                        ax.imshow(img)
                        ax.set_title(
                            f"Cluster {label+1 if label != -1 else 'Uncategorized'}\nSample {j+1}"
                        )
                    else:
                        ax.text(0.5, 0.5, "이미지 로드 실패", ha="center", va="center")
                except Exception as e:
                    print(
                        f"Error displaying sample {samples[j] if j < len(samples) else 'unknown'}: {e}"
                    )
                    ax.text(0.5, 0.5, "이미지 표시 오류", ha="center", va="center")
            else:  # 샘플이 부족한 경우
                ax.text(0.5, 0.5, "샘플 없음", ha="center", va="center")

            ax.axis("off")

    plt.tight_layout()
    try:
        vis_path = os.path.join(output_dir, "cluster_samples.png")
        plt.savefig(vis_path)
        plt.close()
        print(f"클러스터 시각화가 {vis_path}에 저장되었습니다.")
    except Exception as e:
        print(f"시각화 저장 중 오류 발생: {e}")
        plt.close()


def main():
    print("=" * 50)
    print("잔디 깎기 작업 사진 구역별 분류 프로그램 (DINOv2 버전)")
    print("=" * 50)

    # 디렉토리 설정 확인
    print(f"입력 디렉토리: {IMAGE_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 지원하는 이미지 확장자
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    # 모든 이미지 파일 경로 수집
    print("이미지 파일 검색 중...")
    image_paths = []
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(root, file))

    print(f"총 {len(image_paths)}개의 이미지를 발견했습니다.")

    # 구역 수 입력 (자동 결정을 원하면 0 입력)
    try:
        n_zones = int(input("구역 수를 입력하세요 (자동 결정은 0 입력): "))
    except ValueError:
        print("유효하지 않은 입력입니다. 자동으로 구역을 결정합니다.")
        n_zones = 0

    # DINOv2 모델을 사용하여 이미지 특징 추출
    try:
        features, valid_paths, datetime_list = extract_features_with_dinov2(image_paths)

        # 클러스터링
        valid_paths, labels = cluster_images(
            features, valid_paths, datetime_list, n_zones if n_zones > 0 else None
        )

        if valid_paths is None or labels is None:
            print("클러스터링에 실패했습니다.")
            return

        # 클러스터링 결과에 따라 이미지 정리 - 구역별로만 분류
        organize_images_by_cluster(valid_paths, labels, OUTPUT_DIR)

        # 클러스터 시각화
        visualize_clusters(valid_paths, labels, OUTPUT_DIR)

    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        return

    print("\n" + "=" * 50)
    print(f"이미지 분류가 완료되었습니다.")
    print(f"결과는 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
