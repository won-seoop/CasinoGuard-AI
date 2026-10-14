"""Stretch: Person Re-ID - Person Crop -> Embedding (지침 18).

Baseline: color_histogram_embedding() - 고전적 CBIR 방식(Deep Learning 이전
Person Re-ID에서도 흔히 쓰인 Color Histogram). 외부 모델 없이 cv2 연산만으로
계산 가능해 Edge Camera에서도 거의 비용이 없다.

대안 A(검토, 기각): OSNet(torchreid) 등 Person Re-ID 전용 Pretrained 모델.
Pretrained Weight가 공식적으로 Google Drive에만 호스팅되어 있어(torchreid
모델 zoo), 이 프로젝트의 egress 정책(pypi/GitHub Release Asset만 허용,
EXP-027/028/035가 이미 확인한 패턴)에서 다운로드가 막힌다 - EXP-037에서
drive.google.com(403)과 해당 GitHub 저장소 페이지(403, Release Asset이
아닌 일반 페이지라 차단)를 실제로 curl로 확인했다. 추측이 아니라 실측
결과다.

대안 B(채택): yolo11n-cls.pt(ImageNet Pretrained, GitHub Release Asset으로
접근 가능 - 이미 이 프로젝트가 Detection/Segmentation에 쓰는 것과 동일한
호스팅 경로)의 분류 Backbone에서 최종 Linear 직전의 Global Average Pool
특징을 범용 임베딩으로 재사용한다. Person Re-ID 전용 모델이 아니라
ImageNet 분류용이라는 한계가 있지만, 추가 모델/Weight 없이 이미 파이프라인이
로드하는 것과 동일한 YOLO 계열 Backbone만으로 동작한다는 Edge 배포 장점이
있다(지침 25 Edge Device SW 연결).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HIST_BINS = (8, 8, 8)  # H, S, V


def color_histogram_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """Baseline: HSV 3채널 Histogram을 이어붙여 L2 정규화한 벡터.

    조명 변화에는 S/V bin 경계가 흔들려 약하지만, 동일인의 주요 색상 분포는
    유지되는 경우가 많아 고전 Re-ID Baseline으로 흔히 쓰인다.
    """
    if crop_bgr.size == 0:
        return np.zeros(sum(HIST_BINS), dtype=np.float64)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    channels = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for c in range(3):
        hist = cv2.calcHist([hsv], [c], None, [HIST_BINS[c]], ranges[c])
        channels.append(hist.flatten())
    embedding = np.concatenate(channels).astype(np.float64)
    norm = np.linalg.norm(embedding)
    if norm < 1e-12:
        return embedding
    return embedding / norm


class YoloBackboneEmbedder:
    """대안 B(채택): yolo11n-cls Backbone의 GAP 특징을 임베딩으로 재사용한다.

    torch/ultralytics는 무거운 선택적 의존성이라, import는 이 클래스를 실제로
    쓸 때만 지연 로드한다(matching.py와 color_histogram_embedding은 cv2/numpy만
    있으면 동작·테스트 가능하게 유지).
    """

    def __init__(self, weights_path: str | Path = "yolo11n-cls.pt", input_size: int = 224) -> None:
        import torch
        from ultralytics import YOLO

        self._torch = torch
        self.input_size = input_size
        self.model = YOLO(str(weights_path)).model
        self.model.eval()
        self._feature: np.ndarray | None = None

        head = self.model.model[-1]
        linear = head.linear if hasattr(head, "linear") else head

        def _capture_input(_module, inputs):
            self._feature = inputs[0].detach().cpu().numpy()

        linear.register_forward_pre_hook(_capture_input)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Person Crop(BGR) -> (D,) GAP 특징 벡터(L2 정규화)."""
        if crop_bgr.size == 0:
            raise ValueError("empty crop")
        torch = self._torch
        resized = cv2.resize(crop_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            self.model(tensor)
        feature = self._feature
        if feature is None:
            raise RuntimeError("forward hook이 특징을 포착하지 못함")
        feature = feature.reshape(-1).astype(np.float64)
        norm = np.linalg.norm(feature)
        if norm < 1e-12:
            return feature
        return feature / norm
