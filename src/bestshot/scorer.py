"""Phase 4: BestShot Score 계산 로직.

지침 12번 설계를 그대로 따른다.
BestShot Score = Confidence + Sharpness + Object Size + Occlusion + Position
각 서브 점수는 0~1로 정규화한다. 가중치는 처음엔 동일(1.0)로 시작하고
실험으로 조정한다(아직 튜닝 전 — EXP-004 Decision 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def sharpness_score(crop_bgr: np.ndarray) -> float:
    """Laplacian 분산 기반 Sharpness. 값이 클수록 선명함.
    여러 크롭 간 상대 비교를 위해 log 스케일 후 0~1로 클리핑한다.
    """
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 경험적으로 person crop의 Laplacian 분산은 대략 0~2000 범위에 분포.
    # 그대로 쓰면 스케일이 다른 점수를 압도하므로 0~1로 정규화.
    return float(min(var / 500.0, 1.0))


def size_score(bbox: tuple[float, float, float, float], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
    # 너무 작은 박스는 낮은 점수, 프레임의 2% 이상이면 만점에 가깝게.
    return float(min(area_ratio / 0.02, 1.0))


def position_score_margin_legacy(
    bbox: tuple[float, float, float, float], frame_w: int, frame_h: int, margin_ratio: float = 0.02
) -> float:
    """[FC-003, EXP-018에서 교체됨] 화면 경계 margin(기본 2%) '근접' 여부만 보고
    0.3/1.0 이진 점수를 매기던 이전 구현. 실제로 잘리지 않은 사진도 경계에
    가깝기만 하면 낮은 점수를 받는 문제(FC-003)가 있어 production에서는
    더 이상 쓰지 않는다. EXP-018에서 position_score()와의 비교 baseline으로만
    남겨둔다."""
    x1, y1, x2, y2 = bbox
    margin_x, margin_y = frame_w * margin_ratio, frame_h * margin_ratio
    touches_edge = x1 <= margin_x or y1 <= margin_y or x2 >= frame_w - margin_x or y2 >= frame_h - margin_y
    return 0.3 if touches_edge else 1.0


def position_score(
    bbox: tuple[float, float, float, float], frame_w: int, frame_h: int, boundary_eps: float = 2.0
) -> float:
    """화면 경계에 실제로 잘린 정도에 따라 점수를 매긴다 (FC-003 수정, EXP-018).

    이전 구현(position_score_margin_legacy)은 bbox가 경계에서 margin_ratio(기본 2%,
    1080p 기준 약 21px) 이내로 '근접'하기만 하면 실제 잘림 여부와 무관하게 0.3점을
    줬다. 이는 "경계 근접"과 "실제로 잘려서 몸의 일부가 안 보임"을 혼동한 것이다.

    실제로 카메라 프레임은 사람의 '보이는 부분'만 담으므로, 몸이 잘린 경우 Detector가
    내놓는 bbox 좌표 자체가 이미지 경계(0 또는 frame_w/frame_h)에 거의 정확히 맞닿는다.
    반면 사람이 경계 근처에 서 있어도 몸 전체가 보이면 bbox와 경계 사이에 여백이 남는다.
    boundary_eps(기본 2px)는 이 여백을 요구하되, Detector 좌표의 반올림 오차만 흡수할
    정도로 작게 잡아 "근접"과 "실제로 닿음"을 구분한다.

    닿은 변의 개수로 등급을 나눈다 (변 1개만 닿으면 프레임 한쪽으로 나가는 중,
    2개 이상 닿으면 모서리로 빠지는 중이라 더 많이 잘렸을 가능성이 높다).
    """
    x1, y1, x2, y2 = bbox
    touched = 0
    if x1 <= boundary_eps:
        touched += 1
    if y1 <= boundary_eps:
        touched += 1
    if x2 >= frame_w - boundary_eps:
        touched += 1
    if y2 >= frame_h - boundary_eps:
        touched += 1
    if touched == 0:
        return 1.0
    if touched == 1:
        return 0.6
    return 0.2


def occlusion_score(bbox: tuple[float, float, float, float], other_boxes: list[tuple[float, float, float, float]]) -> float:
    """같은 프레임의 다른 박스와 IoU가 클수록(=많이 겹칠수록) 낮은 점수.
    실제 Occlusion 여부를 직접 관측할 수는 없어 IoU를 Proxy로 사용한다(한계로 명시).
    """
    if not other_boxes:
        return 1.0
    ax1, ay1, ax2, ay2 = bbox
    max_iou = 0.0
    for ob in other_boxes:
        bx1, by1, bx2, by2 = ob
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union > 0:
            max_iou = max(max_iou, inter / union)
    return float(max(0.0, 1.0 - max_iou))


@dataclass
class BestShotWeights:
    confidence: float = 1.0
    sharpness: float = 1.0
    size: float = 1.0
    occlusion: float = 1.0
    position: float = 1.0


def bestshot_score(
    confidence: float,
    crop_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    other_boxes: list[tuple[float, float, float, float]],
    frame_w: int,
    frame_h: int,
    weights: BestShotWeights = BestShotWeights(),
) -> dict:
    s_conf = confidence
    s_sharp = sharpness_score(crop_bgr)
    s_size = size_score(bbox, frame_w, frame_h)
    s_occ = occlusion_score(bbox, other_boxes)
    s_pos = position_score(bbox, frame_w, frame_h)
    total = (
        weights.confidence * s_conf
        + weights.sharpness * s_sharp
        + weights.size * s_size
        + weights.occlusion * s_occ
        + weights.position * s_pos
    )
    return {
        "confidence": s_conf,
        "sharpness": s_sharp,
        "size": s_size,
        "occlusion": s_occ,
        "position": s_pos,
        "total": total,
    }
