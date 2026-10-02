"""Stretch: Attribute Metadata - 상의/하의 색상 분류 (지침 19).

Person Crop -> (upper_color, lower_color) 검색 가능한 Metadata를 만드는 것이 목표다
(Attribute Classification 자체는 프로젝트 중심이 아님). 카지노처럼 조명이 강하거나
변하는 환경에서 Attribute가 얼마나 안정적인지가 핵심 질문이라 색상 분류 자체보다
"조명이 바뀌어도 같은 색으로 분류되는가"를 중점적으로 설계한다 (EXP-025 참고).

세 가지 방식을 제공한다:
- whole_bbox_mean_color(): Baseline. bbox 전체를 한 번에 평균 -> 단일 색상.
  상/하의를 구분하지 못하고, 배경 Bleed와 밝기 변화에 그대로 흔들린다.
- region_mean_color(): 대안 A. 상/하 고정 비율로 나눠 영역별 평균 BGR.
  상/하 구분은 가능해졌지만 여전히 배경 Bleed, 머리/피부색, 밝기 변화에 취약하다.
- region_dominant_color(): 대안 B(채택). 상/하 분리 + 머리 영역 제외 +
  그림자/하이라이트(S/V 극단) 픽셀 제거 + 남은 픽셀 중 최빈 Hue 클러스터 선택.
  배경이 소수 픽셀만 섞여 있으면 억제되고, 극단 밝기 픽셀을 걸러내 Hue 기반 분류가
  밝기 변화에 덜 흔들린다.
"""

from __future__ import annotations

import cv2
import numpy as np

COLOR_BUCKETS = (
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
)

# OpenCV HSV: H in [0,179], S/V in [0,255].
VAL_BLACK = 45
SAT_LOW = 30
VAL_WHITE = 195


def classify_hsv_pixel(h: float, s: float, v: float) -> str:
    """단일 HSV 값을 COLOR_BUCKETS 중 하나로 분류한다."""
    if v < VAL_BLACK:
        return "black"
    if s < SAT_LOW:
        return "white" if v >= VAL_WHITE else "gray"
    if h <= 8 or h >= 173:
        return "red"
    if h <= 20:
        return "orange"
    if h <= 33:
        return "yellow"
    if h <= 80:
        return "green"
    if h <= 130:
        return "blue"
    if h <= 155:
        return "purple"
    return "pink"


def _to_hsv(crop_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)


def whole_bbox_mean_color(crop_bgr: np.ndarray) -> str:
    """Baseline: bbox 전체 픽셀의 평균 BGR -> HSV 변환 -> 분류. 상/하의 구분 없음."""
    if crop_bgr.size == 0:
        return "gray"
    mean_bgr = crop_bgr.reshape(-1, 3).mean(axis=0).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = _to_hsv(mean_bgr)[0, 0]
    return classify_hsv_pixel(float(h), float(s), float(v))


def split_upper_lower(
    crop_bgr: np.ndarray,
    head_skip_ratio: float = 0.0,
    upper_end_ratio: float = 0.55,
    foot_skip_ratio: float = 0.0,
    side_margin_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """bbox를 세로로 상/하 영역으로 나눈다.

    head_skip_ratio: 위에서부터 이 비율만큼(얼굴/머리)을 상의 영역에서 제외한다.
    upper_end_ratio: 상의 영역이 끝나는 세로 위치(bbox 높이 대비 비율).
    foot_skip_ratio: 아래에서부터 이 비율만큼(신발)을 하의 영역에서 제외한다.
    side_margin_ratio: 좌우 바깥쪽을 이 비율만큼 제외해 배경 Bleed를 줄인다.
    """
    height, width = crop_bgr.shape[:2]
    x0 = int(width * side_margin_ratio)
    x1 = width - x0
    upper = crop_bgr[int(height * head_skip_ratio) : int(height * upper_end_ratio), x0:x1]
    lower = crop_bgr[int(height * upper_end_ratio) : int(height * (1.0 - foot_skip_ratio)), x0:x1]
    return upper, lower


def region_mean_color(region_bgr: np.ndarray) -> str:
    """대안 A: 영역 전체 픽셀의 단순 평균 BGR -> HSV 변환 -> 분류."""
    if region_bgr.size == 0:
        return "gray"
    mean_bgr = region_bgr.reshape(-1, 3).mean(axis=0).astype(np.uint8).reshape(1, 1, 3)
    h, s, v = _to_hsv(mean_bgr)[0, 0]
    return classify_hsv_pixel(float(h), float(s), float(v))


# 피부색으로 흔히 관측되는 HSV 대역(경험적 휴리스틱, 조명/인종에 따라 완전하지 않음).
# EXP-025에서 클로즈업 crop의 "상의" 영역에 얼굴/손 픽셀이 섞여 실제로는 검정 옷인데
# 피부색(주황 계열 Hue)으로 다수결이 넘어가는 오분류를 관찰해 추가했다. Trade-off:
# 옷이 실제로 주황/빨강 계열이면 이 필터가 옷 픽셀 일부도 함께 제거할 수 있다.
def _is_skin_like(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (h <= 25) & (s >= 40) & (s <= 150) & (v >= 60)


def region_dominant_color(
    region_bgr: np.ndarray,
    sat_min: int = 25,
    val_min: int = 35,
    val_max: int = 245,
    hue_bin_width: int = 10,
    exclude_skin: bool = True,
    min_filtered_fraction: float = 0.3,
) -> str:
    """대안 B(채택): 그림자/하이라이트/피부색 픽셀을 제거한 뒤, 남은 픽셀을 Hue bin으로
    묶어 가장 많은 픽셀이 속한 bin의 대표 HSV로 분류한다.

    S/V 극단값을 필터링하는 이유: 그림자(V 낮음)·하이라이트(V 높음)·배경의
    저채도 영역(S 낮음)은 실제 옷 색상이 아니라 조명/배경 Noise일 가능성이 높다.
    피부색을 추가로 제외하는 이유(exclude_skin): bbox 상단 영역은 머리를 일부
    제외해도 얼굴 하단·목·손 등 피부 픽셀이 섞이기 쉽고, 이 픽셀들이 Hue bin
    다수결을 skin-tone Hue(주황 계열) 쪽으로 왜곡시키는 것을 EXP-025에서 실측했다.
    Hue bin으로 다수결을 취하는 이유: 소수 픽셀(배경 Bleed, 경계선)이 평균을
    왜곡하지 않도록 단순 평균 대신 최빈값에 가까운 통계를 쓰기 위함이다.

    min_filtered_fraction: 필터링 후 남은 픽셀이 전체의 이 비율보다 적으면 필터를
    신뢰하지 않고 원본 전체 픽셀로 되돌아간다(EXP-025에서 실제로 발견한 버그:
    진짜 검정 옷(V가 매우 낮음)은 S/V 필터 자체에 걸려 항상 제외되므로, 소수
    피부색 픽셀만 필터를 통과해 Hue 다수결을 오염시켰다 — 필터 통과 비율이
    낮다는 것 자체가 "필터링된 집합이 실제 옷 색상을 대표하지 못한다"는 신호다).
    """
    if region_bgr.size == 0:
        return "gray"
    hsv = _to_hsv(region_bgr).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mask = (s >= sat_min) & (v >= val_min) & (v <= val_max)
    if exclude_skin:
        mask = mask & ~_is_skin_like(h, s, v)
    filtered = hsv[mask]
    if filtered.shape[0] < min_filtered_fraction * hsv.shape[0]:
        filtered = hsv
    bins = (filtered[:, 0] // hue_bin_width).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    dominant_bin = values[np.argmax(counts)]
    in_bin = filtered[bins == dominant_bin]
    med_h, med_s, med_v = np.median(in_bin, axis=0)
    return classify_hsv_pixel(float(med_h), float(med_s), float(med_v))


def classify_person_attributes(crop_bgr: np.ndarray, method: str = "b") -> dict[str, str]:
    """person crop -> {"upper": color, "lower": color} (method: "baseline"|"a"|"b")."""
    if method == "baseline":
        color = whole_bbox_mean_color(crop_bgr)
        return {"upper": color, "lower": color}
    if method == "a":
        upper, lower = split_upper_lower(crop_bgr, head_skip_ratio=0.0, upper_end_ratio=0.5)
        return {"upper": region_mean_color(upper), "lower": region_mean_color(lower)}
    if method == "b":
        upper, lower = split_upper_lower(
            crop_bgr,
            head_skip_ratio=0.22,
            upper_end_ratio=0.55,
            foot_skip_ratio=0.08,
            side_margin_ratio=0.12,
        )
        return {"upper": region_dominant_color(upper), "lower": region_dominant_color(lower)}
    raise ValueError(f"unknown method: {method}")
