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
- classify_person_attributes_with_mask(): 대안 C(EXP-028, 검증 대상). method b의
  side_margin_ratio(좌우 고정 비율 제외)를 person Segmentation Mask 기반 픽셀 단위
  배경 제외로 교체한다. "영역을 더 정확히 자르면 Accuracy가 오르는가"를 Mask 자체는
  이 모듈에서 만들지 않고(세그멘테이션 모델 의존성을 분리) 호출부(EXP-028 스크립트)가
  전달한다.
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


def white_balance_gray_world(crop_bgr: np.ndarray, gain_min: float = 0.3, gain_max: float = 3.0) -> np.ndarray:
    """카지노 조명(강한 인공조명/난색·한색 조명)으로 생긴 채널 간 색 Cast를 완화한다
    (FC-012, EXP-026 대안 B).

    Gray World 가정(crop 전체 채널 평균이 중립 회색에 가까워야 한다)으로 채널별
    게인을 계산해 보정한다. EXP-025에서 검정 옷이 strong_light/cool_cast 조건에서
    blue로 오분류된 원인은, HSV의 S(채도)가 R/G/B의 "상대적" 차이이기 때문에 어두운
    픽셀(V가 작음)일수록 Cast로 생긴 작은 절대 채널 차이도 S를 크게 부풀린다는
    것이었다(원인 분석: EXP-026 experiment.md). Hue 분류 이전에 채널 Cast 자체를
    줄이면 이 부풀림도 같이 줄어든다.

    gain_min/gain_max: crop이 이미 거의 무채색(검정/흰색 단색)이면 평균이 0에 가까워
    게인이 극단적으로 커질 수 있어 안전 범위로 clip한다.
    """
    if crop_bgr.size == 0:
        return crop_bgr
    img = crop_bgr.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()
    gains = overall_mean / np.clip(channel_means, 1.0, None)
    gains = np.clip(gains, gain_min, gain_max)
    corrected = img * gains.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)


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


def region_color_signal(region_bgr: np.ndarray) -> dict[str, float]:
    """영역의 raw HSV 채도(S)와 raw BGR 채널 표준편차 평균을 계산한다.

    EXP-027(무채색 신호 분리도 분석)에서 "achromatic(검정/흰색/회색) 영역과 chromatic
    영역을 확신도 신호로 구분해 낮은 확신도를 unknown으로 표시할 수 있는가"를 검증하기
    위해 쓰였다. 실측 결과(coco128 실제 crop 21개, 상/하 42개 영역) 두 신호 모두
    achromatic/chromatic 범위가 완전히 겹쳐(achromatic 채도 최댓값 158.89 > chromatic
    채도 최솟값 62.92) 분리 가능한 신호가 아님을 확인했다 - PAR-018 Decision 참고.
    이 함수 자체는 계속 쓰일 수 있어 src/에 남기지만, 이 신호만으로 confidence
    threshold를 만드는 기능은 추가하지 않는다.
    """
    if region_bgr.size == 0:
        return {"mean_saturation": 0.0, "bgr_channel_std": 0.0}
    hsv = _to_hsv(region_bgr).reshape(-1, 3).astype(np.float32)
    mean_saturation = float(hsv[:, 1].mean())
    bgr_channel_std = float(region_bgr.reshape(-1, 3).astype(np.float32).std(axis=0).mean())
    return {"mean_saturation": mean_saturation, "bgr_channel_std": bgr_channel_std}


def region_dominant_color_from_pixels(pixels_bgr: np.ndarray, **kwargs) -> str:
    """세그멘테이션 마스크로 걸러낸 1차원 픽셀 집합(N,3)에 region_dominant_color와 동일한
    필터링+Hue 다수결 로직을 적용한다 (대안 C, EXP-028). region_dominant_color는 내부적으로
    입력을 (-1, 3)으로 reshape하므로 (N,1,3) 형태로 바꿔 그대로 재사용할 수 있다."""
    if pixels_bgr.size == 0:
        return "gray"
    pseudo_region = pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    return region_dominant_color(pseudo_region, **kwargs)


def split_upper_lower_by_mask(
    mask: np.ndarray,
    head_skip_ratio: float = 0.22,
    upper_end_ratio: float = 0.55,
    foot_skip_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Person Segmentation Mask(crop과 동일 HxW의 bool 배열)를 상/하 영역으로 나눈다.

    method b의 side_margin_ratio(좌우 고정 비율 제외)는 배경 Bleed를 "근사"로 줄이는
    수단이었다. 마스크가 있으면 배경 픽셀 자체가 False이므로 좌우 마진이 필요 없다 —
    세로 방향(머리/발)만 기존과 동일 비율로 제외한다.
    """
    height = mask.shape[0]
    upper_mask = np.zeros_like(mask)
    lower_mask = np.zeros_like(mask)
    r0 = int(height * head_skip_ratio)
    r1 = int(height * upper_end_ratio)
    r2 = int(height * (1.0 - foot_skip_ratio))
    upper_mask[r0:r1, :] = mask[r0:r1, :]
    lower_mask[r1:r2, :] = mask[r1:r2, :]
    return upper_mask, lower_mask


def classify_person_attributes_with_mask(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    head_skip_ratio: float = 0.22,
    upper_end_ratio: float = 0.55,
    foot_skip_ratio: float = 0.08,
    min_mask_fraction: float = 0.05,
) -> dict[str, str]:
    """대안 C(채택 여부 검증 대상, EXP-028): method b의 side_margin_ratio 고정 비율
    배경 제외를 person Instance Segmentation Mask 기반 픽셀 단위 배경 제외로 교체한다.
    상/하 분리 비율, 그림자/하이라이트/피부색 필터, Hue 다수결은 method b와 동일하게
    유지해 "영역을 더 정확하게 자르면 Accuracy가 오르는가"만 분리해서 검증한다.

    min_mask_fraction: 세그멘테이션이 실패하거나(Occlusion, 작은 Instance) 마스크가
    영역 내 픽셀의 이 비율보다 적게 덮으면 마스크를 신뢰하지 않고 method b와 동일한
    고정 비율 사각형 영역(side_margin_ratio=0.12)으로 되돌아간다 — method b의
    min_filtered_fraction과 같은 이유(소수 마스크 픽셀만으로는 대표성이 없다).
    """
    if crop_bgr.size == 0 or mask.size == 0:
        return {"upper": "gray", "lower": "gray"}
    upper_mask, lower_mask = split_upper_lower_by_mask(mask, head_skip_ratio, upper_end_ratio, foot_skip_ratio)
    fallback_upper, fallback_lower = split_upper_lower(
        crop_bgr,
        head_skip_ratio=head_skip_ratio,
        upper_end_ratio=upper_end_ratio,
        foot_skip_ratio=foot_skip_ratio,
        side_margin_ratio=0.12,
    )

    def _resolve(region_mask: np.ndarray, fallback_region: np.ndarray) -> str:
        total = region_mask.size
        if total == 0 or region_mask.sum() < min_mask_fraction * total:
            return region_dominant_color(fallback_region)
        return region_dominant_color_from_pixels(crop_bgr[region_mask.astype(bool)])

    return {"upper": _resolve(upper_mask, fallback_upper), "lower": _resolve(lower_mask, fallback_lower)}


def classify_person_attributes_with_mask_and_wb(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    **kwargs,
) -> dict[str, str]:
    """대안 C+WB(EXP-028): 대안 C(Segmentation Mask 배경 제외)와 EXP-026 대안 B(Gray World
    White Balance)를 함께 적용한다. EXP-028 실측에서 대안 C는 전체/GT=black Accuracy를
    개선했지만 FC-012가 원래 겨냥한 cool_cast 흑백->blue 오분류율은 b_wb(WB만 적용)보다
    오히려 나빴다(0.50 > 0.23) - 둘이 서로 다른 오분류 원인(배경 Bleed vs 채널 Cast)을
    해결하므로 함께 적용하면 더 나을 수 있다는 가설을 검증하기 위해 추가했다.
    """
    balanced = white_balance_gray_world(crop_bgr)
    return classify_person_attributes_with_mask(balanced, mask, **kwargs)


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
    if method == "b_wb":
        # EXP-026: Gray World WB를 method "b" 앞에 추가한 실험용 변형. FC-012(검정
        # 옷의 strong_light/cool_cast 오분류)를 겨냥했으나, 측정 결과 검정 GT 셀
        # 자체의 정확도는 WB 적용 전후로 동일했고(오분류 조건만 재배치), low_light에
        # 새로운 regression(-10pp)이 생겨 production 기본값("b")으로 승격하지 않았다
        # (experiment.md EXP-026 Decision 참고). 전체 평균 정확도는 올랐지만(다른
        # 색상 셀에서 개선) 이는 이 실험이 원래 겨냥한 문제와 무관하다.
        balanced = white_balance_gray_world(crop_bgr)
        upper, lower = split_upper_lower(
            balanced,
            head_skip_ratio=0.22,
            upper_end_ratio=0.55,
            foot_skip_ratio=0.08,
            side_margin_ratio=0.12,
        )
        return {"upper": region_dominant_color(upper), "lower": region_dominant_color(lower)}
    raise ValueError(f"unknown method: {method}")
