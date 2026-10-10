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


def estimate_gamma_from_reference(
    reference_bgr: np.ndarray,
    target_mean: float = 128.0,
    gamma_min: float = 0.4,
    gamma_max: float = 2.5,
) -> float:
    """reference_bgr의 평균 밝기를 target_mean으로 맞추는 감마 값을 추정한다
    (표준 Auto-Exposure 기법): (mean/255)^gamma = target_mean/255 를 gamma에 대해 풀면
    gamma = log(target_mean/255) / log(mean/255).

    gamma_min/gamma_max: reference가 거의 순수 흑/백이면 mean이 0/255에 가까워 gamma가
    극단적으로 커지거나 작아질 수 있어 안전 범위로 clip한다(white_balance_gray_world의
    gain_min/gain_max와 동일한 이유).
    """
    if reference_bgr.size == 0:
        return 1.0
    mean = float(reference_bgr.astype(np.float32).mean())
    if mean <= 1.0 or mean >= 254.0:
        return 1.0
    gamma = np.log(target_mean / 255.0) / np.log(mean / 255.0)
    return float(np.clip(gamma, gamma_min, gamma_max))


def apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """img_bgr 전체에 감마 보정(power law)을 적용한다."""
    if img_bgr.size == 0:
        return img_bgr
    normalized = img_bgr.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def adaptive_gamma_correct(
    crop_bgr: np.ndarray,
    target_mean: float = 128.0,
    gamma_min: float = 0.4,
    gamma_max: float = 2.5,
) -> np.ndarray:
    """카지노 CCTV의 강한 조명/역광으로 생긴 노출 손실(과다/저노출)을 완화하려는 시도
    (FC-012, EXP-030 대안 A - **기각됨**, strong_light 전용으로 설계).

    EXP-026의 White Balance는 "채널 간 상대적 Cast"를 보정하지만, strong_light(EXP-025/027의
    `img * 1.9 + 25`)는 모든 채널이 함께 밝아지는 전역 노출 문제라 WB로는 전혀 개선되지
    않았다(PAR-017/PAR-019에서 실측: WB·Segmentation Mask 적용 후에도 GT=black→blue
    오분류율이 오히려 18.2%→22.7%→27.3%로 악화). 이 함수는 crop 자신의 실측 평균 밝기로
    감마를 추정해(estimate_gamma_from_reference) crop 자신에게 적용한다.

    **EXP-030에서 기각된 이유(중요)**: crop(person bbox)의 평균 밝기는 "조명 노출"과
    "옷 색상(albedo)"이 분리되지 않은 혼합 신호다 - 실측(coco128 n=21, normal 조명)에서
    검정 옷 GT crop의 평균 밝기는 24.7~153.3까지 퍼져 있고 흰 옷 GT crop과 겹친다. crop
    자신의 밝기를 target_mean(128)으로 강제로 맞추면 "검정 옷은 어둡다"는 분류 신호 자체를
    지워버려, normal 조명에서도 Accuracy가 33.3%→9.5%로 급락했다(의도한 strong_light뿐
    아니라 모든 조건이 악화). 노출은 object 자신이 아니라 주변 배경(Scene)에서 추정해야
    한다는 결론을 얻었고, 그 방향은 adaptive_gamma_correct_from_background()로 별도
    검증한다. 이 함수는 "crop 자체 밝기로 노출을 추정하면 안 된다"는 실패 사례를 보존하기
    위해서만 남긴다(지침 38, 실패한 실험을 삭제하지 않는다).
    """
    gamma = estimate_gamma_from_reference(crop_bgr, target_mean, gamma_min, gamma_max)
    return apply_gamma(crop_bgr, gamma)


def adaptive_gamma_correct_from_background(
    crop_bgr: np.ndarray,
    background_bgr: np.ndarray,
    target_mean: float = 128.0,
    gamma_min: float = 0.4,
    gamma_max: float = 2.5,
) -> np.ndarray:
    """EXP-030 대안 B(채택 여부 검증 대상): 노출(감마)을 crop 자신이 아니라 같은 Frame의
    배경(person bbox 밖 영역)에서 추정해 crop에 적용한다.

    adaptive_gamma_correct()가 기각된 이유(crop 자체 밝기는 옷 색상과 노출이 혼합된 신호)를
    그대로 해결하려는 설계다 - 배경은 (대체로) 옷 색상과 무관하므로 배경의 평균 밝기가
    target_mean에서 벗어난 정도가 더 순수하게 "이 Frame의 조명 노출"을 반영한다고 가정한다.
    """
    gamma = estimate_gamma_from_reference(background_bgr, target_mean, gamma_min, gamma_max)
    return apply_gamma(crop_bgr, gamma)


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


def region_dominant_color_diagnostic(
    region_bgr: np.ndarray,
    sat_min: int = 25,
    val_min: int = 35,
    val_max: int = 245,
    hue_bin_width: int = 10,
    exclude_skin: bool = True,
    min_filtered_fraction: float = 0.3,
) -> dict:
    """대안 B(채택) 로직의 진단 버전 (EXP-032, Error Analysis). region_dominant_color와
    완전히 동일한 필터링+Hue 다수결을 수행하지만, 최종 색상 하나만 반환하는 대신
    "왜 이 색이 나왔는가"를 설명하는 중간값(어떤 필터가 발동했는지, 다수결 1/2위 Hue
    bin이 얼마나 차이나는지)을 함께 반환한다. region_dominant_color는 이 함수를 감싸는
    thin wrapper로 리팩터링해 로직 중복을 피했다(동작은 100% 동일, 기존 테스트가 그대로
    통과함으로 확인).

    반환 dict:
    - color: region_dominant_color()와 동일한 최종 예측.
    - used_fallback: True면 필터 통과 픽셀이 min_filtered_fraction보다 적어 필터를
      버리고 원본 전체 픽셀로 되돌아갔다(EXP-025가 발견한 "진짜 검정 옷이 필터에 전부
      걸림" 패턴과 같은 경로).
    - filtered_fraction: 필터 통과 픽셀 비율(0~1).
    - dominant_bin / dominant_bin_count / runner_up_bin / runner_up_bin_count:
      Hue 다수결 1위/2위 bin과 그 픽셀 수(그 둘의 격차가 작으면 다수결이 불안정하다는
      신호).
    - median_h / median_s / median_v: 다수결 bin의 대표 HSV(최종 분류에 실제로 쓰인 값).
    """
    if region_bgr.size == 0:
        return {
            "color": "gray",
            "used_fallback": False,
            "filtered_fraction": 0.0,
            "dominant_bin": None,
            "dominant_bin_count": 0,
            "runner_up_bin": None,
            "runner_up_bin_count": 0,
            "median_h": None,
            "median_s": None,
            "median_v": None,
        }
    hsv = _to_hsv(region_bgr).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mask = (s >= sat_min) & (v >= val_min) & (v <= val_max)
    if exclude_skin:
        mask = mask & ~_is_skin_like(h, s, v)
    filtered = hsv[mask]
    filtered_fraction = filtered.shape[0] / hsv.shape[0]
    used_fallback = filtered_fraction < min_filtered_fraction
    if used_fallback:
        filtered = hsv
    bins = (filtered[:, 0] // hue_bin_width).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1]
    dominant_bin = int(values[order[0]])
    dominant_bin_count = int(counts[order[0]])
    runner_up_bin = int(values[order[1]]) if len(order) > 1 else None
    runner_up_bin_count = int(counts[order[1]]) if len(order) > 1 else 0
    in_bin = filtered[bins == dominant_bin]
    med_h, med_s, med_v = np.median(in_bin, axis=0)
    color = classify_hsv_pixel(float(med_h), float(med_s), float(med_v))
    return {
        "color": color,
        "used_fallback": bool(used_fallback),
        "filtered_fraction": round(float(filtered_fraction), 4),
        "dominant_bin": dominant_bin,
        "dominant_bin_count": dominant_bin_count,
        "runner_up_bin": runner_up_bin,
        "runner_up_bin_count": runner_up_bin_count,
        "median_h": round(float(med_h), 1),
        "median_s": round(float(med_s), 1),
        "median_v": round(float(med_v), 1),
    }


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
    return region_dominant_color_diagnostic(
        region_bgr,
        sat_min=sat_min,
        val_min=val_min,
        val_max=val_max,
        hue_bin_width=hue_bin_width,
        exclude_skin=exclude_skin,
        min_filtered_fraction=min_filtered_fraction,
    )["color"]


def region_dominant_color_graded_fallback_diagnostic(
    region_bgr: np.ndarray,
    sat_min: int = 25,
    val_min_steps: tuple[int, ...] = (35, 15, 0),
    val_max: int = 245,
    hue_bin_width: int = 10,
    exclude_skin: bool = True,
    min_filtered_fraction: float = 0.3,
) -> dict:
    """대안 B(EXP-033, 채택): FC-012 Next Action(EXP-032) - min_filtered_fraction 폴백
    경로 자체의 재설계.

    EXP-032가 실측한 원인(Result 3): region_dominant_color의 이분법적 폴백(필터 통과
    픽셀이 min_filtered_fraction 미만이면 필터를 완전히 버리고 원본 전체 픽셀로 복귀 -
    skin exclusion까지 함께 사라짐)이 GT=black 오분류의 35~40%를 차지한다. 그 폴백이
    자주 발동하는 이유는 val_min=35(그림자 제거용 하한)가 "그림자"와 "진짜 어두운
    검정 옷"을 구분하지 못해 함께 제외시키기 때문이다 - 둘 다 V(명도)가 낮다는 점에서는
    동일한 신호를 낸다.

    이 함수는 폴백이 발동할 때 전부 포기하는 대신 val_min_steps(기본 35->15->0)를
    따라 그림자 하한만 단계적으로 완화한다. sat_min(채도 하한)·val_max(하이라이트)·
    skin exclusion은 모든 단계에서 그대로 유지되므로, 배경 Bleed와 피부색은 계속
    걸러내면서 "진짜 검정"만 더 많이 통과시킨다. 모든 단계가 min_filtered_fraction을
    못 넘기면(crop이 실제로 거의 전부 배경/피부색) 가장 완화된 단계의 결과를 쓰고,
    그마저도 0픽셀이면 그제서야(기존 함수와 동일하게) 완전 원본으로 되돌아간다.
    """
    if region_bgr.size == 0:
        return {
            "color": "gray",
            "used_fallback": False,
            "fallback_level": None,
            "filtered_fraction": 0.0,
            "dominant_bin": None,
            "dominant_bin_count": 0,
            "runner_up_bin": None,
            "runner_up_bin_count": 0,
            "median_h": None,
            "median_s": None,
            "median_v": None,
        }
    hsv = _to_hsv(region_bgr).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    skin_mask = _is_skin_like(h, s, v) if exclude_skin else np.zeros(h.shape, dtype=bool)

    filtered = None
    chosen_level = None
    filtered_fraction = 0.0
    for val_min in val_min_steps:
        candidate_mask = (s >= sat_min) & (v >= val_min) & (v <= val_max) & ~skin_mask
        candidate = hsv[candidate_mask]
        fraction = candidate.shape[0] / hsv.shape[0]
        if fraction >= min_filtered_fraction:
            filtered, chosen_level, filtered_fraction = candidate, val_min, fraction
            break
    if filtered is None:
        val_min = val_min_steps[-1]
        candidate_mask = (s >= sat_min) & (v >= val_min) & (v <= val_max) & ~skin_mask
        filtered = hsv[candidate_mask]
        filtered_fraction = filtered.shape[0] / hsv.shape[0]
        chosen_level = val_min
        if filtered.shape[0] == 0:
            filtered = hsv
            chosen_level = None

    bins = (filtered[:, 0] // hue_bin_width).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1]
    dominant_bin = int(values[order[0]])
    dominant_bin_count = int(counts[order[0]])
    runner_up_bin = int(values[order[1]]) if len(order) > 1 else None
    runner_up_bin_count = int(counts[order[1]]) if len(order) > 1 else 0
    in_bin = filtered[bins == dominant_bin]
    med_h, med_s, med_v = np.median(in_bin, axis=0)
    color = classify_hsv_pixel(float(med_h), float(med_s), float(med_v))
    return {
        "color": color,
        "used_fallback": chosen_level != val_min_steps[0],
        "fallback_level": chosen_level,
        "filtered_fraction": round(float(filtered_fraction), 4),
        "dominant_bin": dominant_bin,
        "dominant_bin_count": dominant_bin_count,
        "runner_up_bin": runner_up_bin,
        "runner_up_bin_count": runner_up_bin_count,
        "median_h": round(float(med_h), 1),
        "median_s": round(float(med_s), 1),
        "median_v": round(float(med_v), 1),
    }


def region_dominant_color_graded_fallback(region_bgr: np.ndarray, **kwargs) -> str:
    """region_dominant_color_graded_fallback_diagnostic의 color만 반환하는 thin wrapper."""
    return region_dominant_color_graded_fallback_diagnostic(region_bgr, **kwargs)["color"]


def region_dominant_color_graded_fallback_from_pixels(pixels_bgr: np.ndarray, **kwargs) -> str:
    """region_dominant_color_from_pixels와 동일하게, 세그멘테이션 마스크로 걸러낸
    1차원 픽셀 집합(N,3)에 region_dominant_color_graded_fallback을 적용한다."""
    if pixels_bgr.size == 0:
        return "gray"
    pseudo_region = pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    return region_dominant_color_graded_fallback(pseudo_region, **kwargs)


def classify_person_attributes_with_mask_relaxed(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    head_skip_ratio: float = 0.22,
    upper_end_ratio: float = 0.55,
    foot_skip_ratio: float = 0.08,
    min_mask_fraction: float = 0.05,
    val_min: int = 10,
) -> dict[str, str]:
    """대안 A(EXP-033): classify_person_attributes_with_mask(대안 C)와 영역 분리는
    동일하게 두고, region_dominant_color의 val_min만 35->10으로 낮춘 가장 단순한
    수정이다(폴백 설계 자체는 그대로 이분법). "그림자 하한을 그냥 낮추면 되지 않을까"
    라는 가장 직관적인 첫 시도를 대안 B(graded fallback)와 비교하기 위해 둔다.
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
            return region_dominant_color(fallback_region, val_min=val_min)
        return region_dominant_color_from_pixels(crop_bgr[region_mask.astype(bool)], val_min=val_min)

    return {"upper": _resolve(upper_mask, fallback_upper), "lower": _resolve(lower_mask, fallback_lower)}


def classify_person_attributes_with_mask_and_wb_relaxed(
    crop_bgr: np.ndarray, mask: np.ndarray, **kwargs
) -> dict[str, str]:
    """classify_person_attributes_with_mask_and_wb(c_wb, production)와 동일하게
    Mask+White Balance를 적용하되, 대안 A(val_min 완화)만 추가한 변형."""
    balanced = white_balance_gray_world(crop_bgr)
    return classify_person_attributes_with_mask_relaxed(balanced, mask, **kwargs)


def classify_person_attributes_with_mask_graded(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    head_skip_ratio: float = 0.22,
    upper_end_ratio: float = 0.55,
    foot_skip_ratio: float = 0.08,
    min_mask_fraction: float = 0.05,
) -> dict[str, str]:
    """대안 B(EXP-033, 채택): classify_person_attributes_with_mask(대안 C)와 영역
    분리는 동일하게 두고, Hue 다수결의 min_filtered_fraction 폴백만
    region_dominant_color_graded_fallback(단계적 val_min 완화)으로 교체한다.
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
            return region_dominant_color_graded_fallback(fallback_region)
        return region_dominant_color_graded_fallback_from_pixels(crop_bgr[region_mask.astype(bool)])

    return {"upper": _resolve(upper_mask, fallback_upper), "lower": _resolve(lower_mask, fallback_lower)}


def classify_person_attributes_with_mask_and_wb_graded(
    crop_bgr: np.ndarray, mask: np.ndarray, **kwargs
) -> dict[str, str]:
    """classify_person_attributes_with_mask_and_wb(c_wb, production)와 동일하게
    Mask+White Balance를 적용하되, 대안 B(graded fallback)만 추가한 변형. EXP-033에서
    production 승격 여부를 결정하기 위한 비교 대상이다.
    """
    balanced = white_balance_gray_world(crop_bgr)
    return classify_person_attributes_with_mask_graded(balanced, mask, **kwargs)


def frame_mean_brightness(frame_bgr: np.ndarray) -> float:
    """Frame 전체(HSV V 채널)의 평균 밝기 (EXP-031, FC-012 Next Action: pixel-level 보정
    대신 "구간 단위 낮은 신뢰도 표시"로의 설계 전환).

    EXP-030의 adaptive_gamma_correct()가 기각된 이유는 person crop 자신의 평균 밝기가
    "옷 색상"과 "조명 노출"이 섞인 신호였기 때문이다(검정 옷과 흰 옷의 crop 밝기 범위가
    겹침). 이 함수는 crop이 아니라 Frame 전체(배경+모든 object 포함)의 밝기를 재는 것이라
    같은 confound가 생기지 않는다 - Frame 전체가 특정 한 사람의 옷 색상에 좌우되지
    않으므로, "이 Frame이 관측된 순간의 조명 노출 수준"을 더 순수하게 반영한다.

    **EXP-031에서 "Accuracy를 예측하는 신호로는 기각"됨(중요)**: confound가 없다는
    가정은 맞았지만(위 설명), 실측 결과 이 신호로 낮은 신뢰도를 표시한 구간(flagged)의
    Accuracy가 오히려 표시하지 않은 구간(unflagged)보다 높았다(coco128 n=21,
    Alt A 22.3%<30.6%, Alt B 15.3%<31.9%) - 정반대 방향이다. 원인은 method "b"
    분류기의 실제 실패가 노출(밝기) 문제보다 warm_cast/cool_cast(채널 Cast, 밝기는
    거의 안 변함) 문제에 더 크게 좌우되기 때문이다(PAR-022 참고). 이 함수 자체(Frame
    밝기 측정)는 올바르게 동작하지만, "밝기 편차 = 낮은 신뢰도"라는 가정이 이
    classifier에는 맞지 않았다 - 다른 목적(예: 노출 자체를 보정하는 EXP-030의
    배경 기반 Gamma)에는 여전히 유효하게 쓰인다.
    """
    if frame_bgr.size == 0:
        return 0.0
    v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(v.mean())


def classify_exposure_level(mean_v: float, low_threshold: float, high_threshold: float) -> str:
    """Frame 평균 밝기(mean_v)를 "low_light"/"strong_light"/"normal" 중 하나로 분류한다.

    low_threshold/high_threshold는 임의로 정하지 않고 실제 Dataset의 밝기 분포를 측정해
    정의해야 한다(지침 22와 동일 원칙 - EXP-031 참고). 이 함수 자체는 순수 함수로 유지해
    Threshold 값(Alt A 고정값 vs Alt B Dataset 기반 값)을 자유롭게 바꿔 비교할 수 있게 한다.
    """
    if mean_v < low_threshold:
        return "low_light"
    if mean_v > high_threshold:
        return "strong_light"
    return "normal"


def frame_channel_cast_deviation(frame_bgr: np.ndarray) -> float:
    """Frame 전체 채널 평균이 중립 회색(B=G=R)에서 벗어난 정도 (EXP-031, Alt C: exposure
    신호[frame_mean_brightness]만으로는 FC-012의 warm_cast/cool_cast(채널 Cast)류
    실패를 포착하지 못해 추가한 보조 신호). white_balance_gray_world의 게인 계산과
    동일한 가정(Gray World)을 쓰되, 보정이 아니라 "이 Frame이 Cast가 심한가"를 재는
    측정값만 반환한다.

    **EXP-031에서 조건별 분리도를 실측한 결과(기각)**: 실제 COCO 사진은 합성 조명을
    가하지 않은 normal 조건에서도 이미 채널 편차가 크고 넓게 퍼져 있어(normal
    mean=0.173, std=0.138, cool_cast mean=0.206과 거의 겹침) normal과 cool_cast를
    분리하지 못했다. exposure 신호와 결합(Alt C)해도 Accuracy를 예측하는 데 도움이
    되지 않았다(PAR-022 참고). 실패 사례 보존을 위해 함수는 남긴다(지침 38).
    """
    if frame_bgr.size == 0:
        return 0.0
    channel_means = frame_bgr.astype(np.float32).reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()
    if overall_mean < 1e-6:
        return 0.0
    deviation = np.abs(channel_means - overall_mean) / overall_mean
    return float(deviation.max())


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


def region_dominant_color_diagnostic_from_pixels(pixels_bgr: np.ndarray, **kwargs) -> dict:
    """region_dominant_color_from_pixels의 진단 버전 (EXP-032). 세그멘테이션 마스크로
    걸러낸 픽셀 집합에 region_dominant_color_diagnostic을 그대로 적용한다."""
    if pixels_bgr.size == 0:
        return region_dominant_color_diagnostic(pixels_bgr, **kwargs)
    pseudo_region = pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    return region_dominant_color_diagnostic(pseudo_region, **kwargs)


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


def classify_person_attributes_with_mask_diagnostic(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    head_skip_ratio: float = 0.22,
    upper_end_ratio: float = 0.55,
    foot_skip_ratio: float = 0.08,
    min_mask_fraction: float = 0.05,
) -> dict:
    """classify_person_attributes_with_mask(대안 C)의 진단 버전 (EXP-032, Error Analysis -
    production이 실제로 쓰는 c_wb의 오분류 원인을 region_dominant_color_diagnostic 수준까지
    들여다본다). 각 region(upper/lower)에 대해 "마스크를 실제로 썼는가 사각형 폴백을
    썼는가"(used_mask)와 Hue 다수결 진단(region_dominant_color_diagnostic 반환값)을
    합쳐서 반환한다.
    """
    if crop_bgr.size == 0 or mask.size == 0:
        empty = region_dominant_color_diagnostic(np.zeros((0, 0, 3), dtype=np.uint8))
        empty["used_mask"] = False
        return {"upper": dict(empty), "lower": dict(empty)}
    upper_mask, lower_mask = split_upper_lower_by_mask(mask, head_skip_ratio, upper_end_ratio, foot_skip_ratio)
    fallback_upper, fallback_lower = split_upper_lower(
        crop_bgr,
        head_skip_ratio=head_skip_ratio,
        upper_end_ratio=upper_end_ratio,
        foot_skip_ratio=foot_skip_ratio,
        side_margin_ratio=0.12,
    )

    def _resolve(region_mask: np.ndarray, fallback_region: np.ndarray) -> dict:
        total = region_mask.size
        if total == 0 or region_mask.sum() < min_mask_fraction * total:
            diag = region_dominant_color_diagnostic(fallback_region)
            diag["used_mask"] = False
            return diag
        diag = region_dominant_color_diagnostic_from_pixels(crop_bgr[region_mask.astype(bool)])
        diag["used_mask"] = True
        return diag

    return {"upper": _resolve(upper_mask, fallback_upper), "lower": _resolve(lower_mask, fallback_lower)}


def classify_person_attributes_with_mask_and_wb_diagnostic(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    **kwargs,
) -> dict:
    """classify_person_attributes_with_mask_and_wb(c_wb, production이 실제로 쓰는 method)의
    진단 버전 (EXP-032)."""
    balanced = white_balance_gray_world(crop_bgr)
    return classify_person_attributes_with_mask_diagnostic(balanced, mask, **kwargs)


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
    if method == "b_gamma":
        # EXP-030: method b 앞에 adaptive_gamma_correct를 추가한 실험용 변형.
        # WB(b_wb)가 해결하지 못한 strong_light(전역 노출 과다)를 겨냥한다.
        corrected = adaptive_gamma_correct(crop_bgr)
        upper, lower = split_upper_lower(
            corrected,
            head_skip_ratio=0.22,
            upper_end_ratio=0.55,
            foot_skip_ratio=0.08,
            side_margin_ratio=0.12,
        )
        return {"upper": region_dominant_color(upper), "lower": region_dominant_color(lower)}
    if method == "b_wb_gamma":
        # EXP-030: Gamma(노출 보정) 다음에 WB(채널 Cast 보정)를 순서대로 적용한 결합형.
        # 두 보정이 서로 다른 원인(노출 vs 채널 Cast)을 겨냥하므로 순서를 Gamma -> WB로
        # 고정한다(노출을 먼저 정상화해야 Gray World의 "전체 평균이 중립 회색"이라는
        # 가정이 더 잘 맞는다).
        corrected = white_balance_gray_world(adaptive_gamma_correct(crop_bgr))
        upper, lower = split_upper_lower(
            corrected,
            head_skip_ratio=0.22,
            upper_end_ratio=0.55,
            foot_skip_ratio=0.08,
            side_margin_ratio=0.12,
        )
        return {"upper": region_dominant_color(upper), "lower": region_dominant_color(lower)}
    if method == "b_relaxed":
        # EXP-033 대안 A: method b와 영역 분리는 동일, region_dominant_color의
        # val_min만 35->10으로 낮춘 가장 단순한 수정(폴백 설계 자체는 이분법 그대로).
        upper, lower = split_upper_lower(
            crop_bgr,
            head_skip_ratio=0.22,
            upper_end_ratio=0.55,
            foot_skip_ratio=0.08,
            side_margin_ratio=0.12,
        )
        return {
            "upper": region_dominant_color(upper, val_min=10),
            "lower": region_dominant_color(lower, val_min=10),
        }
    if method == "b_graded":
        # EXP-033 대안 B(채택): method b와 영역 분리는 동일, Hue 다수결의
        # min_filtered_fraction 폴백만 region_dominant_color_graded_fallback(단계적
        # val_min 완화 35->15->0)으로 교체한다.
        upper, lower = split_upper_lower(
            crop_bgr,
            head_skip_ratio=0.22,
            upper_end_ratio=0.55,
            foot_skip_ratio=0.08,
            side_margin_ratio=0.12,
        )
        return {
            "upper": region_dominant_color_graded_fallback(upper),
            "lower": region_dominant_color_graded_fallback(lower),
        }
    raise ValueError(f"unknown method: {method}")
