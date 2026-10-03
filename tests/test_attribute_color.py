import numpy as np
import pytest

from src.attributes.color import (
    COLOR_BUCKETS,
    classify_hsv_pixel,
    classify_person_attributes,
    region_dominant_color,
    region_mean_color,
    split_upper_lower,
    white_balance_gray_world,
    whole_bbox_mean_color,
)


def solid_bgr(color_bgr: tuple[int, int, int], h: int = 100, w: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


# 대표적인 BGR 색상 (OpenCV는 BGR 순서)
BLACK = (10, 10, 10)
WHITE = (240, 240, 240)
RED = (20, 20, 200)
BLUE = (200, 60, 20)
GREEN = (20, 160, 20)


class TestClassifyHsvPixel:
    def test_dark_pixel_is_black_regardless_of_hue(self):
        assert classify_hsv_pixel(h=100, s=200, v=10) == "black"

    def test_low_saturation_bright_is_white(self):
        assert classify_hsv_pixel(h=0, s=5, v=220) == "white"

    def test_low_saturation_mid_value_is_gray(self):
        assert classify_hsv_pixel(h=0, s=5, v=120) == "gray"

    def test_red_hue_near_zero(self):
        assert classify_hsv_pixel(h=2, s=180, v=180) == "red"

    def test_red_hue_wraps_near_max(self):
        assert classify_hsv_pixel(h=177, s=180, v=180) == "red"

    def test_blue_hue(self):
        assert classify_hsv_pixel(h=110, s=180, v=180) == "blue"

    def test_green_hue(self):
        assert classify_hsv_pixel(h=60, s=180, v=180) == "green"


class TestWholeBboxMeanColor:
    def test_solid_red_crop_classified_red(self):
        assert whole_bbox_mean_color(solid_bgr(RED)) == "red"

    def test_solid_blue_crop_classified_blue(self):
        assert whole_bbox_mean_color(solid_bgr(BLUE)) == "blue"

    def test_empty_crop_defaults_to_gray(self):
        assert whole_bbox_mean_color(np.zeros((0, 0, 3), dtype=np.uint8)) == "gray"

    def test_two_tone_crop_blends_to_neither_color(self):
        """Baseline의 핵심 한계: 상의(검정)+하의(흰색) crop을 통째로 평균 내면
        둘 중 어느 쪽 색도 아닌 값(gray)으로 블렌딩된다 (상/하 구분 불가)."""
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:50, :] = BLACK
        img[50:, :] = WHITE
        result = whole_bbox_mean_color(img)
        assert result not in ("black", "white")


class TestSplitUpperLower:
    def test_default_split_is_half_height(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        upper, lower = split_upper_lower(img, upper_end_ratio=0.5)
        assert upper.shape[0] == 50
        assert lower.shape[0] == 50

    def test_head_skip_removes_top_rows(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        upper, _ = split_upper_lower(img, head_skip_ratio=0.2, upper_end_ratio=0.5)
        assert upper.shape[0] == 30  # 100 * (0.5 - 0.2)

    def test_side_margin_shrinks_width(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        upper, lower = split_upper_lower(img, side_margin_ratio=0.1)
        assert upper.shape[1] == 48  # 60 - 2*6
        assert lower.shape[1] == 48


class TestRegionMeanColor:
    def test_solid_region(self):
        assert region_mean_color(solid_bgr(GREEN)) == "green"

    def test_empty_region_defaults_to_gray(self):
        assert region_mean_color(np.zeros((0, 0, 3), dtype=np.uint8)) == "gray"


class TestRegionDominantColor:
    def test_solid_region(self):
        assert region_dominant_color(solid_bgr(RED)) == "red"

    def test_robust_to_minority_background_bleed(self):
        """대안 B의 핵심 설계 목표: 영역의 소수 픽셀이 배경(검은 테두리)이어도
        다수 픽셀(실제 옷 색상)로 분류되어야 한다. 대안 A(단순 평균)는 이 비율에서
        이미 검은색 쪽으로 끌려갈 수 있다."""
        img = solid_bgr(BLUE, h=100, w=100)
        img[:15, :] = BLACK  # 상단 15%를 배경(검은 테두리)으로 오염
        assert region_dominant_color(img) == "blue"

    def test_robust_to_shadow_outliers(self):
        """그림자(매우 어두운 픽셀)가 섞여도 다수 픽셀 색상으로 분류되어야 한다."""
        img = solid_bgr(GREEN, h=100, w=100)
        img[80:, :] = (0, 0, 0)  # 하단 20%는 그림자(거의 검정)
        assert region_dominant_color(img) == "green"

    def test_all_shadow_falls_back_to_unfiltered(self):
        """필터링 후 남는 픽셀이 0개면(전체가 그림자) 예외 없이 fallback해야 한다."""
        img = solid_bgr((5, 5, 5), h=50, w=50)
        result = region_dominant_color(img)
        assert result == "black"

    def test_empty_region_defaults_to_gray(self):
        assert region_dominant_color(np.zeros((0, 0, 3), dtype=np.uint8)) == "gray"

    def test_skin_tone_minority_does_not_flip_black_garment(self):
        """얼굴/손 등 피부색 픽셀이 소수 섞여도(머리를 완전히 못 걸러낸 클로즈업 crop
        상황을 재현) 검정 옷으로 분류되어야 한다 (EXP-025에서 실제로 발견한 문제)."""
        img = solid_bgr(BLACK, h=100, w=100)
        skin_bgr = (120, 170, 210)  # 전형적인 피부색 근사 (BGR)
        img[:20, :] = skin_bgr  # 상단 20%는 피부(얼굴 하단/턱선)로 오염
        assert region_dominant_color(img) == "black"

    def test_exclude_skin_false_disables_skin_filter(self):
        img = solid_bgr((120, 170, 210), h=50, w=50)  # 전부 피부색
        assert region_dominant_color(img, exclude_skin=False) != "gray"
        # 피부색 전체 영역에서 skin 필터를 켜면 걸러낼 픽셀이 없어 완화 단계로 fallback
        result = region_dominant_color(img, exclude_skin=True)
        assert result in COLOR_BUCKETS


class TestWhiteBalanceGrayWorld:
    """EXP-026 (FC-012): Gray World White Balance. 채널 간 Cast를 줄이는지는 직접
    검증하고, FC-012(검정 옷이 blue로 오분류)가 실제로 줄었는지는 별도로 측정했다
    (결과: 겨냥한 검정 GT 셀 정확도는 불변 - experiment.md Decision 참고). 여기서는
    "Cast를 줄인다"는 WB 자체의 좁은 책임만 Unit Test로 고정한다.
    """

    def test_neutral_gray_crop_stays_achromatic(self):
        img = solid_bgr((128, 128, 128), h=50, w=50)
        result = white_balance_gray_world(img)
        assert region_mean_color(result) in ("gray", "white", "black")

    def test_reduces_blue_cast_on_otherwise_neutral_region(self):
        """R/G는 그대로, B만 인위적으로 올린(한색 Cast) 영역은 WB 후 채널 간 평균
        차이가 원본보다 줄어들어야 한다(완전히 0이 될 필요는 없음)."""
        img = solid_bgr((180, 90, 80), h=50, w=50)  # B가 R/G보다 훨씬 큼 (cool_cast 모사)
        original_spread = max(180, 90, 80) - min(180, 90, 80)
        corrected = white_balance_gray_world(img)
        b, g, r = corrected[0, 0].astype(int)
        corrected_spread = max(b, g, r) - min(b, g, r)
        assert corrected_spread < original_spread

    def test_empty_region_passthrough(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = white_balance_gray_world(empty)
        assert result.shape == empty.shape

    def test_gain_clip_prevents_extreme_amplification(self):
        """한 채널이 거의 0에 가까우면(거의 포화된 단색) 게인이 무한대로 커질 수
        있으므로 gain_max로 clip되어야 한다 (어두운 단색 crop이 WB 후 날아가 버리는
        것을 방지)."""
        img = solid_bgr((1, 1, 90), h=50, w=50)  # R만 매우 큼, B/G는 거의 0
        corrected = white_balance_gray_world(img, gain_min=0.3, gain_max=3.0)
        assert corrected.max() <= 90 * 3.0 + 1  # clip 범위를 벗어난 폭주가 없어야 함


class TestClassifyPersonAttributesBWb:
    def test_method_b_wb_returns_valid_colors(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:55, :] = WHITE
        img[55:, :] = BLUE
        result = classify_person_attributes(img, method="b_wb")
        assert result["upper"] in COLOR_BUCKETS
        assert result["lower"] in COLOR_BUCKETS

    def test_method_b_unaffected_by_white_balance_addition(self):
        """production 기본값("b")은 EXP-026 이후에도 동작이 바뀌지 않아야 한다
        (White Balance는 "b_wb"에서만 실험적으로 적용됨 - Decision 참고)."""
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:55, :] = WHITE
        img[55:, :] = BLUE
        result = classify_person_attributes(img, method="b")
        assert result["upper"] == "white"
        assert result["lower"] == "blue"


class TestClassifyPersonAttributes:
    def test_two_tone_crop_baseline_cannot_separate(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:50, :] = WHITE
        img[50:, :] = BLUE
        result = classify_person_attributes(img, method="baseline")
        assert result["upper"] == result["lower"]

    def test_two_tone_crop_method_b_separates_upper_lower(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:55, :] = WHITE
        img[55:, :] = BLUE
        result = classify_person_attributes(img, method="b")
        assert result["upper"] == "white"
        assert result["lower"] == "blue"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            classify_person_attributes(solid_bgr(RED), method="z")
