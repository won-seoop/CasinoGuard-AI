import numpy as np
import pytest

from src.attributes.color import (
    COLOR_BUCKETS,
    classify_hsv_pixel,
    classify_person_attributes,
    region_dominant_color,
    region_mean_color,
    split_upper_lower,
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
