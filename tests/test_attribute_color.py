import cv2
import numpy as np
import pytest

from src.attributes.color import (
    COLOR_BUCKETS,
    adaptive_gamma_correct,
    adaptive_gamma_correct_from_background,
    apply_gamma,
    classify_exposure_level,
    classify_hsv_pixel,
    classify_person_attributes,
    classify_person_attributes_with_mask,
    classify_person_attributes_with_mask_and_wb,
    classify_person_attributes_with_mask_and_wb_diagnostic,
    classify_person_attributes_with_mask_and_wb_graded,
    classify_person_attributes_with_mask_and_wb_relaxed,
    classify_person_attributes_with_mask_diagnostic,
    classify_person_attributes_with_mask_graded,
    classify_person_attributes_with_mask_relaxed,
    estimate_gamma_from_reference,
    frame_channel_cast_deviation,
    frame_mean_brightness,
    gamma_background_then_white_balance,
    gamma_background_then_white_balance_masked,
    region_color_signal,
    region_dominant_color,
    region_dominant_color_diagnostic,
    region_dominant_color_from_pixels,
    region_dominant_color_graded_fallback,
    region_dominant_color_graded_fallback_diagnostic,
    region_dominant_color_graded_fallback_from_pixels,
    region_mean_color,
    split_upper_lower,
    split_upper_lower_by_mask,
    white_balance_gray_world,
    white_balance_gray_world_diagnostic,
    white_balance_gray_world_masked,
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


class TestRegionDominantColorDiagnostic:
    """EXP-032 (Error Analysis): region_dominant_color_diagnostic이 region_dominant_color와
    동일한 최종 색상을 내면서, 중간값(fallback 여부/margin)도 올바르게 보고하는지 검증한다.
    region_dominant_color 자체는 이 진단 함수를 감싸는 thin wrapper로 리팩터링했으므로
    (동작 비변경, 기존 TestRegionDominantColor 전부 통과로 확인), 여기서는 진단 전용
    필드만 추가로 검증한다."""

    def test_color_matches_non_diagnostic_function(self):
        img = solid_bgr(RED, h=100, w=100)
        diag = region_dominant_color_diagnostic(img)
        assert diag["color"] == region_dominant_color(img) == "red"

    def test_solid_region_has_no_runner_up_and_full_margin(self):
        """단색 영역은 Hue bin이 하나뿐이므로 runner_up이 없어야 한다(동전 던지기가
        아니라 완전히 확정적인 다수결이라는 신호)."""
        img = solid_bgr(RED, h=50, w=50)
        diag = region_dominant_color_diagnostic(img)
        assert diag["runner_up_bin"] is None
        assert diag["runner_up_bin_count"] == 0
        assert diag["used_fallback"] is False
        assert diag["filtered_fraction"] == pytest.approx(1.0)

    def test_all_shadow_reports_fallback_used(self):
        """EXP-032가 실측에서 발견한 핵심 경로: 필터 통과 픽셀이 거의 없으면(진짜 검정)
        used_fallback=True로 명시적으로 보고해야 한다 - region_dominant_color는 이
        신호를 버리고 색상만 반환하므로, 어떤 예측이 fallback 경로에서 나왔는지는 이
        진단 함수로만 구분할 수 있다."""
        img = solid_bgr((5, 5, 5), h=50, w=50)
        diag = region_dominant_color_diagnostic(img)
        assert diag["used_fallback"] is True
        assert diag["filtered_fraction"] == pytest.approx(0.0)
        assert diag["color"] == "black"

    def test_empty_region_defaults_to_gray_without_crashing(self):
        diag = region_dominant_color_diagnostic(np.zeros((0, 0, 3), dtype=np.uint8))
        assert diag["color"] == "gray"
        assert diag["used_fallback"] is False
        assert diag["dominant_bin"] is None

    def test_narrow_margin_reported_for_evenly_split_hues(self):
        """두 Hue bin이 거의 동률로 나뉘면 margin이 좁다는 것을 runner_up_bin_count로
        확인할 수 있어야 한다(다수결이 "거의 동전 던지기"였다는 EXP-032의 분석 지표)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = RED
        img[50:, :] = GREEN
        diag = region_dominant_color_diagnostic(img)
        assert diag["runner_up_bin_count"] is not None and diag["runner_up_bin_count"] > 0
        margin = diag["dominant_bin_count"] / (diag["dominant_bin_count"] + diag["runner_up_bin_count"])
        assert margin == pytest.approx(0.5, abs=0.05)


class TestRegionDominantColorGradedFallback:
    """EXP-033 대안 B(채택): min_filtered_fraction 폴백을 이분법(필터 통과 미달 ->
    완전 포기하고 원본 전체로 복귀) 대신 val_min을 단계적으로(35->15->0) 완화하는
    다단 폴백으로 바꾼 함수. EXP-032가 실측한 원인(그림자 제거용 val_min=35가 그림자와
    진짜 어두운 옷을 구분하지 못해 함께 제외시키고, 그 결과 폴백이 과도하게 자주
    발동해 오분류의 35~40%를 차지함)을 직접 겨냥한다."""

    def test_solid_region_matches_binary_function_when_no_fallback_needed(self):
        img = solid_bgr(RED, h=50, w=50)
        assert region_dominant_color_graded_fallback(img) == region_dominant_color(img) == "red"

    def test_color_matches_own_diagnostic_function(self):
        img = solid_bgr(RED, h=50, w=50)
        diag = region_dominant_color_graded_fallback_diagnostic(img)
        assert diag["color"] == region_dominant_color_graded_fallback(img) == "red"
        assert diag["used_fallback"] is False
        assert diag["fallback_level"] == 35  # 기본 val_min_steps[0]과 동일

    def test_dark_but_saturated_pixel_rescued_by_intermediate_step(self):
        """V=20(기본 val_min=35 미달)이지만 양자화 Noise로 S=179까지 뜨는 어두운
        픽셀(진짜 검정 옷에서 흔한 패턴, EXP-032 Result 분석)은 val_min=15 단계에서
        구제되어야 한다 - 완전 raw로 가지 않고도 필터를 통과한다."""
        img = solid_bgr((8, 6, 20), h=50, w=50)  # H=176,S=179,V=20 (EXP-033 측정값)
        diag = region_dominant_color_graded_fallback_diagnostic(img)
        assert diag["fallback_level"] == 15
        assert diag["used_fallback"] is True
        assert diag["filtered_fraction"] == pytest.approx(1.0)
        assert diag["color"] == "black"

    def test_rescues_dark_garment_from_skin_contamination_that_flips_binary_fallback(self):
        """핵심 회귀 시나리오(EXP-033): 진짜 어두운 옷(V=20)이 서로 다른 Hue로 흩어져
        있고(단일 Hue bin이 과반을 못 차지함) 피부색이 30% 섞이면, 이분법적 폴백은
        전부 원본으로 돌아가 피부색 Hue bin(단일 bin에 30%, 어두운 옷의 각 하위
        bin보다 큼)이 다수결을 가져가 "orange"로 오분류한다. 단계적 폴백은 val_min=15
        단계에서 피부색만 걸러낸 채(skin exclusion은 모든 단계에서 유지) 어두운
        픽셀만으로 다수결을 돌려 "black"을 올바르게 유지해야 한다."""
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[0:17, :] = (8, 6, 20)  # 어두운 옷, Hue 1
        img[17:34, :] = (20, 6, 8)  # 어두운 옷, Hue 2
        img[34:52, :] = (6, 20, 8)  # 어두운 옷, Hue 3
        img[52:70, :] = (8, 20, 6)  # 어두운 옷, Hue 4 (옷 전체 70%, 4개 Hue bin으로 분산)
        img[70:100, :] = (120, 170, 210)  # 피부색 30% (단일 Hue bin)

        assert region_dominant_color(img) == "orange"  # 기존 이분법 폴백은 오분류
        assert region_dominant_color_graded_fallback(img) == "black"  # 단계적 폴백은 보존

    def test_pure_zero_saturation_shadow_still_falls_back_to_full_raw(self):
        """채도가 0인(순수 회색) 영역은 val_min을 아무리 낮춰도 sat_min(채도 하한)
        필터 자체에 걸려 모든 단계에서 0픽셀이 남는다 - 이 경우에는 기존 함수와
        동일하게 완전 원본으로 되돌아가는 최후 수단이 그대로 동작해야 한다(회귀 방지,
        TestRegionDominantColor.test_all_shadow_falls_back_to_unfiltered와 동일 입력)."""
        img = solid_bgr((5, 5, 5), h=50, w=50)
        diag = region_dominant_color_graded_fallback_diagnostic(img)
        assert diag["fallback_level"] is None
        assert diag["used_fallback"] is True
        assert diag["color"] == "black"

    def test_empty_region_defaults_to_gray_without_crashing(self):
        diag = region_dominant_color_graded_fallback_diagnostic(np.zeros((0, 0, 3), dtype=np.uint8))
        assert diag["color"] == "gray"
        assert diag["fallback_level"] is None
        assert diag["used_fallback"] is False


class TestRegionDominantColorGradedFallbackFromPixels:
    def test_flat_pixel_array_classified_same_as_2d_region(self):
        region = solid_bgr((8, 6, 20), h=50, w=50)
        pixels = region.reshape(-1, 3)
        assert region_dominant_color_graded_fallback_from_pixels(pixels) == region_dominant_color_graded_fallback(
            region
        )

    def test_empty_pixels_defaults_to_gray(self):
        assert region_dominant_color_graded_fallback_from_pixels(np.zeros((0, 3), dtype=np.uint8)) == "gray"


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


class TestWhiteBalanceGrayWorldDiagnostic:
    """EXP-040: white_balance_gray_world_diagnostic이 white_balance_gray_world와
    동일한 보정 결과를 만들어내는 thin wrapper임을 고정한다(region_dominant_color_
    diagnostic과 동일한 패턴, EXP-032)."""

    def test_reconstructs_same_corrected_image_as_non_diagnostic_function(self):
        img = solid_bgr((180, 90, 80), h=50, w=50)
        diag = white_balance_gray_world_diagnostic(img)
        gains = np.array(diag["gains_after_clip"], dtype=np.float32)
        reconstructed = np.clip(img.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        assert np.array_equal(reconstructed, white_balance_gray_world(img))

    def test_empty_region_returns_identity_gains(self):
        diag = white_balance_gray_world_diagnostic(np.zeros((0, 0, 3), dtype=np.uint8))
        assert diag["gains_after_clip"] == [1.0, 1.0, 1.0]
        assert diag["n_channels_clipped"] == 0

    def test_reports_clipped_channel_count_for_near_monochrome_crop(self):
        """채널 하나가 거의 0에 가까우면(near_monochrome) 그 채널의 clip 전 게인이
        gain_max를 초과해 clip이 발동해야 한다(EXP-040이 로깅하는 신호 자체의 정확성)."""
        img = solid_bgr((1, 1, 90), h=50, w=50)  # R만 매우 큼, B/G는 거의 0
        diag = white_balance_gray_world_diagnostic(img, gain_min=0.3, gain_max=3.0)
        assert diag["n_channels_clipped"] >= 1
        assert max(diag["gains_before_clip"]) > 3.0

    def test_neutral_gray_crop_has_no_clipped_channels_and_near_unit_gain(self):
        img = solid_bgr((128, 128, 128), h=50, w=50)
        diag = white_balance_gray_world_diagnostic(img)
        assert diag["n_channels_clipped"] == 0
        assert all(abs(g - 1.0) < 1e-3 for g in diag["gains_after_clip"])


class TestRegionColorSignal:
    """EXP-027: '무채색 확신도' fallback을 설계하기 전에, 그 신호(채도/raw BGR
    표준편차) 계산 자체를 고정해두는 Regression Test. 신호가 achromatic/chromatic을
    실제로 분리하지 못한다는 결론(PAR-018)은 coco128 실측 데이터 기반이라 여기서
    재현하지 않지만, 극단적인 합성 케이스(순수 무채색 vs 순수 채도)에서 신호가 방향성은
    맞게 움직이는지는 고정해 둔다.
    """

    def test_empty_region_returns_zero_signal(self):
        result = region_color_signal(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result == {"mean_saturation": 0.0, "bgr_channel_std": 0.0}

    def test_pure_gray_has_zero_saturation(self):
        img = solid_bgr((128, 128, 128), h=50, w=50)
        result = region_color_signal(img)
        assert result["mean_saturation"] == pytest.approx(0.0, abs=1e-6)
        assert result["bgr_channel_std"] == pytest.approx(0.0, abs=1e-6)

    def test_saturated_color_has_nonzero_saturation(self):
        img = solid_bgr(RED, h=50, w=50)
        result = region_color_signal(img)
        assert result["mean_saturation"] > 100.0

    def test_mixed_region_has_nonzero_bgr_std(self):
        """영역 내에 서로 다른 색(배경 Bleed 등)이 섞이면 채널 표준편차가 0보다
        커야 한다 - 단색 영역(표준편차 0)과 구분되는 기본 성질."""
        img = solid_bgr(BLACK, h=100, w=50)
        img[:20, :] = WHITE
        result = region_color_signal(img)
        assert result["bgr_channel_std"] > 0.0


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


class TestRegionDominantColorFromPixels:
    def test_flat_pixel_array_classified_same_as_2d_region(self):
        """(N,3) 픽셀 배열을 넣어도 동등한 2D 영역과 같은 결과가 나와야 한다
        (reshape 래퍼가 region_dominant_color와 동치임을 고정)."""
        region = solid_bgr(BLUE, h=40, w=40)
        pixels = region.reshape(-1, 3)
        assert region_dominant_color_from_pixels(pixels) == region_dominant_color(region)

    def test_empty_pixels_defaults_to_gray(self):
        assert region_dominant_color_from_pixels(np.zeros((0, 3), dtype=np.uint8)) == "gray"


class TestSplitUpperLowerByMask:
    def test_mask_split_matches_vertical_ratio(self):
        mask = np.ones((100, 60), dtype=bool)
        upper_mask, lower_mask = split_upper_lower_by_mask(
            mask, head_skip_ratio=0.0, upper_end_ratio=0.5, foot_skip_ratio=0.0
        )
        assert upper_mask.sum() == 50 * 60
        assert lower_mask.sum() == 50 * 60

    def test_false_outside_mask_is_excluded(self):
        """마스크가 False인(배경) 픽셀은 좌우 마진 없이도 상/하 영역에서 제외돼야 한다."""
        mask = np.zeros((100, 60), dtype=bool)
        mask[:, 20:40] = True  # 중앙 폭만 person
        upper_mask, lower_mask = split_upper_lower_by_mask(mask, upper_end_ratio=0.5)
        assert upper_mask[:, :20].sum() == 0
        assert upper_mask[:, 40:].sum() == 0


class TestClassifyPersonAttributesWithMask:
    def test_mask_excludes_background_bleed_that_defeats_method_b(self):
        """person이 bbox 폭의 20%만 차지하고 나머지 80%가 다른 Hue(빨강) 배경이면,
        method b의 side_margin_ratio=0.12(좌우 12%)로는 배경을 다 걷어내지 못해 Hue
        다수결이 배경 쪽으로 넘어간다. 정확한 person mask를 쓰는 대안 C는 배경 폭과
        무관하게 실제 옷 색(파랑)을 올바르게 분류해야 한다."""
        img = solid_bgr(RED, h=100, w=100)
        img[:, 40:60] = BLUE  # person(파랑)은 중앙 20%뿐, 나머지 80%는 배경(빨강)
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 40:60] = True

        method_b_result = classify_person_attributes(img, method="b")
        mask_result = classify_person_attributes_with_mask(img, mask)
        assert mask_result["upper"] == "blue"
        assert mask_result["lower"] == "blue"
        assert method_b_result["upper"] == "red"  # method b는 여전히 배경(빨강)에 속음

    def test_mask_below_min_fraction_falls_back_to_method_b_region(self):
        """세그멘테이션이 거의 실패해 마스크 픽셀이 거의 없으면(occlusion 등) method b와
        동일한 고정 비율 사각형 영역으로 폴백해야 한다(마스크를 과신하지 않음)."""
        img = solid_bgr(GREEN, h=100, w=100)
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True  # 거의 비어 있는 마스크
        result = classify_person_attributes_with_mask(img, mask, min_mask_fraction=0.05)
        fallback = classify_person_attributes(img, method="b")
        assert result == fallback

    def test_empty_crop_defaults_to_gray(self):
        result = classify_person_attributes_with_mask(
            np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool)
        )
        assert result == {"upper": "gray", "lower": "gray"}


class TestClassifyPersonAttributesWithMaskDiagnostic:
    """EXP-032: classify_person_attributes_with_mask_diagnostic이 production(c_wb)이
    실제로 마스크를 썼는지 사각형 폴백을 썼는지(used_mask)를 region별로 올바르게
    구분해서 보고하는지 검증한다."""

    def test_color_matches_non_diagnostic_function_when_mask_used(self):
        img = solid_bgr(RED, h=100, w=100)
        img[:, 40:60] = BLUE
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 40:60] = True

        plain = classify_person_attributes_with_mask(img, mask)
        diag = classify_person_attributes_with_mask_diagnostic(img, mask)
        assert diag["upper"]["color"] == plain["upper"] == "blue"
        assert diag["lower"]["color"] == plain["lower"] == "blue"
        assert diag["upper"]["used_mask"] is True
        assert diag["lower"]["used_mask"] is True

    def test_used_mask_false_when_mask_fraction_too_low(self):
        img = solid_bgr(GREEN, h=100, w=100)
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True  # 거의 비어 있는 마스크 -> 사각형 폴백으로 되돌아가야 함
        diag = classify_person_attributes_with_mask_diagnostic(img, mask, min_mask_fraction=0.05)
        assert diag["upper"]["used_mask"] is False
        assert diag["lower"]["used_mask"] is False
        assert diag["upper"]["color"] == "green"

    def test_empty_crop_defaults_to_gray_without_crashing(self):
        diag = classify_person_attributes_with_mask_diagnostic(
            np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool)
        )
        assert diag["upper"]["color"] == "gray"
        assert diag["lower"]["color"] == "gray"


class TestClassifyPersonAttributesWithMaskAndWbDiagnostic:
    def test_matches_non_diagnostic_wb_function(self):
        img = solid_bgr(RED, h=100, w=100)
        img[:, 40:60] = BLUE
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 40:60] = True

        plain = classify_person_attributes_with_mask_and_wb(img, mask)
        diag = classify_person_attributes_with_mask_and_wb_diagnostic(img, mask)
        assert diag["upper"]["color"] == plain["upper"]
        assert diag["lower"]["color"] == plain["lower"]
        assert diag["upper"]["used_mask"] is True


class TestClassifyPersonAttributesWithMaskAndWb:
    def test_combines_mask_and_white_balance(self):
        """대안 C+WB: White Balance가 입력에 적용된 뒤 Mask 기반 분리가 이어져야 한다
        (순서가 바뀌면 안 됨 - WB는 crop 전체 통계를 쓰므로 마스킹 이후에 적용하면
        배경이 제외된 통계로 게인이 달라진다)."""
        img = solid_bgr((180, 90, 80), h=100, w=100)  # cool_cast 모사
        mask = np.ones((100, 100), dtype=bool)
        result = classify_person_attributes_with_mask_and_wb(img, mask)
        assert result["upper"] in COLOR_BUCKETS
        assert result["lower"] in COLOR_BUCKETS


class TestClassifyPersonAttributesWithMaskGraded:
    """EXP-033 대안 B를 production 경로(Mask 기반 영역 분리)에 그대로 연결한 변형.
    영역 분리/마스크 폴백 동작은 classify_person_attributes_with_mask(대안 C)와
    동일해야 하고, Hue 다수결 폴백만 graded로 바뀐 효과가 나타나야 한다."""

    def test_mask_excludes_background_bleed_same_as_non_graded(self):
        img = solid_bgr(RED, h=100, w=100)
        img[:, 40:60] = BLUE
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 40:60] = True
        result = classify_person_attributes_with_mask_graded(img, mask)
        assert result["upper"] == "blue"
        assert result["lower"] == "blue"

    def test_mask_below_min_fraction_falls_back_to_graded_rectangle_region(self):
        img = solid_bgr(GREEN, h=100, w=100)
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True
        result = classify_person_attributes_with_mask_graded(img, mask, min_mask_fraction=0.05)
        fallback = classify_person_attributes(img, method="b_graded")
        assert result == fallback

    def test_rescues_dark_garment_within_mask_that_flips_binary_mask_fallback(self):
        """TestRegionDominantColorGradedFallback의 핵심 회귀 시나리오를 Mask 기반
        production 경로(대안 C)에 그대로 연결해도 동일하게 재현되는지 확인한다 - 마스크
        안쪽 픽셀 집합도 결국 region_dominant_color(_from_pixels)를 거치므로 동일한
        폴백 버그/수정이 적용되어야 한다."""
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[0:17, :] = (8, 6, 20)
        img[17:34, :] = (20, 6, 8)
        img[34:52, :] = (6, 20, 8)
        img[52:70, :] = (8, 20, 6)
        img[70:100, :] = (120, 170, 210)
        mask = np.ones((100, 60), dtype=bool)

        assert classify_person_attributes_with_mask(img, mask)["lower"] == "orange"
        assert classify_person_attributes_with_mask_graded(img, mask)["lower"] == "black"

    def test_empty_crop_defaults_to_gray(self):
        result = classify_person_attributes_with_mask_graded(
            np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool)
        )
        assert result == {"upper": "gray", "lower": "gray"}


class TestClassifyPersonAttributesWithMaskAndWbGraded:
    def test_combines_mask_and_white_balance(self):
        img = solid_bgr((180, 90, 80), h=100, w=100)
        mask = np.ones((100, 100), dtype=bool)
        result = classify_person_attributes_with_mask_and_wb_graded(img, mask)
        assert result["upper"] in COLOR_BUCKETS
        assert result["lower"] in COLOR_BUCKETS


class TestClassifyPersonAttributesWithMaskRelaxed:
    """EXP-033 대안 A(val_min=35->10 단순 완화)를 production 경로에 연결한 변형."""

    def test_mask_excludes_background_bleed_same_as_non_relaxed(self):
        img = solid_bgr(RED, h=100, w=100)
        img[:, 40:60] = BLUE
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 40:60] = True
        result = classify_person_attributes_with_mask_relaxed(img, mask)
        assert result["upper"] == "blue"
        assert result["lower"] == "blue"

    def test_empty_crop_defaults_to_gray(self):
        result = classify_person_attributes_with_mask_relaxed(
            np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((0, 0), dtype=bool)
        )
        assert result == {"upper": "gray", "lower": "gray"}


class TestClassifyPersonAttributesWithMaskRelaxedSatMin:
    """EXP-034: sat_min을 val_min과 독립적으로 Sweep할 수 있도록 classify_person_
    attributes_with_mask_relaxed에 노출한 파라미터가 실제로 region_dominant_color까지
    전달되는지 고정한다(기본값 25는 EXP-033 당시 동작과 동일해야 한다)."""

    @staticmethod
    def _hsv_block(h: int, s: int, v: int, rows: int, cols: int) -> np.ndarray:
        block = np.zeros((rows, cols, 3), dtype=np.uint8)
        block[:, :] = (h, s, v)
        return cv2.cvtColor(block, cv2.COLOR_HSV2BGR)

    def _two_tone_upper_region(self) -> tuple[np.ndarray, np.ndarray]:
        """상의 영역(행 22:55)을 다수(60%, 저채도 s=18) + 소수(40%, 고채도 blue s=60)로
        채운다. sat_min=25(기본)는 다수 그룹을 걸러내 소수(blue)만 남기지만, sat_min=0은
        다수 그룹까지 포함시켜 다수결 자체를 뒤집는다 - EXP-034가 측정하려는 "sat_min을
        낮추면 그림자/저채도 Noise를 더 많이 포함시킨다"는 메커니즘을 그대로 재현한다.
        (h=100은 _is_skin_like 범위(h<=25) 밖이라 skin exclusion과는 독립적으로 둔다.)
        """
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        majority = self._hsv_block(h=120, s=18, v=100, rows=33, cols=36)
        minority = self._hsv_block(h=100, s=60, v=100, rows=33, cols=24)
        img[22:55, :36] = majority
        img[22:55, 36:] = minority
        mask = np.ones((100, 60), dtype=bool)
        return img, mask

    def test_default_sat_min_filters_out_majority_low_saturation_group(self):
        img, mask = self._two_tone_upper_region()
        result = classify_person_attributes_with_mask_relaxed(img, mask, val_min=10)
        assert result["upper"] == "blue"

    def test_sat_min_zero_lets_majority_low_saturation_group_dominate(self):
        img, mask = self._two_tone_upper_region()
        result = classify_person_attributes_with_mask_relaxed(img, mask, val_min=10, sat_min=0)
        assert result["upper"] != "blue"
        assert result["upper"] == "gray"

    def test_sat_min_relaxation_is_not_monotonic_middle_value_can_be_worse_than_both_extremes(self):
        """EXP-034가 실측 Dataset에서 발견한 비단조(non-monotonic) 패턴을 결정론적
        합성 픽셀로 고정한다. 진짜 검정(S=5,V=20, 50%) + 중간 채도 오염(S=15,V=100,
        H=100, 30%) + 고채도 소수(S=80,V=100,H=0, 20%)를 섞으면:
        - sat_min=25: 중간 오염이 필터에 걸려(S=15<25) 제외되고 고채도 소수(20%)만
          남아 min_filtered_fraction(30%) 미달로 폴백 발동 -> 전체 픽셀 중 진짜 검정이
          최다수(50%) -> "black"(정답).
        - sat_min=15/10/5: 중간 오염까지 필터를 통과해(30%+20%=50%>=30%) 폴백이
          발동하지 않고, 중간 오염이 폴백 없이 다수결에서 진짜 검정을 이김 -> "gray"
          (오답) - 딱 그 중간 구간만 걸리는 "함정 지대".
        - sat_min=0: 필터를 완전히 버려 모든 픽셀(진짜 검정 50% 포함)이 다수결에
          들어가고, 진짜 검정이 다시 최다수가 되어 "black"(정답)으로 복귀한다.
        이 U자형 패턴이 EXP-034가 "sat_min을 낮출수록 단조 개선될 것"이라는 가설을
        반증한 실제 메커니즘이다(Hypothesis 2, experiment.md Analysis 참고).
        """
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        true_black = self._hsv_block(h=0, s=5, v=20, rows=33, cols=30)
        mid_contaminant = self._hsv_block(h=100, s=15, v=100, rows=33, cols=18)
        high_sat_minority = self._hsv_block(h=0, s=80, v=100, rows=33, cols=12)
        img[22:55, :30] = true_black
        img[22:55, 30:48] = mid_contaminant
        img[22:55, 48:] = high_sat_minority
        mask = np.ones((100, 60), dtype=bool)

        at_sat_min = lambda sat_min: classify_person_attributes_with_mask_relaxed(
            img, mask, val_min=0, sat_min=sat_min
        )["upper"]

        assert at_sat_min(25) == "black"
        assert at_sat_min(15) == "gray"
        assert at_sat_min(10) == "gray"
        assert at_sat_min(0) == "black"

    def test_sat_min_forwards_through_wb_variant_too(self):
        """classify_person_attributes_with_mask_and_wb_relaxed(**kwargs)가 sat_min을
        실제로 전달하는지, White Balance 적용 후 동일 입력에 대한 non-WB 버전과
        동치임을 비교해 고정한다(WB 자체가 채도를 바꾸므로 색상 산출값이 아니라
        '값이 전달된다'는 사실 자체를 검증 대상으로 삼는다)."""
        img, mask = self._two_tone_upper_region()
        wb_img = white_balance_gray_world(img)
        expected = classify_person_attributes_with_mask_relaxed(wb_img, mask, val_min=10, sat_min=0)
        actual = classify_person_attributes_with_mask_and_wb_relaxed(img, mask, val_min=10, sat_min=0)
        assert actual == expected


class TestClassifyPersonAttributesWithMaskAndWbRelaxed:
    def test_combines_mask_and_white_balance(self):
        img = solid_bgr((180, 90, 80), h=100, w=100)
        mask = np.ones((100, 100), dtype=bool)
        result = classify_person_attributes_with_mask_and_wb_relaxed(img, mask)
        assert result["upper"] in COLOR_BUCKETS
        assert result["lower"] in COLOR_BUCKETS


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

    def test_method_b_relaxed_separates_upper_lower_like_method_b(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:55, :] = WHITE
        img[55:, :] = BLUE
        result = classify_person_attributes(img, method="b_relaxed")
        assert result["upper"] == "white"
        assert result["lower"] == "blue"

    def test_method_b_graded_separates_upper_lower_like_method_b(self):
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:55, :] = WHITE
        img[55:, :] = BLUE
        result = classify_person_attributes(img, method="b_graded")
        assert result["upper"] == "white"
        assert result["lower"] == "blue"

    def test_method_b_graded_rescues_dark_garment_that_flips_method_b(self):
        """EXP-033 핵심 회귀 시나리오를 classify_person_attributes(method=...) 공개
        API 수준에서도 재현한다 - method b의 lower 영역(head_skip 22%~foot_skip
        92% 행, side_margin 12% 제외 열)에 맞춰 fixture를 배치했다(아래 좌표는 실제
        split_upper_lower 경계에서 역산해 맞춘 값)."""
        img = np.zeros((100, 60, 3), dtype=np.uint8)
        img[:92, :] = WHITE  # upper 영역(22~55%)은 흰색 그대로
        img[55:62, :] = (8, 6, 20)  # lower 영역 시작(55%)부터 어두운 옷 70%, 4개 Hue로 분산
        img[62:69, :] = (20, 6, 8)
        img[69:76, :] = (6, 20, 8)
        img[76:81, :] = (8, 20, 6)
        img[81:92, :] = (120, 170, 210)  # 피부색 30%(foot_skip 92% 직전까지)

        assert classify_person_attributes(img, method="b")["lower"] == "orange"
        assert classify_person_attributes(img, method="b_graded")["lower"] == "black"


class TestEstimateGammaFromReference:
    def test_bright_reference_gives_gamma_greater_than_one(self):
        """평균 밝기가 target보다 높으면(과다 노출) 감마가 1보다 커야 한다(어둡게 보정)."""
        bright = solid_bgr((220, 220, 220))
        gamma = estimate_gamma_from_reference(bright, target_mean=128.0)
        assert gamma > 1.0

    def test_dark_reference_gives_gamma_less_than_one(self):
        """평균 밝기가 target보다 낮으면(저노출) 감마가 1보다 작아야 한다(밝게 보정)."""
        dark = solid_bgr((30, 30, 30))
        gamma = estimate_gamma_from_reference(dark, target_mean=128.0)
        assert gamma < 1.0

    def test_mid_gray_reference_gives_gamma_near_one(self):
        mid = solid_bgr((128, 128, 128))
        gamma = estimate_gamma_from_reference(mid, target_mean=128.0)
        assert abs(gamma - 1.0) < 0.05

    def test_near_pure_black_reference_clips_instead_of_exploding(self):
        """mean이 0에 가까우면 log(mean/255)가 0에 가까워져 감마가 무한히 커질 수 있다
        (white_balance_gray_world의 gain_min/gain_max와 동일한 안전장치)."""
        near_black = solid_bgr((1, 1, 1))
        gamma = estimate_gamma_from_reference(near_black, target_mean=128.0, gamma_max=2.5)
        assert gamma == 1.0  # mean<=1.0 가드에 걸려 보정 없음(원본 유지)

    def test_empty_reference_returns_neutral_gamma(self):
        assert estimate_gamma_from_reference(np.zeros((0, 0, 3), dtype=np.uint8)) == 1.0


class TestApplyGamma:
    def test_gamma_one_is_identity(self):
        img = solid_bgr((100, 150, 200))
        result = apply_gamma(img, 1.0)
        assert np.array_equal(result, img)

    def test_gamma_greater_than_one_darkens(self):
        img = solid_bgr((200, 200, 200))
        result = apply_gamma(img, 2.0)
        assert result.mean() < img.mean()

    def test_gamma_less_than_one_brightens(self):
        img = solid_bgr((50, 50, 50))
        result = apply_gamma(img, 0.5)
        assert result.mean() > img.mean()

    def test_empty_image_passthrough(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert apply_gamma(empty, 1.5).size == 0


class TestAdaptiveGammaCorrect:
    def test_overexposed_crop_is_darkened_toward_target(self):
        overexposed = solid_bgr((230, 230, 230))
        corrected = adaptive_gamma_correct(overexposed, target_mean=128.0)
        assert corrected.mean() < overexposed.mean()

    def test_crop_self_brightness_confounds_true_black_clothing(self):
        """EXP-030에서 기각된 핵심 이유를 고정하는 Regression Test: 실제로 어두운 옷
        (낮은 V)을 입은 crop도 "노출 보정 대상"으로 취급돼 밝게 끌어올려진다 - crop
        자신의 평균 밝기만으로는 "조명이 과다/저露출"과 "옷이 원래 검정색"을 구분할
        수 없다는 confound를 직접 검증한다."""
        true_black_clothing = solid_bgr((15, 15, 15))  # 정상 조명, 진짜 검정 옷
        corrected = adaptive_gamma_correct(true_black_clothing, target_mean=128.0)
        # 진짜 검정 옷인데도 평균 밝기가 target_mean 근처로 끌어올려져 더 이상 "검정"으로
        # 분류되기 어려운 수준까지 밝아진다(VAL_BLACK=45보다 훨씬 커짐).
        assert corrected.mean() > 45


class TestAdaptiveGammaCorrectFromBackground:
    def test_gamma_comes_from_background_not_crop(self):
        """crop 자신은 어둡지만(진짜 검정 옷) 배경이 과다 노출이면, 배경 기준 감마가
        crop에 적용돼 crop이 밝아진다 - 노출 추정을 crop 자신이 아니라 배경에서 가져오는
        설계(EXP-030 대안 B)가 실제로 crop이 아닌 reference를 쓰는지 검증한다."""
        dark_crop = solid_bgr((15, 15, 15))
        overexposed_background = solid_bgr((230, 230, 230))
        corrected = adaptive_gamma_correct_from_background(dark_crop, overexposed_background)
        assert corrected.mean() < dark_crop.mean()  # 배경이 과다노출 -> 어둡게 보정

    def test_normally_exposed_background_leaves_crop_unchanged(self):
        dark_crop = solid_bgr((15, 15, 15))
        normal_background = solid_bgr((128, 128, 128))
        corrected = adaptive_gamma_correct_from_background(dark_crop, normal_background)
        assert np.array_equal(corrected, dark_crop)


class TestGammaBackgroundThenWhiteBalance:
    """EXP-038: Re-ID FC-013(b) warm_cast/cool_cast 검증을 위한 Gamma-Background+WB
    조합(gamma 먼저, WB 나중 - EXP-030 b_wb_gamma와 동일한 순서)."""

    def test_equals_manual_composition(self):
        """순서가 명세대로(gamma 먼저 -> WB 나중)인지, 수동 합성과 동일한지 확인한다."""
        crop = np.random.default_rng(0).integers(0, 255, size=(20, 20, 3), dtype=np.uint8)
        background = solid_bgr((60, 90, 150))  # cool_cast 유사 배경(B 채널이 큼)
        expected = white_balance_gray_world(adaptive_gamma_correct_from_background(crop, background))
        actual = gamma_background_then_white_balance(crop, background)
        assert np.array_equal(actual, expected)

    def test_overexposed_background_still_darkens_crop_after_wb(self):
        """Gamma 보정 방향(adaptive_gamma_correct_from_background의 기존 검증된 동작 -
        배경이 과다노출이면 crop을 더 어둡게 보정)이 WB를 거친 뒤에도 뒤집히지 않는지
        확인한다(두 보정이 서로를 상쇄해 방향 자체가 반전되면 안 된다는 최소 Sanity
        Check). 완전한 0으로 Clip되는 극단값은 WB가 항등 연산이 되어 방향성 검증에
        쓸모가 없으므로(0*gain=0), 중간값을 사용한다."""
        crop = solid_bgr((60, 50, 55))
        overexposed_background = solid_bgr((190, 195, 185))
        gamma_then_wb = gamma_background_then_white_balance(crop, overexposed_background)
        assert gamma_then_wb.mean() < crop.mean()  # 어둡게 보정되는 방향 유지

    def test_cool_cast_background_reduces_blue_dominance_after_wb(self):
        """Cast가 있는 배경(B 채널이 큼, cool_cast 유사)에서도 WB 단계가 crop의 채널 간
        상대적 편향을 완화하는지 확인한다(Gamma 단독으로는 채널 간 비율이 보존되므로
        Cast가 그대로 남아야 하고, WB가 추가된 뒤에만 줄어들어야 한다)."""
        crop = solid_bgr((180, 100, 100))  # B가 상대적으로 큰 crop(cast 유사)
        cool_background = solid_bgr((160, 100, 90))
        gamma_only = adaptive_gamma_correct_from_background(crop, cool_background)
        gamma_then_wb = gamma_background_then_white_balance(crop, cool_background)
        b_minus_r_gamma_only = int(gamma_only[0, 0, 0]) - int(gamma_only[0, 0, 2])
        b_minus_r_gamma_then_wb = int(gamma_then_wb[0, 0, 0]) - int(gamma_then_wb[0, 0, 2])
        assert b_minus_r_gamma_then_wb < b_minus_r_gamma_only


class TestWhiteBalanceGrayWorldMasked:
    """EXP-039: FC-014(Re-ID Occlusion Top-1 저하) 검증 - Gray World 평균 계산을 person
    Segmentation Mask 픽셀에만 한정한다."""

    def test_equals_unmasked_when_mask_is_all_true(self):
        crop = solid_bgr((180, 100, 100))
        mask = np.ones(crop.shape[:2], dtype=bool)
        assert np.array_equal(white_balance_gray_world_masked(crop, mask), white_balance_gray_world(crop))

    def test_falls_back_to_unmasked_when_mask_coverage_below_min_fraction(self):
        """Segmentation 매칭 실패 등으로 mask가 거의 비어 있으면(기본 min_mask_fraction=0.05
        미달) 평균을 신뢰할 수 없다고 보고 전체 픽셀 기준(white_balance_gray_world)으로
        폴백한다 - classify_person_attributes_with_mask_relaxed()의 동일한 폴백 설계."""
        crop = solid_bgr((180, 100, 100))
        mask = np.zeros(crop.shape[:2], dtype=bool)
        mask[0, 0] = True  # 1픽셀만 True -> 0.05 미달
        assert np.array_equal(white_balance_gray_world_masked(crop, mask), white_balance_gray_world(crop))

    def test_empty_crop_returns_crop_unchanged(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        mask = np.zeros((0, 0), dtype=bool)
        assert white_balance_gray_world_masked(empty, mask).size == 0

    def test_occlusion_patch_outside_mask_does_not_influence_gain(self):
        """FC-014 재현 - crop 절반이 옷 색상(Cast가 있는 색), 절반이 중립 회색 Occlusion
        패치다. mask가 옷 영역만 True로 표시하면, 패치가 평균에 섞이지 않아야 하므로
        masked 결과는 "crop 전체가 옷 색상뿐이었을 때"의 WB 결과(패치 없는 crop)와
        동일해야 한다 - 반면 Masking 없이(white_balance_gray_world) 전체 평균을 쓰면
        패치가 평균을 중립 쪽으로 당겨 다른 게인이 나온다."""
        h, w = 40, 40
        clothing_bgr = (180, 100, 100)  # B가 상대적으로 큰 cast 유사 색
        occlusion_bgr = (120, 120, 120)  # 중립 회색 Occlusion 패치(EXP-036~039와 동일 값)
        crop_with_patch = np.zeros((h, w, 3), dtype=np.uint8)
        crop_with_patch[:, :] = clothing_bgr
        crop_with_patch[: h // 2, :] = occlusion_bgr  # 위쪽 절반을 패치로 덮음

        mask_clothing_only = np.zeros((h, w), dtype=bool)
        mask_clothing_only[h // 2 :, :] = True  # 패치가 없는 아래쪽 절반만 True

        masked_result = white_balance_gray_world_masked(crop_with_patch, mask_clothing_only)
        unmasked_result = white_balance_gray_world(crop_with_patch)

        clothing_only_crop = np.zeros((h, w, 3), dtype=np.uint8)
        clothing_only_crop[:, :] = clothing_bgr
        clothing_only_reference = white_balance_gray_world(clothing_only_crop)

        # masked 결과의 옷 영역 픽셀은 "패치가 전혀 없었을 때"의 WB 결과와 동일해야 한다.
        assert np.array_equal(masked_result[h // 2 :, :], clothing_only_reference[h // 2 :, :])
        # 반면 Masking 없이 전체 평균을 쓰면 패치가 평균을 끌어당겨 다른 게인이 나온다.
        assert not np.array_equal(unmasked_result[h // 2 :, :], clothing_only_reference[h // 2 :, :])


class TestGammaBackgroundThenWhiteBalanceMasked:
    """EXP-039: gamma_background_then_white_balance()와 동일한 순서(감마 먼저 -> WB
    나중)이지만 WB 단계만 Mask 기반으로 교체한 조합."""

    def test_equals_manual_composition(self):
        crop = np.random.default_rng(1).integers(0, 255, size=(20, 20, 3), dtype=np.uint8)
        background = solid_bgr((60, 90, 150))
        mask = np.ones((20, 20), dtype=bool)
        mask[:10, :] = False
        expected = white_balance_gray_world_masked(adaptive_gamma_correct_from_background(crop, background), mask)
        actual = gamma_background_then_white_balance_masked(crop, background, mask)
        assert np.array_equal(actual, expected)

    def test_mask_excluding_occlusion_differs_from_unmasked_version(self):
        """같은 Gamma-Background 입력이라도 WB 단계에서 Occlusion 패치를 mask로 제외하면
        gamma_background_then_white_balance(Masking 없음)과 다른 결과가 나와야 한다 -
        이것이 FC-014를 완화하려는 핵심 메커니즘이다."""
        h, w = 40, 40
        clothing_bgr = (180, 100, 100)
        occlusion_bgr = (120, 120, 120)
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        crop[:, :] = clothing_bgr
        crop[: h // 2, :] = occlusion_bgr
        background = solid_bgr((128, 128, 128))  # 노출 변화 없음(Gamma 항등) -> WB 차이만 측정
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2 :, :] = True

        masked = gamma_background_then_white_balance_masked(crop, background, mask)
        unmasked = gamma_background_then_white_balance(crop, background)
        assert not np.array_equal(masked[h // 2 :, :], unmasked[h // 2 :, :])


class TestFrameMeanBrightness:
    """EXP-031: Frame 전체(crop 아님) 밝기 측정 - adaptive_gamma_correct의 confound
    (crop 밝기 = 옷 색상 + 조명 노출 혼합)가 생기지 않는지 확인한다."""

    def test_dark_frame_has_low_mean_brightness(self):
        dark_frame = solid_bgr((20, 20, 20), h=200, w=300)
        assert frame_mean_brightness(dark_frame) < 45

    def test_bright_frame_has_high_mean_brightness(self):
        bright_frame = solid_bgr((230, 230, 230), h=200, w=300)
        assert frame_mean_brightness(bright_frame) > 200

    def test_not_confounded_by_a_single_dark_person_in_a_bright_scene(self):
        """crop 자체 밝기(adaptive_gamma_correct가 기각된 원인)와 달리, Frame 전체
        밝기는 한 사람의 옷 색상(일부 영역)만으로 좌우되지 않는다 - 밝은 배경 안의
        작은 검정 옷 영역은 Frame 전체 평균을 크게 끌어내리지 않는다."""
        frame = solid_bgr((230, 230, 230), h=200, w=300)
        frame[80:120, 100:140] = (15, 15, 15)  # 작은 검정 옷 person 영역
        assert frame_mean_brightness(frame) > 200

    def test_empty_frame_returns_zero(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert frame_mean_brightness(empty) == 0.0


class TestClassifyExposureLevel:
    def test_below_low_threshold_is_low_light(self):
        assert classify_exposure_level(40.0, low_threshold=80.0, high_threshold=130.0) == "low_light"

    def test_above_high_threshold_is_strong_light(self):
        assert classify_exposure_level(200.0, low_threshold=80.0, high_threshold=130.0) == "strong_light"

    def test_within_band_is_normal(self):
        assert classify_exposure_level(100.0, low_threshold=80.0, high_threshold=130.0) == "normal"

    def test_boundary_values_are_inclusive_of_normal(self):
        assert classify_exposure_level(80.0, low_threshold=80.0, high_threshold=130.0) == "normal"
        assert classify_exposure_level(130.0, low_threshold=80.0, high_threshold=130.0) == "normal"


class TestFrameChannelCastDeviation:
    """EXP-031 Alt C(기각): warm_cast/cool_cast를 겨냥해 추가했으나 normal 조건
    자체의 편차 분포와 거의 겹쳐 분리되지 않음(실측은 EXP-031 summary.json 참고).
    여기서는 함수 자체의 계산이 올바른지(메커니즘)만 검증한다."""

    def test_neutral_gray_frame_has_zero_deviation(self):
        neutral = solid_bgr((128, 128, 128), h=100, w=100)
        assert frame_channel_cast_deviation(neutral) == pytest.approx(0.0, abs=1e-6)

    def test_strong_blue_cast_has_high_deviation(self):
        blue_cast = solid_bgr((230, 80, 80), h=100, w=100)  # B channel much higher
        assert frame_channel_cast_deviation(blue_cast) > 0.3

    def test_empty_frame_returns_zero(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert frame_channel_cast_deviation(empty) == 0.0
