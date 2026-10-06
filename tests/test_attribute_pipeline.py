import numpy as np

from src.attributes.pipeline import classify_track_attributes_once


def solid_bgr(color_bgr: tuple[int, int, int], h: int = 100, w: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


BLACK = (10, 10, 10)


def test_classify_track_attributes_once_calls_mask_provider_exactly_once():
    crop = solid_bgr(BLACK)
    calls = []

    def fake_mask_provider(crop_bgr: np.ndarray) -> np.ndarray:
        calls.append(crop_bgr.shape)
        return np.ones(crop_bgr.shape[:2], dtype=bool)

    result = classify_track_attributes_once(crop, fake_mask_provider)
    assert len(calls) == 1  # Track당 1회만 호출되어야 한다 (EXP-029 설계 의도)
    assert result["upper"] == "black"
    assert result["lower"] == "black"


def test_classify_track_attributes_once_skips_mask_provider_for_empty_crop():
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    calls = []

    def fake_mask_provider(crop_bgr: np.ndarray) -> np.ndarray:
        calls.append(crop_bgr.shape)
        return np.ones(crop_bgr.shape[:2], dtype=bool)

    result = classify_track_attributes_once(empty_crop, fake_mask_provider)
    assert len(calls) == 0  # 빈 crop으로 Segmentation 모델(실제 추론)을 호출하지 않는다
    assert result == {"upper": "gray", "lower": "gray"}


def test_classify_track_attributes_once_falls_back_without_mask():
    crop = solid_bgr(BLACK)

    def empty_mask_provider(crop_bgr: np.ndarray) -> np.ndarray:
        return np.zeros(crop_bgr.shape[:2], dtype=bool)

    result = classify_track_attributes_once(crop, empty_mask_provider)
    # mask가 전부 False여도(세그멘테이션 실패) classify_person_attributes_with_mask_and_wb의
    # min_mask_fraction fallback이 동작해 여전히 유효한 색상을 반환해야 한다.
    assert result["upper"] == "black"
    assert result["lower"] == "black"
