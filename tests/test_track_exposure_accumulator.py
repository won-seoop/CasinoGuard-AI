from src.attributes.confidence import TrackExposureAccumulator


def test_confidence_is_high_before_any_observation():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    assert acc.confidence(track_id=1) == "high"


def test_all_normal_frames_stay_high_confidence():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    for _ in range(10):
        acc.update(track_id=1, exposure_level="normal")
    assert acc.confidence(track_id=1) == "high"


def test_majority_abnormal_frames_flip_to_low_confidence():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    for _ in range(3):
        acc.update(track_id=1, exposure_level="normal")
    for _ in range(7):
        acc.update(track_id=1, exposure_level="strong_light")
    assert acc.confidence(track_id=1, min_abnormal_fraction=0.5) == "low"


def test_minority_abnormal_frames_stay_high_confidence():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    for _ in range(8):
        acc.update(track_id=1, exposure_level="normal")
    for _ in range(2):
        acc.update(track_id=1, exposure_level="low_light")
    assert acc.confidence(track_id=1, min_abnormal_fraction=0.5) == "high"


def test_tracks_are_independent():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    for _ in range(10):
        acc.update(track_id=1, exposure_level="strong_light")
    for _ in range(10):
        acc.update(track_id=2, exposure_level="normal")
    assert acc.confidence(track_id=1) == "low"
    assert acc.confidence(track_id=2) == "high"


def test_forget_track_clears_accumulated_state():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    for _ in range(10):
        acc.update(track_id=1, exposure_level="strong_light")
    acc.forget_track(1)
    assert acc.confidence(track_id=1) == "high"  # 관측 기록이 없으므로 기본값(high)으로 리셋
    assert 1 not in acc.total_frames
    assert 1 not in acc.abnormal_frames


def test_forget_unknown_track_does_not_raise():
    acc = TrackExposureAccumulator(low_threshold=80.0, high_threshold=130.0)
    acc.forget_track(999)  # no-op, 예외 없이 통과해야 함
