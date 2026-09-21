import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from events.line_crossing import LineCrossingDetector, side_of_line  # noqa: E402

# 수평선: (0,50) -> (100,50). 위(y<50)와 아래(y>50)를 구분.
LINE_A, LINE_B = (0, 50), (100, 50)


def test_side_of_line_opposite_signs_on_each_side():
    s1 = side_of_line((50, 10), LINE_A, LINE_B)  # 위
    s2 = side_of_line((50, 90), LINE_A, LINE_B)  # 아래
    assert s1 != 0 and s2 != 0
    assert s1 != s2


def test_no_event_on_first_observation():
    det = LineCrossingDetector(LINE_A, LINE_B)
    ev = det.update(track_id=1, point=(50, 10), frame_idx=0)
    assert ev is None


def test_no_event_while_staying_on_same_side():
    det = LineCrossingDetector(LINE_A, LINE_B)
    det.update(track_id=1, point=(50, 10), frame_idx=0)
    ev = det.update(track_id=1, point=(60, 20), frame_idx=1)
    assert ev is None


def test_event_fires_once_on_crossing():
    det = LineCrossingDetector(LINE_A, LINE_B)
    det.update(track_id=1, point=(50, 10), frame_idx=0)  # 위에서 시작
    ev = det.update(track_id=1, point=(50, 90), frame_idx=1)  # 아래로 이동 -> Crossing
    assert ev is not None
    # 다시 같은 쪽(아래)에 머무르면 이벤트 없음
    ev2 = det.update(track_id=1, point=(55, 95), frame_idx=2)
    assert ev2 is None


def test_direction_is_opposite_for_opposite_crossings():
    det1 = LineCrossingDetector(LINE_A, LINE_B)
    det1.update(track_id=1, point=(50, 10), frame_idx=0)
    ev_down = det1.update(track_id=1, point=(50, 90), frame_idx=1)

    det2 = LineCrossingDetector(LINE_A, LINE_B)
    det2.update(track_id=2, point=(50, 90), frame_idx=0)
    ev_up = det2.update(track_id=2, point=(50, 10), frame_idx=1)

    assert ev_down.direction != ev_up.direction


def test_point_exactly_on_line_does_not_trigger_or_reset_state():
    det = LineCrossingDetector(LINE_A, LINE_B)
    det.update(track_id=1, point=(50, 10), frame_idx=0)  # 위
    ev_on_line = det.update(track_id=1, point=(50, 50), frame_idx=1)  # 선 위(경계)
    assert ev_on_line is None
    # 선 위를 거친 뒤 같은 쪽(위)로 복귀 -> 여전히 이벤트 없어야 함
    ev_back = det.update(track_id=1, point=(50, 5), frame_idx=2)
    assert ev_back is None


def test_tracks_are_independent():
    det = LineCrossingDetector(LINE_A, LINE_B)
    det.update(track_id=1, point=(50, 10), frame_idx=0)
    det.update(track_id=2, point=(50, 90), frame_idx=0)
    ev1 = det.update(track_id=1, point=(50, 90), frame_idx=1)
    ev2 = det.update(track_id=2, point=(50, 90), frame_idx=1)  # 그대로 머무름
    assert ev1 is not None
    assert ev2 is None


# --- FC-007: 선 근처 박스 흔들림으로 인한 왕복 중복 이벤트 (EXP-014, band_px Hysteresis) ---


def test_band_px_zero_reproduces_fc007_duplicate_events_on_jitter():
    """band_px=0(기존 동작)에서는 선(y=50) 바로 옆에서 +-2px로 흔들리는 것만으로도
    매번 Crossing 이벤트가 발생한다 — FC-007이 실제로 재현됨을 보이는 회귀 테스트."""
    det = LineCrossingDetector(LINE_A, LINE_B)  # band_px 기본값 0
    det.update(track_id=1, point=(50, 40), frame_idx=0)  # 위(확실히 y<50)
    jitter_ys = [48, 52, 49, 51, 48, 52]  # 선(y=50) 주변 +-2px 흔들림
    events = [det.update(track_id=1, point=(50, y), frame_idx=i + 1) for i, y in enumerate(jitter_ys)]
    fired = [e for e in events if e is not None]
    assert len(fired) >= 4  # 흔들릴 때마다 반복적으로 이벤트가 남 (중복 문제 재현)


def test_band_px_suppresses_jitter_within_band():
    """band_px=5로 두면 y=50 +-2px 흔들림(band 이내)은 이벤트를 만들지 않는다."""
    det = LineCrossingDetector(LINE_A, LINE_B, band_px=5)
    det.update(track_id=1, point=(50, 40), frame_idx=0)  # 위, band 밖에서 확정
    jitter_ys = [48, 52, 49, 51, 48, 52]
    events = [det.update(track_id=1, point=(50, y), frame_idx=i + 1) for i, y in enumerate(jitter_ys)]
    assert all(e is None for e in events)


def test_band_px_still_fires_on_genuine_crossing_beyond_band():
    """band_px가 있어도 band 밖으로 확실히 넘어가면 여전히 1회 이벤트가 발생해야 한다."""
    det = LineCrossingDetector(LINE_A, LINE_B, band_px=5)
    det.update(track_id=1, point=(50, 40), frame_idx=0)  # 위, band 밖(거리 10)
    ev = det.update(track_id=1, point=(50, 60), frame_idx=1)  # 아래, band 밖(거리 10) -> 진짜 통과
    assert ev is not None
    assert ev.direction == "A_TO_B"
    # 통과 후 같은 쪽(band 밖)에 머무르면 추가 이벤트 없음
    ev2 = det.update(track_id=1, point=(50, 65), frame_idx=2)
    assert ev2 is None


def test_band_px_default_is_zero_backward_compatible():
    det = LineCrossingDetector(LINE_A, LINE_B)
    assert det.band_px == 0.0


def test_signed_distance_matches_pixel_scale():
    from events.line_crossing import signed_distance

    # 선(y=50)에서 y=60인 점까지 수직 거리는 10px이어야 한다 (외적을 선분 길이로 정규화).
    d = signed_distance((50, 60), LINE_A, LINE_B)
    assert abs(abs(d) - 10.0) < 1e-6
