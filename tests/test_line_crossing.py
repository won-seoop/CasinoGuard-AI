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
