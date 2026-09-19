import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from events.loitering import LoiteringDetector  # noqa: E402

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]
INSIDE_POINT = (50, 50)
OUTSIDE_POINT = (200, 200)


def test_no_event_before_threshold():
    det = LoiteringDetector(polygon=SQUARE, threshold_frames=10)
    for f in range(9):
        ev = det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
        assert ev is None


def test_event_fires_exactly_at_threshold():
    det = LoiteringDetector(polygon=SQUARE, threshold_frames=10)
    # enter_frame은 frame_idx=0에서 설정되므로, dwell=frame_idx-enter_frame이
    # threshold(10)에 도달하는 시점은 frame_idx=10이다.
    for f in range(11):
        ev = det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
    assert ev is not None
    assert ev.dwell_frames == 10


def test_event_does_not_repeat_while_still_inside():
    det = LoiteringDetector(polygon=SQUARE, threshold_frames=5)
    events = []
    for f in range(20):
        ev = det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
        if ev:
            events.append(ev)
    assert len(events) == 1  # 계속 안에 있어도 알림은 한 번만


def test_leaving_roi_resets_dwell_timer():
    det = LoiteringDetector(polygon=SQUARE, threshold_frames=5)
    for f in range(4):
        det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
    det.update(track_id=1, point=OUTSIDE_POINT, frame_idx=4)  # 잠깐 나감 -> 리셋
    ev = None
    for f in range(5, 9):
        ev = det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
    assert ev is None  # 리셋됐으므로 threshold(5)에 아직 도달 못함


def test_reentry_after_leaving_can_fire_again():
    det = LoiteringDetector(polygon=SQUARE, threshold_frames=3)
    for f in range(4):
        ev1 = det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
    assert ev1 is not None
    det.update(track_id=1, point=OUTSIDE_POINT, frame_idx=4)
    fired_again = [det.update(track_id=1, point=INSIDE_POINT, frame_idx=f) for f in range(5, 10)]
    assert any(e is not None for e in fired_again)  # 새로 들어와서 다시 threshold 도달 -> 재알림


def test_tracks_are_independent():
    det = LoiteringDetector(polygon=SQUARE, threshold_frames=5)
    for f in range(6):
        ev_a = det.update(track_id=1, point=INSIDE_POINT, frame_idx=f)
    ev_b = det.update(track_id=2, point=INSIDE_POINT, frame_idx=0)
    assert ev_a is not None
    assert ev_b is None
