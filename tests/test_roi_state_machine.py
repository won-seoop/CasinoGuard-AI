import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from events.roi_state_machine import ROIStateMachine, bottom_center, point_in_polygon  # noqa: E402

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_bottom_center_uses_bottom_edge_midpoint():
    assert bottom_center((0, 0, 10, 20)) == (5, 20)


def test_point_in_polygon_inside():
    assert point_in_polygon((50, 50), SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon((200, 200), SQUARE) is False


def test_no_event_while_staying_outside():
    sm = ROIStateMachine(polygon=SQUARE)
    ev1 = sm.update(track_id=1, bbox=(200, 200, 210, 220), frame_idx=0)
    ev2 = sm.update(track_id=1, bbox=(210, 210, 220, 230), frame_idx=1)
    assert ev1 is None
    assert ev2 is None


def test_enter_event_fires_once_on_transition():
    sm = ROIStateMachine(polygon=SQUARE)
    # 밖에서 시작
    sm.update(track_id=1, bbox=(200, 200, 210, 220), frame_idx=0)
    # 안으로 들어옴 -> ENTER 이벤트 1회
    ev = sm.update(track_id=1, bbox=(40, 40, 60, 60), frame_idx=1)
    assert ev is not None
    assert ev.event_type == "ENTER"
    # 계속 안에 머무름 -> 이벤트 없음 (중복 방지)
    ev2 = sm.update(track_id=1, bbox=(45, 45, 65, 65), frame_idx=2)
    ev3 = sm.update(track_id=1, bbox=(42, 42, 62, 62), frame_idx=3)
    assert ev2 is None
    assert ev3 is None


def test_exit_event_fires_once_on_leaving():
    sm = ROIStateMachine(polygon=SQUARE)
    sm.update(track_id=1, bbox=(40, 40, 60, 60), frame_idx=0)  # INSIDE로 시작 상태 세팅
    ev = sm.update(track_id=1, bbox=(200, 200, 210, 220), frame_idx=1)
    assert ev is not None
    assert ev.event_type == "EXIT"


def test_tracks_are_independent():
    sm = ROIStateMachine(polygon=SQUARE)
    ev_a = sm.update(track_id=1, bbox=(40, 40, 60, 60), frame_idx=0)
    ev_b = sm.update(track_id=2, bbox=(200, 200, 210, 220), frame_idx=0)
    assert ev_a.event_type == "ENTER"
    assert ev_b is None  # track 2는 계속 OUTSIDE


def test_forget_track_resets_state_to_outside():
    sm = ROIStateMachine(polygon=SQUARE)
    sm.update(track_id=1, bbox=(40, 40, 60, 60), frame_idx=0)  # INSIDE
    sm.forget_track(1, frame_idx=5)
    assert sm.active_track_count() == 0
    # 다시 나타나면 OUTSIDE부터 시작 -> 안으로 들어오면 다시 ENTER
    ev = sm.update(track_id=1, bbox=(40, 40, 60, 60), frame_idx=10)
    assert ev.event_type == "ENTER"


def test_forget_track_emits_exit_when_was_inside():
    """FC-004: INSIDE 상태에서 Track이 사라지면 EXIT 이벤트를 강제로 내야 한다."""
    sm = ROIStateMachine(polygon=SQUARE)
    sm.update(track_id=1, bbox=(40, 40, 60, 60), frame_idx=0)  # INSIDE
    ev = sm.forget_track(1, frame_idx=7)
    assert ev is not None
    assert ev.event_type == "EXIT"
    assert ev.frame_idx == 7


def test_forget_track_emits_nothing_when_was_outside():
    sm = ROIStateMachine(polygon=SQUARE)
    sm.update(track_id=1, bbox=(200, 200, 210, 220), frame_idx=0)  # OUTSIDE
    ev = sm.forget_track(1, frame_idx=7)
    assert ev is None


def test_boundary_flicker_does_not_duplicate_when_state_unchanged():
    """경계 근처에서 같은 상태(INSIDE)를 계속 유지하면 이벤트가 중복 발생하지 않아야 한다."""
    sm = ROIStateMachine(polygon=SQUARE)
    events = []
    # 경계(x=99~101) 근처에서 흔들리지만 모두 INSIDE로 판정되는 경우
    for x in [95, 98, 99, 97, 96]:
        ev = sm.update(track_id=1, bbox=(x, 50, x + 2, 52), frame_idx=x)
        if ev:
            events.append(ev)
    assert len(events) == 1  # 최초 ENTER 1회만
    assert events[0].event_type == "ENTER"
