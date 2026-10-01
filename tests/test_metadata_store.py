import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from metadata.store import MetadataStore  # noqa: E402


def make_store():
    return MetadataStore(":memory:")


def test_upsert_track_creates_new_row():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="OUTSIDE")
    rows = store.query_tracks()
    assert len(rows) == 1
    assert rows[0]["track_id"] == 1
    assert rows[0]["first_seen"] == 0
    assert rows[0]["last_seen"] == 0


def test_upsert_track_updates_last_seen_and_appends_trajectory():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="OUTSIDE")
    store.upsert_track(1, "person", 5, 0.8, (1, 1, 11, 11), (6, 11), zone="INSIDE")
    rows = store.query_tracks()
    assert len(rows) == 1  # 같은 track_id는 새 row가 아니라 업데이트
    assert rows[0]["last_seen"] == 5
    assert rows[0]["current_zone"] == "INSIDE"


def test_add_event_and_query_by_type():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    store.add_event(1, "INTRUSION_ENTER", frame_idx=3, detail={"zone": "vault"})
    store.add_event(1, "LOITERING", frame_idx=100, detail={"dwell_sec": 12.5})
    intrusion_events = store.query_events(event_type="INTRUSION_ENTER")
    assert len(intrusion_events) == 1
    assert intrusion_events[0]["detail"]["zone"] == "vault"


def test_query_tracks_filters_by_min_dwell():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    store.upsert_track(2, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    store.set_dwell(1, 300)
    store.set_dwell(2, 10)
    rows = store.query_tracks(min_dwell_frames=100)
    assert [r["track_id"] for r in rows] == [1]


def test_update_bestshot_sets_path_and_score():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone=None)
    store.update_bestshot(1, "results/bestshot/track_1.jpg", 4.2)
    rows = store.query_tracks()
    assert rows[0]["bestshot_path"] == "results/bestshot/track_1.jpg"


def test_add_event_returns_event_id():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    event_id = store.add_event(1, "INTRUSION_ENTER", frame_idx=3, detail={"zone": "vault"})
    assert isinstance(event_id, int)
    events = store.query_events()
    assert events[0]["event_id"] == event_id


def test_update_event_detail_merges_without_overwriting_existing_keys():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    event_id = store.add_event(1, "INTRUSION_ENTER", frame_idx=3, detail={"zone": "vault"})
    store.update_event_detail(event_id, {"event_clip_path": "results/EXP-023/clip_0.mp4"})
    events = store.query_events()
    assert events[0]["detail"]["zone"] == "vault"
    assert events[0]["detail"]["event_clip_path"] == "results/EXP-023/clip_0.mp4"


def test_update_event_detail_raises_for_unknown_event_id():
    store = make_store()
    with pytest.raises(KeyError):
        store.update_event_detail(999, {"event_clip_path": "x.mp4"})


def test_query_events_exposes_event_clip_path_as_top_level_field():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    event_id = store.add_event(1, "INTRUSION_ENTER", frame_idx=3, detail={"zone": "vault"})
    store.update_event_detail(event_id, {"event_clip_path": "results/EXP-023/event_clips/event_clip_1.mp4"})
    events = store.query_events()
    assert events[0]["event_clip_path"] == "results/EXP-023/event_clips/event_clip_1.mp4"


def test_query_events_event_clip_path_is_none_when_not_set():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    store.add_event(1, "LOITERING", frame_idx=3, detail={"dwell_sec": 12.5})
    events = store.query_events()
    assert events[0]["event_clip_path"] is None


def test_query_events_has_clip_filter_true_returns_only_events_with_clip():
    store = make_store()
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="INSIDE")
    with_clip_id = store.add_event(1, "INTRUSION_ENTER", frame_idx=3, detail={})
    store.update_event_detail(with_clip_id, {"event_clip_path": "clip.mp4"})
    store.add_event(1, "LOITERING", frame_idx=10, detail={})

    with_clip = store.query_events(has_clip=True)
    without_clip = store.query_events(has_clip=False)

    assert [e["event_id"] for e in with_clip] == [with_clip_id]
    assert len(without_clip) == 1
    assert without_clip[0]["event_clip_path"] is None
