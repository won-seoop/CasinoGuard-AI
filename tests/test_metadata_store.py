import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
