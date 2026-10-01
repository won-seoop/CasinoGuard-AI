"""EXP-024: VMS Search API에 event_clip_path 응답 필드 + Event Clip 재생 엔드포인트 추가.

tests/test_api.py의 기존 테스트는 results/metadata_pipeline/metadata.db(run_full_pipeline.py
산출물)가 있어야만 돌아가도록 module-level skipif가 걸려 있다. event_clip_path는
run_full_pipeline.py가 아직 만들지 않으므로(원본 영상 접근 가능한 세션으로 이월, EXP-023 Next
Action), 이 모듈은 자체 DB를 만들어 API 로직만(모델 추론과 무관) 검증한다. (EXP-024에서
run_exp023_pipeline_integration.py로 재생성한 실제 EXP-023 데이터로 수동 end-to-end 검증도
거쳤다 — experiments/EXP-024/experiment.md 참고.)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLIP_RELATIVE_PATH = "event_clips/event_clip_1.mp4"


def make_client_with_clip_fixture(tmp_path, monkeypatch, clip_bytes: bytes = b"fake-mp4-bytes"):
    from fastapi.testclient import TestClient

    import api.main as api_main
    from metadata.store import MetadataStore

    monkeypatch.setattr(api_main, "ROOT", tmp_path)

    clip_dir = tmp_path / "event_clips"
    clip_dir.mkdir()
    (clip_dir / "event_clip_1.mp4").write_bytes(clip_bytes)

    db_path = tmp_path / "metadata.db"
    store = MetadataStore(db_path)
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="restricted_zone")
    enter_id = store.add_event(1, "INTRUSION_ENTER", frame_idx=639, detail={"zone": "restricted_zone"})
    store.update_event_detail(enter_id, {"event_clip_path": CLIP_RELATIVE_PATH})
    exit_id = store.add_event(1, "INTRUSION_EXIT", frame_idx=687, detail={"zone": "restricted_zone"})
    store.close()

    monkeypatch.setattr(api_main, "DB_PATH", db_path)
    return TestClient(api_main.app), enter_id, exit_id


def test_list_events_exposes_event_clip_path_top_level(tmp_path, monkeypatch):
    client, enter_id, exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch)

    body = client.get("/events").json()
    assert body["count"] == 2
    enter_row = next(e for e in body["results"] if e["event_id"] == enter_id)
    exit_row = next(e for e in body["results"] if e["event_id"] == exit_id)
    assert enter_row["event_clip_path"] == CLIP_RELATIVE_PATH
    assert exit_row["event_clip_path"] is None


def test_list_events_has_clip_filter(tmp_path, monkeypatch):
    client, enter_id, exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch)

    with_clip = client.get("/events", params={"has_clip": True}).json()
    without_clip = client.get("/events", params={"has_clip": False}).json()
    assert [e["event_id"] for e in with_clip["results"]] == [enter_id]
    assert [e["event_id"] for e in without_clip["results"]] == [exit_id]


def test_get_track_embeds_event_clip_path(tmp_path, monkeypatch):
    client, enter_id, _exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch)

    body = client.get("/tracks/1").json()
    enter_row = next(e for e in body["events"] if e["event_id"] == enter_id)
    assert enter_row["event_clip_path"] == CLIP_RELATIVE_PATH


def test_get_event_clip_serves_file(tmp_path, monkeypatch):
    client, enter_id, _exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch, clip_bytes=b"\x00\x01mp4-payload")

    r = client.get(f"/events/{enter_id}/clip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.content == b"\x00\x01mp4-payload"


def test_get_event_clip_404_when_event_has_no_clip(tmp_path, monkeypatch):
    client, _enter_id, exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch)

    r = client.get(f"/events/{exit_id}/clip")
    assert r.status_code == 404


def test_get_event_clip_404_when_event_id_unknown(tmp_path, monkeypatch):
    client, _enter_id, _exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch)

    r = client.get("/events/999999/clip")
    assert r.status_code == 404


def test_get_event_clip_404_when_clip_file_missing_on_disk(tmp_path, monkeypatch):
    import api.main as api_main
    from fastapi.testclient import TestClient
    from metadata.store import MetadataStore

    monkeypatch.setattr(api_main, "ROOT", tmp_path)
    db_path = tmp_path / "metadata.db"
    store = MetadataStore(db_path)
    store.upsert_track(1, "person", 0, 0.9, (0, 0, 10, 10), (5, 10), zone="restricted_zone")
    enter_id = store.add_event(1, "INTRUSION_ENTER", frame_idx=639, detail={})
    store.update_event_detail(enter_id, {"event_clip_path": "event_clips/does_not_exist.mp4"})
    store.close()
    monkeypatch.setattr(api_main, "DB_PATH", db_path)

    client = TestClient(api_main.app)
    r = client.get(f"/events/{enter_id}/clip")
    assert r.status_code == 404


def test_dashboard_shows_clip_link_when_event_has_clip(tmp_path, monkeypatch):
    client, _enter_id, _exit_id = make_client_with_clip_fixture(tmp_path, monkeypatch)

    r = client.get("/")
    assert r.status_code == 200
    assert "clip</a>" in r.text
