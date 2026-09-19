import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DB_PATH = ROOT / "results/metadata_pipeline/metadata.db"

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="metadata.db가 아직 생성되지 않음 (run_full_pipeline.py 먼저 실행 필요)")


def get_client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def test_health_ok():
    client = get_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["db_exists"] is True


def test_list_tracks_returns_results():
    client = get_client()
    r = client.get("/tracks")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0


def test_list_tracks_min_dwell_filter_returns_subset():
    client = get_client()
    all_tracks = client.get("/tracks").json()["count"]
    filtered = client.get("/tracks", params={"min_dwell_sec": 100}).json()["count"]
    assert filtered <= all_tracks


def test_get_single_track_includes_events():
    client = get_client()
    any_track_id = client.get("/tracks").json()["results"][0]["track_id"]
    r = client.get(f"/tracks/{any_track_id}")
    body = r.json()
    assert body["track"]["track_id"] == any_track_id
    assert "events" in body


def test_get_missing_track_returns_error():
    client = get_client()
    r = client.get("/tracks/999999")
    assert r.json() == {"error": "not found"}


def test_list_events_filter_by_type():
    client = get_client()
    r = client.get("/events", params={"event_type": "LOITERING"})
    body = r.json()
    assert all(e["event_type"] == "LOITERING" for e in body["results"])


def test_dashboard_returns_html():
    client = get_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "CasinoGuard AI" in r.text
