"""VMS Search API — 최소 기능만 구현한다 (지침 17: Microservice/K8s/복잡한 인증/Cloud 없음).

FastAPI로 Metadata Pipeline(SQLite)에 저장된 Track/Event를 검색한다.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from metadata.store import MetadataStore

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "results/metadata_pipeline/metadata.db"

app = FastAPI(title="CasinoGuard AI — VMS Search API")


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def health():
    return {"status": "ok", "db_exists": DB_PATH.exists()}


@app.get("/tracks")
def list_tracks(
    zone: str | None = None,
    min_dwell_sec: float | None = None,
    fps: float = 23.976,
):
    """지침 17: 특정 ROI에 일정 시간 이상 있었던 사람 검색."""
    conn = get_conn()
    sql = "SELECT track_id, class, first_seen, last_seen, current_zone, dwell_frames, bestshot_path, bestshot_score FROM tracks WHERE 1=1"
    params: list = []
    if zone is not None:
        sql += " AND current_zone = ?"
        params.append(zone)
    if min_dwell_sec is not None:
        sql += " AND dwell_frames >= ?"
        params.append(min_dwell_sec * fps)
    sql += " ORDER BY dwell_frames DESC"
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return {"count": len(rows), "results": rows}


@app.get("/tracks/{track_id}")
def get_track(track_id: int):
    """지침 17: 특정 Track ID의 이동 기록 조회."""
    conn = get_conn()
    track = conn.execute("SELECT * FROM tracks WHERE track_id=?", (track_id,)).fetchone()
    conn.close()
    if track is None:
        return {"error": "not found"}
    store = MetadataStore(DB_PATH)
    events = store.query_events(track_id=track_id)
    store.close()
    return {"track": dict(track), "events": events}


@app.get("/events")
def list_events(
    event_type: str | None = Query(default=None),
    has_clip: bool | None = Query(default=None, description="event_clip_path가 있는 Event만(True) 혹은 없는 Event만(False) 필터링"),
):
    """지침 17: 특정 Event Type이 발생한 시점 / Event 전후 영상(event_clip_path) 조회.

    EXP-024: 기존에는 raw SQL `SELECT *`로 detail을 JSON 문자열 그대로 반환해(파싱 안 됨)
    event_clip_path를 꺼내려면 클라이언트가 직접 json.loads 해야 했다. MetadataStore.query_events()를
    재사용해 detail을 dict로 파싱하고 event_clip_path를 최상위 필드로도 노출한다.
    """
    store = MetadataStore(DB_PATH)
    rows = store.query_events(event_type=event_type, has_clip=has_clip)
    store.close()
    return {"count": len(rows), "results": rows}


@app.get("/events/{event_id}/clip")
def get_event_clip(event_id: int):
    """지침 17: 특정 Event 전후 영상(Event Clip) 재생/다운로드."""
    conn = get_conn()
    row = conn.execute("SELECT detail FROM events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="event not found")
    detail = json.loads(row["detail"]) if row["detail"] else {}
    clip_path = detail.get("event_clip_path")
    if not clip_path:
        raise HTTPException(status_code=404, detail="no clip recorded for this event")
    full_path = ROOT / clip_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="clip file missing on disk")
    return FileResponse(str(full_path), media_type="video/mp4", filename=full_path.name)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """지침 17: 간단한 VMS Dashboard (Frontend 디자인에 시간 쓰지 않음, 최소 HTML만)."""
    conn = get_conn()
    n_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    top_dwell = conn.execute(
        "SELECT track_id, dwell_frames, bestshot_path FROM tracks ORDER BY dwell_frames DESC LIMIT 5"
    ).fetchall()
    recent_events = conn.execute(
        "SELECT event_id, track_id, event_type, frame_idx, detail FROM events ORDER BY frame_idx DESC LIMIT 10"
    ).fetchall()
    conn.close()

    def clip_cell(event_id: int, detail_json: str | None) -> str:
        detail = json.loads(detail_json) if detail_json else {}
        if not detail.get("event_clip_path"):
            return "-"
        return f'<a href="/events/{event_id}/clip">▶ clip</a>'

    rows_dwell = "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>" for r in top_dwell)
    rows_events = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{clip_cell(r[0], r[4])}</td></tr>"
        for r in recent_events
    )

    return f"""
    <html><head><title>CasinoGuard AI VMS</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 40px auto;">
    <h1>CasinoGuard AI — VMS Search (Demo)</h1>
    <p>Tracks: {n_tracks} / Events: {n_events}</p>
    <h2>Dwell Time 상위 5개 Track</h2>
    <table border="1" cellpadding="6"><tr><th>track_id</th><th>dwell_frames</th><th>bestshot_path</th></tr>{rows_dwell}</table>
    <h2>최근 Event 10개</h2>
    <table border="1" cellpadding="6"><tr><th>event_id</th><th>track_id</th><th>type</th><th>frame_idx</th><th>event clip</th></tr>{rows_events}</table>
    <p>API: <a href="/docs">/docs</a> (Swagger)</p>
    </body></html>
    """
