"""VMS Search API — 최소 기능만 구현한다 (지침 17: Microservice/K8s/복잡한 인증/Cloud 없음).

FastAPI로 Metadata Pipeline(SQLite)에 저장된 Track/Event를 검색한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

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
    events = conn.execute("SELECT * FROM events WHERE track_id=? ORDER BY frame_idx", (track_id,)).fetchall()
    conn.close()
    if track is None:
        return {"error": "not found"}
    return {"track": dict(track), "events": [dict(e) for e in events]}


@app.get("/events")
def list_events(event_type: str | None = Query(default=None)):
    """지침 17: 특정 Event Type이 발생한 시점 조회 (예: LOITERING, INTRUSION_ENTER, LINE_CROSSING)."""
    conn = get_conn()
    sql = "SELECT * FROM events WHERE 1=1"
    params: list = []
    if event_type is not None:
        sql += " AND event_type = ?"
        params.append(event_type)
    sql += " ORDER BY frame_idx"
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return {"count": len(rows), "results": rows}


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
        "SELECT event_id, track_id, event_type, frame_idx FROM events ORDER BY frame_idx DESC LIMIT 10"
    ).fetchall()
    conn.close()

    rows_dwell = "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>" for r in top_dwell)
    rows_events = "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>" for r in recent_events)

    return f"""
    <html><head><title>CasinoGuard AI VMS</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 40px auto;">
    <h1>CasinoGuard AI — VMS Search (Demo)</h1>
    <p>Tracks: {n_tracks} / Events: {n_events}</p>
    <h2>Dwell Time 상위 5개 Track</h2>
    <table border="1" cellpadding="6"><tr><th>track_id</th><th>dwell_frames</th><th>bestshot_path</th></tr>{rows_dwell}</table>
    <h2>최근 Event 10개</h2>
    <table border="1" cellpadding="6"><tr><th>event_id</th><th>track_id</th><th>type</th><th>frame_idx</th></tr>{rows_events}</table>
    <p>API: <a href="/docs">/docs</a> (Swagger)</p>
    </body></html>
    """
