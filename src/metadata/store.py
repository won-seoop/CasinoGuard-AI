"""Metadata Pipeline — SQLite 기반 저장소.

지침 16번 스키마를 그대로 구현한다.
DB 선택: SQLite vs PostgreSQL(08. Technical Decisions 참고) — 이 프로젝트 규모(단일 프로세스,
포트폴리오용 데모)에서는 별도 서버 프로세스가 필요 없는 SQLite가 적합하다고 판단해 채택.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id INTEGER PRIMARY KEY,
    class TEXT NOT NULL,
    first_seen INTEGER NOT NULL,      -- frame_idx
    last_seen INTEGER NOT NULL,       -- frame_idx
    last_confidence REAL,
    last_bbox TEXT,                   -- JSON [x1,y1,x2,y2]
    bestshot_path TEXT,
    bestshot_score REAL,
    current_zone TEXT,
    dwell_frames INTEGER DEFAULT 0,
    trajectory TEXT                   -- JSON list of [frame_idx, x, y] (bottom-center)
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,         -- INTRUSION_ENTER / INTRUSION_EXIT / LINE_CROSSING / LOITERING
    frame_idx INTEGER NOT NULL,
    detail TEXT,                      -- JSON (direction, dwell_sec 등 이벤트별 부가정보)
    FOREIGN KEY (track_id) REFERENCES tracks(track_id)
);

CREATE INDEX IF NOT EXISTS idx_events_track ON events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


class MetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_track(
        self,
        track_id: int,
        cls: str,
        frame_idx: int,
        confidence: float,
        bbox: tuple[float, float, float, float],
        point: tuple[float, float],
        zone: str | None,
    ) -> None:
        cur = self.conn.execute("SELECT trajectory, first_seen FROM tracks WHERE track_id=?", (track_id,))
        row = cur.fetchone()
        if row is None:
            trajectory = [[frame_idx, point[0], point[1]]]
            self.conn.execute(
                """INSERT INTO tracks(track_id, class, first_seen, last_seen, last_confidence, last_bbox, current_zone, trajectory)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (track_id, cls, frame_idx, frame_idx, confidence, json.dumps(list(bbox)), zone, json.dumps(trajectory)),
            )
        else:
            trajectory = json.loads(row[0])
            trajectory.append([frame_idx, point[0], point[1]])
            self.conn.execute(
                """UPDATE tracks SET last_seen=?, last_confidence=?, last_bbox=?, current_zone=?, trajectory=? WHERE track_id=?""",
                (frame_idx, confidence, json.dumps(list(bbox)), zone, json.dumps(trajectory), track_id),
            )
        self.conn.commit()

    def update_bestshot(self, track_id: int, path: str, score: float) -> None:
        self.conn.execute("UPDATE tracks SET bestshot_path=?, bestshot_score=? WHERE track_id=?", (path, score, track_id))
        self.conn.commit()

    def add_event(self, track_id: int, event_type: str, frame_idx: int, detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events(track_id, event_type, frame_idx, detail) VALUES (?,?,?,?)",
            (track_id, event_type, frame_idx, json.dumps(detail or {})),
        )
        self.conn.commit()

    def set_dwell(self, track_id: int, dwell_frames: int) -> None:
        self.conn.execute("UPDATE tracks SET dwell_frames=? WHERE track_id=?", (dwell_frames, track_id))
        self.conn.commit()

    def query_tracks(self, min_dwell_frames: int | None = None, zone: str | None = None) -> list[dict]:
        sql = "SELECT track_id, class, first_seen, last_seen, current_zone, dwell_frames, bestshot_path FROM tracks WHERE 1=1"
        params: list = []
        if min_dwell_frames is not None:
            sql += " AND dwell_frames >= ?"
            params.append(min_dwell_frames)
        if zone is not None:
            sql += " AND current_zone = ?"
            params.append(zone)
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def query_events(self, event_type: str | None = None, track_id: int | None = None) -> list[dict]:
        sql = "SELECT event_id, track_id, event_type, frame_idx, detail FROM events WHERE 1=1"
        params: list = []
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        if track_id is not None:
            sql += " AND track_id = ?"
            params.append(track_id)
        sql += " ORDER BY frame_idx"
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            r["detail"] = json.loads(r["detail"]) if r["detail"] else {}
        return rows

    def close(self) -> None:
        self.conn.close()
