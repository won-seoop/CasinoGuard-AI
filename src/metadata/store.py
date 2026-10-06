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
    trajectory TEXT,                  -- JSON list of [frame_idx, x, y] (bottom-center)
    upper_color TEXT,                 -- 지침 16 확장: Attribute Metadata (EXP-025~029)
    lower_color TEXT
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

    def add_event(self, track_id: int, event_type: str, frame_idx: int, detail: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events(track_id, event_type, frame_idx, detail) VALUES (?,?,?,?)",
            (track_id, event_type, frame_idx, json.dumps(detail or {})),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_event_detail(self, event_id: int, extra: dict) -> None:
        """이미 저장된 Event의 detail JSON에 필드를 병합한다 (덮어쓰지 않고 merge).

        EXP-023: Adaptive Recording Event Clip은 Post-Roll이 끝나야 파일이 완성되므로
        clip 경로를 이벤트가 처음 기록될 때는 아직 알 수 없다 — 나중에 이 메서드로 붙인다.
        """
        cur = self.conn.execute("SELECT detail FROM events WHERE event_id=?", (event_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"event_id {event_id} not found")
        detail = json.loads(row[0]) if row[0] else {}
        detail.update(extra)
        self.conn.execute("UPDATE events SET detail=? WHERE event_id=?", (json.dumps(detail), event_id))
        self.conn.commit()

    def set_dwell(self, track_id: int, dwell_frames: int) -> None:
        self.conn.execute("UPDATE tracks SET dwell_frames=? WHERE track_id=?", (dwell_frames, track_id))
        self.conn.commit()

    def update_attributes(self, track_id: int, upper_color: str, lower_color: str) -> None:
        """BestShot 확정 시점에 Track당 1회 계산된 상/하의 색상을 저장한다 (EXP-029, 지침 16/19).

        BestShot score와 마찬가지로 Track 생애 전체가 아니라 확정된 BestShot 1장만 기준으로
        계산되므로, BestShot을 먼저 upsert(update_bestshot)한 뒤에 호출하는 순서를 따른다
        (강제하지는 않음 - FK 제약과 달리 두 컬럼 모두 Track row 자체에 속해 순서 의존성 없음).
        """
        self.conn.execute(
            "UPDATE tracks SET upper_color=?, lower_color=? WHERE track_id=?",
            (upper_color, lower_color, track_id),
        )
        self.conn.commit()

    def query_tracks(self, min_dwell_frames: int | None = None, zone: str | None = None) -> list[dict]:
        sql = (
            "SELECT track_id, class, first_seen, last_seen, current_zone, dwell_frames, "
            "bestshot_path, upper_color, lower_color FROM tracks WHERE 1=1"
        )
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

    def query_events(
        self,
        event_type: str | None = None,
        track_id: int | None = None,
        has_clip: bool | None = None,
    ) -> list[dict]:
        """지침 17: Event 검색. `event_clip_path`는 detail JSON 안에 있지만(EXP-023,
        update_event_detail로 사후 부착), VMS가 "Event 전후 영상" 위치를 바로 쓸 수 있도록
        최상위 필드로도 펼쳐서 반환한다(EXP-024)."""
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
            r["event_clip_path"] = r["detail"].get("event_clip_path")
        if has_clip is not None:
            rows = [r for r in rows if (r["event_clip_path"] is not None) == has_clip]
        return rows

    def close(self) -> None:
        self.conn.close()
