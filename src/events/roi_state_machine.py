"""Phase 5: Intrusion Detection — Polygon ROI + State Machine.

지침 13번 설계:
- ROI 판정 기준점은 Bottom Center 사용 (사람이 서 있는 바닥 위치에 가장 가까움)
- State Machine: OUTSIDE <-> INSIDE, 전이(transition)가 일어날 때만 이벤트 발생
  (같은 상태를 유지하는 동안에는 매 프레임 이벤트를 만들지 않는다 — 중복 이벤트 방지)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    poly = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, point, False)
    return result >= 0  # 0: 경계 위, 양수: 내부


@dataclass
class ROIEvent:
    track_id: int
    event_type: str  # "ENTER" or "EXIT"
    frame_idx: int


@dataclass
class ROIStateMachine:
    """Track별 ROI 상태(OUTSIDE/INSIDE)를 관리하고 전이 시에만 이벤트를 낸다."""

    polygon: list[tuple[float, float]]
    states: dict[int, str] = field(default_factory=dict)  # track_id -> "OUTSIDE" | "INSIDE"

    def update(self, track_id: int, bbox: tuple[float, float, float, float], frame_idx: int) -> ROIEvent | None:
        point = bottom_center(bbox)
        is_inside = point_in_polygon(point, self.polygon)
        new_state = "INSIDE" if is_inside else "OUTSIDE"
        old_state = self.states.get(track_id, "OUTSIDE")

        self.states[track_id] = new_state

        if old_state == new_state:
            return None  # 상태 변화 없음 -> 이벤트 없음 (중복 방지 핵심)
        if new_state == "INSIDE":
            return ROIEvent(track_id=track_id, event_type="ENTER", frame_idx=frame_idx)
        return ROIEvent(track_id=track_id, event_type="EXIT", frame_idx=frame_idx)

    def forget_track(self, track_id: int) -> None:
        """더 이상 보이지 않는 Track의 상태를 정리한다.
        (지침 28: 장시간 실행 시 상태가 무한정 쌓이는 Memory 문제 예방)
        """
        self.states.pop(track_id, None)

    def active_track_count(self) -> int:
        return len(self.states)
