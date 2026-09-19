"""Phase 6: Line Crossing Detection.

Virtual Line(두 점으로 정의)을 기준으로 이전/현재 위치의 부호(side)가 바뀌는
순간에만 Crossing 이벤트를 낸다 (지침 14 설계). 부호는 2D 외적(cross product)의
부호로 판정한다 — 어느 쪽에 있는지에 대한 표준적인 기하 판정 방법.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def side_of_line(point: tuple[float, float], line_start: tuple[float, float], line_end: tuple[float, float]) -> int:
    """point가 선의 어느 쪽에 있는지: 양수/음수/0(선 위).
    외적 (end-start) x (point-start) 의 부호를 사용한다.
    """
    (x1, y1), (x2, y2) = line_start, line_end
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 1e-6:
        return 1
    if cross < -1e-6:
        return -1
    return 0


@dataclass
class CrossingEvent:
    track_id: int
    frame_idx: int
    direction: str  # "A_TO_B" or "B_TO_A"


@dataclass
class LineCrossingDetector:
    line_start: tuple[float, float]
    line_end: tuple[float, float]
    last_side: dict[int, int] = field(default_factory=dict)  # track_id -> -1/0/1

    def update(self, track_id: int, point: tuple[float, float], frame_idx: int) -> CrossingEvent | None:
        current = side_of_line(point, self.line_start, self.line_end)

        if current == 0:
            # 선 위 정확히 걸친 경우: 애매한 상태이므로 이전 side를 그대로 유지하고
            # 이벤트를 만들지 않는다 (경계에서의 Flicker로 인한 오탐 방지).
            return None

        previous = self.last_side.get(track_id)
        self.last_side[track_id] = current

        if previous is None or previous == current:
            return None  # 최초 관측이거나 같은 쪽에 머무름 -> 이벤트 없음

        direction = "A_TO_B" if current == 1 else "B_TO_A"
        return CrossingEvent(track_id=track_id, frame_idx=frame_idx, direction=direction)

    def forget_track(self, track_id: int) -> None:
        self.last_side.pop(track_id, None)
