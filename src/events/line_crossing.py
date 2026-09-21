"""Phase 6: Line Crossing Detection.

Virtual Line(두 점으로 정의)을 기준으로 이전/현재 위치의 부호(side)가 바뀌는
순간에만 Crossing 이벤트를 낸다 (지침 14 설계). 부호는 2D 외적(cross product)의
부호로 판정한다 — 어느 쪽에 있는지에 대한 표준적인 기하 판정 방법.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_EPS = 1e-6


def signed_distance(point: tuple[float, float], line_start: tuple[float, float], line_end: tuple[float, float]) -> float:
    """point에서 직선까지의 부호 있는 수직 거리(픽셀 단위).
    외적을 선분 길이로 정규화해 좌표 스케일과 무관하게 실제 거리 단위가 되도록 한다
    (EXP-014: band_px를 "픽셀 단위 불감대"로 의미 있게 쓰려면 정규화가 필요, FC-007).
    """
    (x1, y1), (x2, y2) = line_start, line_end
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < _EPS:
        return 0.0
    cross = dx * (py - y1) - dy * (px - x1)
    return cross / length


def side_of_line(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    band_px: float = 0.0,
) -> int:
    """point가 선의 어느 쪽에 있는지: 양수/음수/0(선 위 또는 band_px 이내 불확정 구역).
    band_px=0(기본값)이면 기존과 동일하게 거리 0만 경계로 취급한다.
    band_px>0이면 선을 중심으로 ±band_px 폭의 "불확정 구역"을 두어, 그 안에서는
    반대쪽으로 완전히 건너가지 않은 것으로 본다(Hysteresis / Schmitt Trigger, FC-007 대응).
    """
    d = signed_distance(point, line_start, line_end)
    if d > band_px + _EPS:
        return 1
    if d < -(band_px + _EPS):
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
    band_px: float = 0.0  # 불감대 폭(픽셀). 0이면 기존 동작과 동일 (FC-007 수정, EXP-014)
    last_side: dict[int, int] = field(default_factory=dict)  # track_id -> -1/0/1 (확정된 마지막 side)

    def update(self, track_id: int, point: tuple[float, float], frame_idx: int) -> CrossingEvent | None:
        current = side_of_line(point, self.line_start, self.line_end, self.band_px)

        if current == 0:
            # 선 위 또는 band_px 불확정 구역: 확정된 side(last_side)를 갱신하지 않고 그대로 유지한다.
            # band_px>0일 때 이 부분이 Hysteresis의 핵심 — 선 근처에서 흔들려도 last_side가
            # 바뀌지 않으므로 반대쪽으로 확실히 넘어가기 전까지는 이벤트가 나지 않는다.
            return None

        previous = self.last_side.get(track_id)
        self.last_side[track_id] = current

        if previous is None or previous == current:
            return None  # 최초 관측이거나 같은 쪽에 머무름 -> 이벤트 없음

        direction = "A_TO_B" if current == 1 else "B_TO_A"
        return CrossingEvent(track_id=track_id, frame_idx=frame_idx, direction=direction)

    def forget_track(self, track_id: int) -> None:
        self.last_side.pop(track_id, None)
