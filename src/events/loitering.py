"""Phase 7: Loitering Detection.

Baseline: Dwell Time(ROI 내 연속 체류 시간) > Threshold 이면 Loitering 이벤트.
같은 체류 구간 동안 이벤트가 반복 발생하지 않도록 Track별로 "이미 알림을 보냈는지" 플래그를 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .roi_state_machine import point_in_polygon


@dataclass
class LoiteringEvent:
    track_id: int
    frame_idx: int
    dwell_frames: int


@dataclass
class LoiteringDetector:
    polygon: list[tuple[float, float]]
    threshold_frames: int
    enter_frame: dict[int, int] = field(default_factory=dict)  # track_id -> ROI에 들어온 frame_idx
    already_fired: dict[int, bool] = field(default_factory=dict)  # 이번 체류 구간에 이미 이벤트를 냈는지

    def update(self, track_id: int, point: tuple[float, float], frame_idx: int) -> LoiteringEvent | None:
        inside = point_in_polygon(point, self.polygon)

        if not inside:
            # ROI 밖으로 나가면 체류 타이머를 리셋 (다음에 다시 들어오면 새로 측정)
            self.enter_frame.pop(track_id, None)
            self.already_fired.pop(track_id, None)
            return None

        if track_id not in self.enter_frame:
            self.enter_frame[track_id] = frame_idx
            self.already_fired[track_id] = False
            return None  # 방금 들어옴 -> 아직 체류시간 0

        dwell = frame_idx - self.enter_frame[track_id]
        if dwell >= self.threshold_frames and not self.already_fired.get(track_id, False):
            self.already_fired[track_id] = True
            return LoiteringEvent(track_id=track_id, frame_idx=frame_idx, dwell_frames=dwell)
        return None

    def forget_track(self, track_id: int) -> None:
        self.enter_frame.pop(track_id, None)
        self.already_fired.pop(track_id, None)
