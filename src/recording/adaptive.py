"""Stretch Goal: Adaptive Recording (지침 23번).

Baseline: 항상 30 FPS High Quality Recording.
Adaptive:
    No Person       -> Low FPS (Frame Skip + 저해상도)
    Person Detected -> Normal FPS
    Security Event  -> High FPS/Quality + Circular Buffer 기반 Event Clip(-10s ~ +10s)

이 모듈은 순수 의사결정 로직만 담당한다 (비디오 인코딩/파일 I/O는 실험 스크립트에서 처리).
그래야 실제 카메라나 코덱 없이도 pytest로 상태 전이/병합 로직을 검증할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RecordingTier(Enum):
    IDLE = "idle"
    NORMAL = "normal"
    EVENT = "event"


@dataclass(frozen=True)
class AdaptiveRecordingConfig:
    fps: float = 25.0
    pre_roll_sec: float = 10.0
    post_roll_sec: float = 10.0
    idle_frame_stride: int = 5
    merge_gap_sec: float = 0.0  # 겹치거나 딱 맞닿은 Event Clip Window만 병합 (대안 B 기본값)

    @property
    def pre_roll_frames(self) -> int:
        return round(self.fps * self.pre_roll_sec)

    @property
    def post_roll_frames(self) -> int:
        return round(self.fps * self.post_roll_sec)

    @property
    def merge_gap_frames(self) -> int:
        return round(self.fps * self.merge_gap_sec)


def compute_tier_sequence(
    person_present: list[bool], event_active: list[bool], post_roll_frames: int
) -> list[RecordingTier]:
    """프레임별 Recording Tier를 결정한다.

    Event가 활성 상태이거나, 활성 종료 후 post_roll_frames 이내면 EVENT Tier를 유지한다
    (Security Event 발생 후에도 곧바로 화질/FPS를 낮추지 않기 위함 — 사후 상황 파악용 여유 확보).
    그 외에는 사람이 있으면 NORMAL, 없으면 IDLE.
    """
    if len(person_present) != len(event_active):
        raise ValueError("person_present와 event_active의 길이가 같아야 한다")

    tiers: list[RecordingTier] = []
    last_event_frame = -(post_roll_frames + 1)  # 아직 Event가 없었던 상태
    for idx, (present, active) in enumerate(zip(person_present, event_active)):
        if active:
            last_event_frame = idx
        if idx - last_event_frame <= post_roll_frames:
            tiers.append(RecordingTier.EVENT)
        elif present:
            tiers.append(RecordingTier.NORMAL)
        else:
            tiers.append(RecordingTier.IDLE)
    return tiers


def should_keep_idle_frame(frame_idx: int, stride: int) -> bool:
    """IDLE Tier에서 Low FPS 저장을 위한 Frame Skip 여부. stride=1이면 스킵 없음."""
    if stride <= 1:
        return True
    return frame_idx % stride == 0


@dataclass(frozen=True)
class EventInterval:
    """하나의 원본 Security Event가 활성 상태였던 구간 (Pre/Post Roll 확장 전)."""

    event_id: int
    start_frame: int
    end_frame: int  # inclusive


@dataclass(frozen=True)
class EventClipPlan:
    """Pre/Post Roll을 적용하고(필요시) 병합한 뒤 실제로 파일로 저장할 Clip 구간."""

    clip_id: int
    event_ids: tuple[int, ...]
    start_frame: int  # inclusive, [0, total_frames) 범위로 clamp됨
    end_frame: int  # inclusive


def event_intervals_from_active_flags(event_active: list[bool]) -> list[EventInterval]:
    """event_active Boolean 시퀀스에서 연속된 True 구간을 Event Interval로 추출한다."""
    intervals: list[EventInterval] = []
    start: int | None = None
    for idx, active in enumerate(event_active):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            intervals.append(EventInterval(event_id=len(intervals), start_frame=start, end_frame=idx - 1))
            start = None
    if start is not None:
        intervals.append(EventInterval(event_id=len(intervals), start_frame=start, end_frame=len(event_active) - 1))
    return intervals


def plan_event_clips(
    events: list[EventInterval],
    pre_roll_frames: int,
    post_roll_frames: int,
    total_frames: int,
    merge_gap_frames: int = 0,
) -> list[EventClipPlan]:
    """Event 구간을 -pre_roll ~ +post_roll로 확장하고 Clip Window를 계획한다.

    merge_gap_frames >= 0 이면 겹치거나 merge_gap_frames 이내로 가까운 Window를 하나의
    Clip으로 합친다(대안 B, 기본 채택). merge_gap_frames를 매우 작은 음수로 주면 사실상
    병합을 하지 않는 대안 A(Event마다 독립 Clip)를 재현할 수 있다 — 이 경우 Window가
    겹쳐도 별도 파일로 남아 같은 프레임이 여러 Clip에 중복 저장될 수 있다.
    """
    if not events:
        return []
    windows = []
    for ev in sorted(events, key=lambda e: e.start_frame):
        start = max(0, ev.start_frame - pre_roll_frames)
        end = min(total_frames - 1, ev.end_frame + post_roll_frames)
        windows.append((start, end, ev.event_id))

    merged: list[EventClipPlan] = []
    cur_start, cur_end, cur_ids = windows[0][0], windows[0][1], [windows[0][2]]
    for start, end, eid in windows[1:]:
        if start <= cur_end + merge_gap_frames:
            cur_end = max(cur_end, end)
            cur_ids.append(eid)
        else:
            merged.append(EventClipPlan(clip_id=len(merged), event_ids=tuple(cur_ids), start_frame=cur_start, end_frame=cur_end))
            cur_start, cur_end, cur_ids = start, end, [eid]
    merged.append(EventClipPlan(clip_id=len(merged), event_ids=tuple(cur_ids), start_frame=cur_start, end_frame=cur_end))
    return merged


def total_planned_frames(clips: list[EventClipPlan]) -> int:
    """Clip Plan들이 실제로 저장할 총 프레임 수 (겹치는 구간이 있으면 중복 카운트됨 -> 낭비 측정용)."""
    return sum(c.end_frame - c.start_frame + 1 for c in clips)
