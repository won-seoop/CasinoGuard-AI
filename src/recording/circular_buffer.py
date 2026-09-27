"""Circular Buffer for Adaptive Recording (지침 23) — 최근 N프레임의 고화질 원본 프레임을
Tier와 무관하게 항상 보관한다.

EXP-019/PAR-010의 Hybrid Event Clip 생성 방식은 IDLE Tier와 겹치는 구간의 원본 프레임을
`make_frame()`으로 다시 "생성"해서 메웠다. 이는 결정론적 합성 시나리오에서만 가능한 편법이며,
실제 카메라는 이미 지나간 원본 프레임을 재생성할 수 없다. 이 모듈은 그 재생성을 "버퍼에서
꺼내기"로 대체하기 위한 순수 자료구조다 (adaptive.py와 동일하게 인코딩/파일 I/O는 실험
스크립트가 담당한다 — 그래야 실제 카메라나 코덱 없이도 pytest로 검증할 수 있다).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

FrameT = TypeVar("FrameT")


@dataclass(frozen=True)
class BufferCoverage:
    """[start_frame, end_frame] 요청 구간에 대해 버퍼가 실제로 커버하는 정도."""

    start_frame: int
    end_frame: int
    missing_frames: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_frames) == 0


class FrameCircularBuffer(Generic[FrameT]):
    """고정 용량 Ring Buffer. 용량을 넘으면 가장 오래된 프레임부터 evict한다."""

    def __init__(self, capacity_frames: int):
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        self.capacity_frames = capacity_frames
        self._frames: dict[int, FrameT] = {}
        self._order: deque[int] = deque()

    def push(self, frame_idx: int, frame: FrameT) -> None:
        """새 프레임을 추가한다. Tier(IDLE/NORMAL/EVENT)와 무관하게 매 프레임 호출되어야
        한다 — 어떤 프레임이 나중에 Event Clip에 필요할지 미리 알 수 없는 실제 카메라
        환경을 재현하기 위함이다.
        """
        if frame_idx in self._frames:
            raise ValueError(f"frame_idx {frame_idx} already pushed")
        self._frames[frame_idx] = frame
        self._order.append(frame_idx)
        if len(self._order) > self.capacity_frames:
            oldest = self._order.popleft()
            del self._frames[oldest]

    def __len__(self) -> int:
        return len(self._frames)

    def coverage(self, start_frame: int, end_frame: int) -> BufferCoverage:
        """[start_frame, end_frame](inclusive) 중 이미 evict됐거나 아직 push되지 않은
        프레임 인덱스를 missing_frames로 보고한다. 절대 조용히 일부만 반환하지 않는다 —
        호출자가 완전성을 먼저 확인하게 강제한다(EXP-019 Naive Stream Copy의 조용한
        불완전성, FC-008과 같은 실수를 이 계층에서 다시 만들지 않기 위함).
        """
        missing = tuple(i for i in range(start_frame, end_frame + 1) if i not in self._frames)
        return BufferCoverage(start_frame=start_frame, end_frame=end_frame, missing_frames=missing)

    def get_range(self, start_frame: int, end_frame: int) -> list[FrameT]:
        """[start_frame, end_frame] 프레임을 순서대로 반환한다. 하나라도 누락되면
        예외를 던진다 — 누락을 감춘 채 부분 결과를 반환하지 않는다.
        """
        cov = self.coverage(start_frame, end_frame)
        if not cov.is_complete:
            raise KeyError(
                f"buffer missing {len(cov.missing_frames)} frame(s) in [{start_frame},{end_frame}]: "
                f"{cov.missing_frames[:5]}{'...' if len(cov.missing_frames) > 5 else ''}"
            )
        return [self._frames[i] for i in range(start_frame, end_frame + 1)]

    def memory_bytes(self, frame_nbytes: int) -> int:
        """현재 버퍼가 실제로 들고 있는 프레임 수 기준 메모리 사용량 추정치."""
        return len(self._frames) * frame_nbytes


def required_capacity_for_pre_roll(pre_roll_frames: int, safety_margin_frames: int = 0) -> int:
    """Pre-Roll을 항상 완전히 복원하기 위한 최소 버퍼 용량을 계산한다.

    Event는 자신이 트리거된 프레임(t) 자체도 Pre-Roll 구간에 포함해서 버퍼에 남아있어야
    한다 — 트리거 순간 버퍼에서 즉시 [t-pre_roll_frames, t] 구간(inclusive, 총
    pre_roll_frames+1개 프레임)을 꺼내 쓰기 때문이다. 따라서 버퍼 용량을 pre_roll_frames와
    "같게"만 설정하면 정확히 1프레임이 모자라는 Off-by-One이 발생한다 — capacity는 최소
    pre_roll_frames + 1 이어야 한다. safety_margin_frames는 스케줄링 지연이나 프레임레이트
    변동을 흡수하기 위한 여유분이다.
    """
    if pre_roll_frames < 0 or safety_margin_frames < 0:
        raise ValueError("pre_roll_frames와 safety_margin_frames는 음수일 수 없다")
    return pre_roll_frames + 1 + safety_margin_frames
