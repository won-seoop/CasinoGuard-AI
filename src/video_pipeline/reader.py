"""Phase 1: Video Input Pipeline.

CCTV 영상(mp4/webm/ogg 등)을 열어 Frame 단위로 순회하는 최소 기능의 Reader.
- 정상 종료 처리
- 영상 읽기 실패(손상된 파일, 중간에 끊긴 스트림) 처리
- Frame 단위 처리 시간 측정
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int  # OpenCV가 보고하는 값 (컨테이너에 따라 부정확할 수 있음)


@dataclass
class FrameResult:
    index: int
    frame: np.ndarray
    read_time_sec: float  # 이 프레임을 읽는 데 걸린 시간


@dataclass
class PipelineStats:
    """한 번의 영상 처리 실행에 대한 측정 결과."""

    video_path: str
    source_fps: float
    source_resolution: tuple[int, int]
    frames_read: int = 0
    frames_failed: int = 0  # read()가 실패해서 못 받은 프레임 수(추정)
    read_times_sec: list[float] = field(default_factory=list)
    total_time_sec: float = 0.0
    opened_ok: bool = False
    error: Optional[str] = None

    @property
    def processed_fps(self) -> float:
        if self.total_time_sec <= 0:
            return 0.0
        return self.frames_read / self.total_time_sec

    @property
    def avg_read_time_ms(self) -> float:
        if not self.read_times_sec:
            return 0.0
        return 1000 * sum(self.read_times_sec) / len(self.read_times_sec)

    def percentile_ms(self, p: float) -> float:
        if not self.read_times_sec:
            return 0.0
        arr = np.array(self.read_times_sec) * 1000
        return float(np.percentile(arr, p))


class VideoReader:
    """OpenCV VideoCapture를 감싸는 최소 기능 Video Input.

    실패 처리 원칙: 영상을 열지 못하거나 중간에 읽기가 실패해도 예외를 던져
    프로그램 전체를 죽이지 않는다. 대신 PipelineStats.error 에 원인을 남기고
    지금까지 읽은 프레임은 그대로 반환한다. (지침 29. Robustness Test)
    """

    def __init__(self, path: str | Path, resize_to: Optional[tuple[int, int]] = None):
        self.path = str(path)
        self.resize_to = resize_to
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> Optional[VideoMeta]:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            return None
        meta = VideoMeta(
            path=self.path,
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 0.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return meta

    def frames(self) -> Iterator[FrameResult]:
        """Frame을 하나씩 내어준다. 읽기 실패 시 조용히 멈춘다(예외 없음)."""
        if self._cap is None:
            return
        idx = 0
        while True:
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            dt = time.perf_counter() - t0
            if not ok or frame is None:
                # 정상 종료(EOF)인지 중간 손상인지는 이 레벨에서 구분하지 않고
                # 상위 run 루프에서 "기대한 frame_count 대비 실제 읽은 수"로 판단한다.
                break
            if self.resize_to is not None:
                frame = cv2.resize(frame, self.resize_to)
            yield FrameResult(index=idx, frame=frame, read_time_sec=dt)
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_baseline(path: str | Path, resize_to: Optional[tuple[int, int]] = None) -> PipelineStats:
    """EXP-001 Baseline: AI 추론 없이 Frame Read만 수행하고 성능을 측정한다."""
    reader = VideoReader(path, resize_to=resize_to)
    meta = reader.open()

    if meta is None:
        return PipelineStats(
            video_path=str(path),
            source_fps=0.0,
            source_resolution=(0, 0),
            opened_ok=False,
            error="OPEN_FAILED: cv2.VideoCapture가 파일을 열지 못함",
        )

    stats = PipelineStats(
        video_path=str(path),
        source_fps=meta.fps,
        source_resolution=(meta.width, meta.height),
        opened_ok=True,
    )

    t_start = time.perf_counter()
    for fr in reader.frames():
        stats.frames_read += 1
        stats.read_times_sec.append(fr.read_time_sec)
    stats.total_time_sec = time.perf_counter() - t_start
    reader.close()

    # 컨테이너가 보고한 frame_count와 실제로 읽은 frame 수가 크게 다르면
    # "일부 프레임 손상/누락"으로 간주한다 (예: corrupted_truncated.mp4).
    expected = meta.frame_count
    if expected > 0 and stats.frames_read < expected * 0.95:
        stats.frames_failed = max(expected - stats.frames_read, 0)
        stats.error = (
            f"PARTIAL_READ: 컨테이너 기대 frame_count={expected}, "
            f"실제 읽은 frame={stats.frames_read} (손상/조기종료 가능성)"
        )

    return stats
