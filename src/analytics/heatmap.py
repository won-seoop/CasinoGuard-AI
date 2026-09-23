"""Phase (Stretch) Heatmap: Track 위치를 시간에 따라 누적한다.

지침 21번 설계:
- Detection Center 누적 vs Track Trajectory 누적을 비교할 수 있다.
- 시간 단위: 전체 / 1분 / 5분.

이 모듈은 두 가지 누적 방식을 모두 제공한다.

1) `HeatmapAccumulator` — 그리드 셀 단위로 점을 누적하는 순수 자료구조.
   "무엇을 누적할지"는 호출자가 결정한다 (Detection Center든 Track Point든 동일하게 사용).
2) `TrackGatedFeeder` — Track이 최소 길이(min_track_len)에 도달하기 전까지는
   포인트를 버퍼에 보류하고, 도달하면 한 번에 커밋한다. 최소 길이에 도달하지 못하고
   사라진 Track(예: 짧게 스쳐간 오탐성 Track)은 버퍼째 버려 누적 대상에서 제외한다.
   EXP-016에서 실측한 문제: raw per-frame Detection Center 누적은 Pan/Jitter 경계에서
   발생하는 짧은 오탐성 Track까지 그대로 누적해 Heatmap에 노이즈를 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HeatmapAccumulator:
    """프레임을 (cell_size x cell_size) 그리드로 나누고 포인트를 누적한다."""

    frame_w: int
    frame_h: int
    cell_size: int = 20
    time_bin_sec: float | None = None  # None이면 전체 누적만. 지정하면 bin별로도 따로 쌓는다.
    total_grid: np.ndarray = field(init=False)
    bins: dict[int, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cols = max(1, (self.frame_w + self.cell_size - 1) // self.cell_size)
        self.rows = max(1, (self.frame_h + self.cell_size - 1) // self.cell_size)
        self.total_grid = np.zeros((self.rows, self.cols), dtype=np.float64)

    def _cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        if x < 0 or y < 0 or x >= self.frame_w or y >= self.frame_h:
            return None
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        col = min(col, self.cols - 1)
        row = min(row, self.rows - 1)
        return row, col

    def add_point(self, x: float, y: float, timestamp_sec: float | None = None, weight: float = 1.0) -> None:
        cell = self._cell_of(x, y)
        if cell is None:
            return  # 프레임 밖 좌표는 누적하지 않는다 (경계 밖 노이즈 방지).
        row, col = cell
        self.total_grid[row, col] += weight

        if self.time_bin_sec is not None and timestamp_sec is not None:
            bin_key = int(timestamp_sec // self.time_bin_sec)
            if bin_key not in self.bins:
                self.bins[bin_key] = np.zeros((self.rows, self.cols), dtype=np.float64)
            self.bins[bin_key][row, col] += weight

    def get_grid(self, bin_key: int | None = None) -> np.ndarray:
        if bin_key is None:
            return self.total_grid
        return self.bins.get(bin_key, np.zeros((self.rows, self.cols), dtype=np.float64))

    def total_hits(self) -> float:
        return float(self.total_grid.sum())

    def to_normalized_image(self) -> np.ndarray:
        """0~255 uint8 grayscale 히트맵 (cv2.applyColorMap과 함께 사용)."""
        grid = self.total_grid
        max_val = grid.max()
        if max_val <= 0:
            norm = np.zeros_like(grid, dtype=np.uint8)
        else:
            norm = (grid / max_val * 255.0).astype(np.uint8)
        return np.kron(norm, np.ones((self.cell_size, self.cell_size), dtype=np.uint8))[: self.frame_h, : self.frame_w]


@dataclass
class _TrackBuffer:
    points: list[tuple[float, float, float | None]] = field(default_factory=list)
    confirmed: bool = False


class TrackGatedFeeder:
    """Track이 min_track_len 프레임 이상 관측된 뒤에만 HeatmapAccumulator로 흘려보낸다.

    min_track_len 미만에서 사라진 Track은 누적되지 않는다 (노이즈 필터링).
    """

    def __init__(self, accumulator: HeatmapAccumulator, min_track_len: int):
        self.accumulator = accumulator
        self.min_track_len = min_track_len
        self._buffers: dict[int, _TrackBuffer] = {}
        self.discarded_track_count = 0
        self.discarded_point_count = 0
        self.confirmed_track_count = 0

    def observe(self, track_id: int, x: float, y: float, timestamp_sec: float | None = None) -> None:
        buf = self._buffers.setdefault(track_id, _TrackBuffer())
        if buf.confirmed:
            self.accumulator.add_point(x, y, timestamp_sec)
            return
        buf.points.append((x, y, timestamp_sec))
        if len(buf.points) >= self.min_track_len:
            for px, py, pt in buf.points:
                self.accumulator.add_point(px, py, pt)
            buf.confirmed = True
            buf.points.clear()
            self.confirmed_track_count += 1

    def forget_track(self, track_id: int) -> None:
        buf = self._buffers.pop(track_id, None)
        if buf is not None and not buf.confirmed:
            self.discarded_track_count += 1
            self.discarded_point_count += len(buf.points)
