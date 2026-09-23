"""Phase (Stretch) Crowd Analysis: ROI(Grid Cell)별 Person Count를 계산한다.

지침 22번 설계:
- ROI별 Person Count를 계산한다 (Normal / Busy / Crowded).
- Threshold는 임의로 정하지 않고, 실제 Dataset의 Distribution을 확인한 뒤 정의한다.

`CrowdAnalyzer`는 프레임을 grid_rows x grid_cols 셀로 나누고 프레임마다
셀별 사람 수를 센다. Threshold는 관측된 셀별 카운트 분포의 percentile로부터
`derive_thresholds_from_distribution()`으로 계산하며, 고정값을 코드에 박아두지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CrowdThresholds:
    normal_max: float  # count <= normal_max -> Normal
    busy_max: float  # normal_max < count <= busy_max -> Busy, count > busy_max -> Crowded


def derive_thresholds_from_distribution(counts: list[float], normal_pct: float = 50.0, busy_pct: float = 90.0) -> CrowdThresholds:
    """실측 카운트 분포의 percentile로 Threshold를 정의한다 (임의 고정값 금지, 지침 22).

    0으로만 이루어진 분포(사람이 거의 없던 구간)처럼 두 percentile이 같아지면
    Busy 구간이 사라지는데, 이는 실제로 그 셀에서 밀집도 변화가 거의 없었다는
    뜻이므로 억지로 구간을 벌리지 않고 그대로 반영한다.
    """
    if not counts:
        return CrowdThresholds(normal_max=0.0, busy_max=0.0)
    arr = np.asarray(counts, dtype=np.float64)
    normal_max = float(np.percentile(arr, normal_pct))
    busy_max = float(np.percentile(arr, busy_pct))
    busy_max = max(busy_max, normal_max)
    return CrowdThresholds(normal_max=normal_max, busy_max=busy_max)


def classify(count: float, thresholds: CrowdThresholds) -> str:
    if count <= thresholds.normal_max:
        return "Normal"
    if count <= thresholds.busy_max:
        return "Busy"
    return "Crowded"


class CrowdAnalyzer:
    """프레임을 grid_rows x grid_cols 셀로 나누고 셀별 사람 수를 센다."""

    def __init__(self, frame_w: int, frame_h: int, grid_rows: int, grid_cols: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_w = frame_w / grid_cols
        self.cell_h = frame_h / grid_rows

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        if x < 0 or y < 0 or x >= self.frame_w or y >= self.frame_h:
            return None
        col = min(int(x // self.cell_w), self.grid_cols - 1)
        row = min(int(y // self.cell_h), self.grid_rows - 1)
        return row, col

    def count_frame(self, points: list[tuple[float, float]]) -> np.ndarray:
        """한 프레임의 포인트 목록으로부터 grid_rows x grid_cols 카운트 행렬을 만든다."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        for x, y in points:
            cell = self.cell_of(x, y)
            if cell is not None:
                grid[cell] += 1
        return grid
