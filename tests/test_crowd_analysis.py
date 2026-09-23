import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytics.crowd import CrowdAnalyzer, classify, derive_thresholds_from_distribution  # noqa: E402


def test_cell_of_maps_point_to_correct_grid_cell():
    analyzer = CrowdAnalyzer(frame_w=90, frame_h=60, grid_rows=3, grid_cols=3)
    assert analyzer.cell_of(5, 5) == (0, 0)
    assert analyzer.cell_of(89, 59) == (2, 2)
    assert analyzer.cell_of(45, 30) == (1, 1)


def test_cell_of_out_of_frame_returns_none():
    analyzer = CrowdAnalyzer(frame_w=90, frame_h=60, grid_rows=3, grid_cols=3)
    assert analyzer.cell_of(-1, 5) is None
    assert analyzer.cell_of(5, 999) is None


def test_count_frame_counts_multiple_points_in_same_cell():
    analyzer = CrowdAnalyzer(frame_w=90, frame_h=60, grid_rows=3, grid_cols=3)
    grid = analyzer.count_frame([(5, 5), (10, 10), (50, 30)])
    assert grid[0, 0] == 2
    assert grid[1, 1] == 1
    assert grid.sum() == 3


def test_count_frame_empty_points_returns_all_zero():
    analyzer = CrowdAnalyzer(frame_w=90, frame_h=60, grid_rows=3, grid_cols=3)
    grid = analyzer.count_frame([])
    assert grid.sum() == 0


def test_derive_thresholds_from_distribution_matches_percentiles():
    counts = list(range(0, 101))  # 0..100 균등분포
    th = derive_thresholds_from_distribution(counts, normal_pct=50.0, busy_pct=90.0)
    assert th.normal_max == 50.0
    assert th.busy_max == 90.0


def test_derive_thresholds_empty_distribution_is_zero():
    th = derive_thresholds_from_distribution([])
    assert th.normal_max == 0.0
    assert th.busy_max == 0.0


def test_derive_thresholds_busy_never_below_normal():
    # normal_pct > busy_pct로 잘못 호출해도 busy_max가 normal_max보다 작아지지 않아야 함
    th = derive_thresholds_from_distribution([1, 2, 3, 4, 5], normal_pct=90.0, busy_pct=10.0)
    assert th.busy_max >= th.normal_max


def test_classify_boundaries():
    th = derive_thresholds_from_distribution([0, 0, 0, 2, 2, 5, 5, 5, 5, 10])
    assert classify(th.normal_max, th) == "Normal"
    assert classify(th.busy_max, th) == "Busy" or classify(th.busy_max, th) == "Normal"
    assert classify(th.busy_max + 1, th) == "Crowded"


def test_classify_all_zero_distribution_has_no_degenerate_crowded():
    # 사람이 거의 없던 셀(전부 0)에서 실제로 1명이라도 나타나면 Crowded로 잡혀야 함
    th = derive_thresholds_from_distribution([0, 0, 0, 0, 0])
    assert th.normal_max == 0.0
    assert classify(0, th) == "Normal"
    assert classify(1, th) == "Crowded"
