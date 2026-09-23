import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytics.heatmap import HeatmapAccumulator, TrackGatedFeeder  # noqa: E402


def test_add_point_accumulates_into_correct_cell():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10)
    acc.add_point(15, 25)  # -> row=2, col=1
    grid = acc.get_grid()
    assert grid[2, 1] == 1.0
    assert grid.sum() == 1.0


def test_add_point_out_of_frame_is_ignored():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10)
    acc.add_point(-5, 5)
    acc.add_point(150, 5)
    assert acc.total_hits() == 0.0


def test_add_point_on_exact_boundary_is_clamped_to_last_cell():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10)
    acc.add_point(99.999, 99.999)
    grid = acc.get_grid()
    assert grid[-1, -1] == 1.0


def test_time_bins_are_separate_from_total():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10, time_bin_sec=60.0)
    acc.add_point(5, 5, timestamp_sec=10.0)  # bin 0
    acc.add_point(5, 5, timestamp_sec=70.0)  # bin 1
    assert acc.get_grid()[0, 0] == 2.0
    assert acc.get_grid(bin_key=0)[0, 0] == 1.0
    assert acc.get_grid(bin_key=1)[0, 0] == 1.0


def test_to_normalized_image_shape_matches_frame():
    acc = HeatmapAccumulator(frame_w=105, frame_h=95, cell_size=10)
    acc.add_point(5, 5)
    img = acc.to_normalized_image()
    assert img.shape == (95, 105)
    assert img.max() == 255


def test_to_normalized_image_all_zero_when_empty():
    acc = HeatmapAccumulator(frame_w=50, frame_h=50, cell_size=10)
    img = acc.to_normalized_image()
    assert img.max() == 0


def test_track_gated_feeder_discards_short_track():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10)
    feeder = TrackGatedFeeder(acc, min_track_len=5)
    for i in range(3):  # 최소 길이(5)에 못 미침
        feeder.observe(track_id=1, x=5, y=5, timestamp_sec=float(i))
    feeder.forget_track(1)
    assert acc.total_hits() == 0.0
    assert feeder.discarded_track_count == 1
    assert feeder.discarded_point_count == 3
    assert feeder.confirmed_track_count == 0


def test_track_gated_feeder_commits_buffered_points_once_confirmed():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10)
    feeder = TrackGatedFeeder(acc, min_track_len=3)
    for i in range(3):
        feeder.observe(track_id=1, x=5, y=5, timestamp_sec=float(i))
    assert acc.total_hits() == 3.0  # 3번째 관측에서 버퍼 전체가 한 번에 커밋됨
    assert feeder.confirmed_track_count == 1

    feeder.observe(track_id=1, x=5, y=5, timestamp_sec=3.0)  # 확정 이후는 즉시 누적
    assert acc.total_hits() == 4.0

    feeder.forget_track(1)  # 이미 확정된 Track은 forget해도 버려지지 않음
    assert feeder.discarded_track_count == 0


def test_track_gated_feeder_independent_tracks():
    acc = HeatmapAccumulator(frame_w=100, frame_h=100, cell_size=10)
    feeder = TrackGatedFeeder(acc, min_track_len=2)
    feeder.observe(track_id=1, x=5, y=5)
    feeder.observe(track_id=2, x=15, y=15)
    feeder.forget_track(1)  # track 1은 1프레임만 관측 -> 버려짐
    feeder.observe(track_id=2, x=15, y=15)  # track 2는 2프레임 관측 -> 확정
    assert feeder.discarded_track_count == 1
    assert feeder.confirmed_track_count == 1
    assert acc.total_hits() == 2.0
