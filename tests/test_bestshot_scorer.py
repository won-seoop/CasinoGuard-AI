import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bestshot.scorer import (  # noqa: E402
    occlusion_score,
    position_score,
    position_score_margin_legacy,
    size_score,
)


def test_size_score_larger_box_scores_higher():
    small = size_score((0, 0, 10, 10), 1000, 1000)
    large = size_score((0, 0, 200, 200), 1000, 1000)
    assert large > small


def test_size_score_capped_at_one():
    huge = size_score((0, 0, 900, 900), 1000, 1000)
    assert huge == 1.0


def test_position_score_penalizes_edge_touching_box():
    center = position_score((400, 400, 500, 500), 1000, 1000)
    edge = position_score((0, 400, 100, 500), 1000, 1000)
    assert center > edge


def test_occlusion_score_no_neighbors_is_perfect():
    assert occlusion_score((0, 0, 10, 10), []) == 1.0


def test_occlusion_score_full_overlap_is_zero():
    box = (0, 0, 10, 10)
    v = occlusion_score(box, [box])
    assert v == 0.0


def test_occlusion_score_partial_overlap_between_zero_and_one():
    v = occlusion_score((0, 0, 10, 10), [(5, 0, 15, 10)])
    assert 0.0 < v < 1.0


# --- FC-003 / EXP-018: position_score 경계 근접 vs 실제 잘림 구분 ---


def test_position_score_touching_boundary_is_penalized():
    edge = position_score((0, 400, 100, 500), 1000, 1000)
    assert edge < 1.0


def test_position_score_near_but_not_touching_boundary_is_full_score():
    # legacy margin_ratio=0.02 기준으로는 1000*0.02=20px 이내라 0.3점을 받았을 구간(x1=15)
    # 이지만, 실제로 경계에 닿지 않았으므로 새 구현은 만점을 줘야 한다.
    near_edge_not_clipped = position_score((15, 400, 115, 500), 1000, 1000)
    assert near_edge_not_clipped == 1.0


def test_position_score_graduated_by_number_of_edges_touched():
    one_edge = position_score((0, 400, 100, 500), 1000, 1000)
    corner_two_edges = position_score((0, 0, 100, 100), 1000, 1000)
    assert corner_two_edges < one_edge < 1.0


def test_position_score_respects_boundary_epsilon_tolerance():
    # Detector 좌표 반올림 오차(1.5px)는 "실제로 닿음"으로 봐야 한다 (기본 eps=2.0).
    barely_over = position_score((1.5, 400, 100, 500), 1000, 1000)
    assert barely_over < 1.0


def test_position_score_legacy_flags_near_edge_as_clipped_new_does_not():
    """FC-003 재현: 실제로 안 잘렸는데도 legacy는 0.3점을 주는 False Penalty 사례."""
    bbox = (10, 400, 110, 500)  # 1000px 프레임에서 10px 여백 (margin 2%=20px 이내)
    legacy = position_score_margin_legacy(bbox, 1000, 1000)
    fixed = position_score(bbox, 1000, 1000)
    assert legacy == 0.3
    assert fixed == 1.0
