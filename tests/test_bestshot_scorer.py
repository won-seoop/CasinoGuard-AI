import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bestshot.scorer import occlusion_score, position_score, size_score  # noqa: E402


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
