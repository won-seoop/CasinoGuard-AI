import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bestshot.tracker import BestShotTracker  # noqa: E402


def _crop(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


def test_keeps_only_highest_score_candidate():
    tracker = BestShotTracker(100, 100)
    tracker.observe(track_id=1, frame_idx=0, crop=_crop(1), score=0.3)
    tracker.observe(track_id=1, frame_idx=1, crop=_crop(2), score=0.9)
    tracker.observe(track_id=1, frame_idx=2, crop=_crop(3), score=0.5)

    best = tracker.best_of(1)
    assert best.frame_idx == 1
    assert best.score == 0.9


def test_buffered_count_is_bounded_by_active_track_count_not_observations():
    tracker = BestShotTracker(100, 100)
    for i in range(1000):
        tracker.observe(track_id=1, frame_idx=i, crop=_crop(i % 255), score=float(i))

    # 1000번 관측했지만 Track 1개당 후보 1개만 유지되어야 한다 (O(1) 메모리 핵심 검증)
    assert tracker.buffered_count() == 1


def test_forget_track_returns_final_best_and_frees_memory():
    tracker = BestShotTracker(100, 100)
    tracker.observe(track_id=5, frame_idx=10, crop=_crop(7), score=0.8)

    final = tracker.forget_track(5)
    assert final.score == 0.8
    assert tracker.buffered_count() == 0
    assert tracker.best_of(5) is None


def test_forget_unknown_track_returns_none():
    tracker = BestShotTracker(100, 100)
    assert tracker.forget_track(999) is None


def test_multiple_tracks_are_independent():
    tracker = BestShotTracker(100, 100)
    tracker.observe(track_id=1, frame_idx=0, crop=_crop(1), score=0.5)
    tracker.observe(track_id=2, frame_idx=0, crop=_crop(2), score=0.1)

    assert tracker.buffered_count() == 2
    assert tracker.best_of(1).score == 0.5
    assert tracker.best_of(2).score == 0.1
