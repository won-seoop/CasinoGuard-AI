"""DwellCounter 단위 테스트 (FC-006 재발 방지, PAR-006).

세 가지 시나리오를 Ground Truth와 비교한다:
1. ROI 안에서 관측되다가 사라짐 (Occlusion) — 연속 체류
2. ROI를 벗어난 뒤에 사라짐 — 벗어나기 전까지의 체류만 누적되어야 함
3. ROI를 벗어났다가 다시 들어옴 (재방문) — 두 방문 구간이 모두 누적되어야 함
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from events.dwell import DwellCounter  # noqa: E402

POLYGON = [(0, 0), (100, 0), (100, 100), (0, 100)]
INSIDE = (50, 50)
OUTSIDE = (500, 500)


def test_continuous_stay_then_occluded_counts_all_inside_frames():
    counter = DwellCounter(polygon=POLYGON)
    for _ in range(40):
        counter.update(1, INSIDE)
    # Track이 이후 관측되지 않아도(Occlusion) 이미 누적된 값은 유지된다.
    assert counter.get(1) == 40


def test_leaving_roi_before_disappearing_only_counts_time_inside():
    counter = DwellCounter(polygon=POLYGON)
    for _ in range(30):
        counter.update(1, INSIDE)
    for _ in range(20):
        counter.update(1, OUTSIDE)  # ROI를 벗어나 배회 -> 이 프레임들은 카운트되지 않아야 함
    assert counter.get(1) == 30


def test_revisit_accumulates_across_separate_visits():
    counter = DwellCounter(polygon=POLYGON)
    for _ in range(20):
        counter.update(1, INSIDE)  # 1차 방문
    for _ in range(15):
        counter.update(1, OUTSIDE)  # ROI 밖에서 배회
    for _ in range(20):
        counter.update(1, INSIDE)  # 2차 방문 (재입장)
    # 두 방문 구간의 합(40)이 유지되어야 한다 — 재방문 시 이전 누적치가 덮어써지면 안 됨.
    assert counter.get(1) == 40


def test_never_observed_track_returns_zero():
    counter = DwellCounter(polygon=POLYGON)
    assert counter.get(999) == 0


def test_tracks_are_independent():
    counter = DwellCounter(polygon=POLYGON)
    for _ in range(10):
        counter.update(1, INSIDE)
    for _ in range(5):
        counter.update(2, INSIDE)
    assert counter.get(1) == 10
    assert counter.get(2) == 5
