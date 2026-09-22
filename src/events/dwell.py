"""ROI 누적 체류 시간(Cumulative Dwell Time) 집계.

FC-006에서 발견된 문제: `LoiteringDetector.enter_frame`은 "지금 이 순간까지의
연속 체류 스트릭"만 표현하도록 설계되어 있다. Loitering 이벤트를 "일정 시간
이상 연속으로 ROI에 머무르면 발생"으로 판정하려는 목적에는 이 정의가 맞고,
그래서 ROI를 벗어나는 순간 `update()`가 즉시 `enter_frame`을 리셋한다.

하지만 VMS Search(지침 17)가 원하는 질문은 "이 사람이 이 구역에 총 얼마나
있었는가"이고, 사람이 구역을 잠깐 벗어났다가 다시 들어오는 경우까지 포함해야
한다. Loitering 판정용 연속 스트릭 상태를 그대로 재사용하면 ROI를 한 번이라도
벗어나는 순간 이전 체류 기록이 사라지므로, 책임을 분리한 별도의 누적 카운터가
필요하다.

DwellCounter는 Track이 ROI 안에서 관측된 프레임 수를 마주칠 때마다 그냥
더하기만 한다. ROI를 벗어나도 리셋하지 않고, Track이 사라져도 pop하지
않는다(같은 Track ID가 나중에 다시 등장해도 이전 누적치가 사라지지 않도록).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .roi_state_machine import point_in_polygon


@dataclass
class DwellCounter:
    polygon: list[tuple[float, float]]
    counts: dict[int, int] = field(default_factory=dict)

    def update(self, track_id: int, point: tuple[float, float]) -> None:
        """이번 프레임에 point가 ROI 안이면 해당 track_id의 누적 카운트를 1 늘린다."""
        if point_in_polygon(point, self.polygon):
            self.counts[track_id] = self.counts.get(track_id, 0) + 1

    def get(self, track_id: int) -> int:
        """지금까지 누적된 ROI 내 체류 프레임 수. 관측된 적 없으면 0."""
        return self.counts.get(track_id, 0)
