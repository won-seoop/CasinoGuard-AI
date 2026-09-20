"""BestShot 후보를 O(1) 메모리로 관리하는 Incremental Best-Tracker.

배경 (EXP-010, PAR-004): 기존 run_full_pipeline.py는 Track마다 관측된
모든 프레임의 crop 이미지를 리스트에 계속 누적한 뒤, Track이 사라질 때
한 번에 정렬해서 최고 점수를 골랐다. 이 방식은 Track이 오래 머물수록
(카지노 딜러/캐셔처럼 한 자리에 오래 있는 경우) crop 이미지가 무한정
쌓여 Memory가 선형으로 증가한다(Long Running Test에서 실측).

이 모듈은 매 프레임 즉시 점수를 계산해 "지금까지의 최고 점수 1개"만
유지하고 나머지는 즉시 버림으로써, Track 하나당 메모리 사용량을
관측 프레임 수와 무관하게 O(1)로 고정한다. 이는 실시간 스트림을
무한정 버퍼링할 수 없는 실제 Edge Camera/NVR 환경에도 더 맞는 방식이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _BestCandidate:
    frame_idx: int
    crop: np.ndarray
    score: float
    observation_count: int = 1


class BestShotTracker:
    """Track별로 지금까지 관측된 것 중 최고 BestShot Score 후보 1개만 유지한다."""

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._best: dict[int, _BestCandidate] = {}

    def observe(self, track_id: int, frame_idx: int, crop: np.ndarray, score: float) -> None:
        """이미 계산된 score로 후보를 갱신한다. 기존 최고 점수보다 높을 때만 교체.
        observation_count는 교체 여부와 무관하게 track마다 계속 누적한다
        (너무 짧게 스쳐 지나간 False Positive성 Track을 걸러내는 데 사용).
        """
        current = self._best.get(track_id)
        if current is None:
            self._best[track_id] = _BestCandidate(frame_idx=frame_idx, crop=crop, score=score)
        elif score > current.score:
            self._best[track_id] = _BestCandidate(
                frame_idx=frame_idx, crop=crop, score=score, observation_count=current.observation_count + 1
            )
        else:
            current.observation_count += 1

    def best_of(self, track_id: int) -> _BestCandidate | None:
        return self._best.get(track_id)

    def forget_track(self, track_id: int) -> _BestCandidate | None:
        """Track 소실 시 최종 후보를 반환하고 버퍼에서 제거한다 (Dict 자체도 무한정 커지지 않도록)."""
        return self._best.pop(track_id, None)

    def buffered_count(self) -> int:
        """현재 메모리에 들고 있는 관측치 개수 (Track당 최대 1개이므로 활성 Track 수와 같다)."""
        return len(self._best)
