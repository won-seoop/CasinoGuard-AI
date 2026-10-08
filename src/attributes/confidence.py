"""Track의 Attribute 관측 구간 전체에서 Frame 노출 상태를 누적해 낮은 신뢰도 구간을
표시한다 (EXP-031, FC-012 Next Action: pixel-level 보정 네 가지[White Balance/
Segmentation Mask/Gamma-self/Gamma-background, EXP-026/028/030]가 모두 부분효과에
그쳤으므로 "구간 단위 낮은 신뢰도 표시"로 설계를 전환한다).

DwellCounter(src/events/dwell.py)와 동일한 패턴이다 - 매 프레임 Track마다 그냥
더하기만 하고, Track이 사라져도(forget_track) 호출부가 명시적으로 지울 때까지는
유지한다. 다른 점은 "ROI 안인가"가 아니라 "이 프레임이 관측됐을 때 노출 상태가
정상이었는가"를 누적한다는 것뿐이다.

**production 파이프라인(run_full_pipeline.py)에 연결하지 않음(중요)**: EXP-031에서
frame_mean_brightness 기반 노출 신호가 실제 Attribute 정확도와 상관관계가 없음이
실측으로 확인됐다(PAR-022) - 신뢰도로 표시한 구간이 오히려 더 정확했다(정반대).
검증되지 않은(오히려 반증된) 신호를 "신뢰도"라는 이름으로 production에 노출하면
사용자가 잘못된 것을 믿게 되므로(지침: 검증 없는 기능을 먼저 만들지 않는다, EXP-027과
동일 원칙), 이 클래스는 메커니즘 자체(누적 로직)만 Unit Test로 검증하고 실제 신뢰도
판단에는 쓰지 않는다. 나중에 Accuracy와 실제로 상관관계가 있는 신호를 찾으면 그
신호를 넘겨 그대로 재사용할 수 있도록 구조만 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackExposureAccumulator:
    low_threshold: float
    high_threshold: float
    abnormal_frames: dict[int, int] = field(default_factory=dict)
    total_frames: dict[int, int] = field(default_factory=dict)

    def update(self, track_id: int, exposure_level: str) -> None:
        """이번 프레임의 Frame 전체 노출 레벨(classify_exposure_level의 결과)을
        해당 track_id에 누적한다. 노출 레벨은 프레임당 1회만 계산해(모든 Track이
        같은 프레임을 공유) 호출부에서 넘겨받는다 - Track마다 같은 값을 다시
        계산하는 중복 비용을 피한다."""
        self.total_frames[track_id] = self.total_frames.get(track_id, 0) + 1
        if exposure_level != "normal":
            self.abnormal_frames[track_id] = self.abnormal_frames.get(track_id, 0) + 1

    def confidence(self, track_id: int, min_abnormal_fraction: float = 0.5) -> str:
        """이 Track이 관측된 구간 중 노출 비정상 프레임의 비율이 min_abnormal_fraction
        이상이면 "low", 아니면 "high"를 반환한다. 관측된 적이 없으면 "high"(판단 보류
        아님 - 애초에 Attribute 자체가 계산되지 않으므로 의미 없는 값)."""
        total = self.total_frames.get(track_id, 0)
        if total == 0:
            return "high"
        abnormal = self.abnormal_frames.get(track_id, 0)
        return "low" if (abnormal / total) >= min_abnormal_fraction else "high"

    def forget_track(self, track_id: int) -> None:
        """BestShot/Attribute가 확정된 뒤에는 더 이상 필요 없으므로 메모리에서 지운다
        (DwellCounter는 재방문 누적을 위해 일부러 지우지 않지만, 이 누적치는 Track의
        Attribute가 한 번 확정되면 다시 쓰이지 않으므로 PAR-004와 같은 이유로 지운다)."""
        self.abnormal_frames.pop(track_id, None)
        self.total_frames.pop(track_id, None)
