# EXP-007 Loitering Detection — Dwell Time Threshold 비교 ✅ 실행 완료

## Goal
Dwell Time > Threshold 방식의 Loitering Baseline을 구현하고, 5/10/20/30초 Threshold를 비교한다. 동시에 EXP-005에서 발견한 FC-004(Track 소실 시 EXIT 유실)를 ROIStateMachine에 수정 반영한다.

## Dataset
crowded_intersection_1080p.webm, 800 frame(약 33초, fps=23.98) — 30초 Threshold까지 테스트하기 위해 EXP-005/006보다 관찰 구간을 늘림. ROI는 EXP-005와 동일한 횡단보도 Polygon 재사용.

## Configuration
Threshold: 5초/10초/20초/30초 (frame 환산: 120/240/480/719)

## Baseline
Dwell Time(같은 ROI에 연속으로 머문 시간) > Threshold

## Result

| Threshold | Frame 환산 | Loitering 이벤트 수 |
|---|---|---|
| 5초 | 120 | 6 |
| 10초 | 240 | 1 |
| 20초 | 480 | 0 |
| 30초 | 719 | 0 |

Track 100은 5초/10초 Threshold 모두에서 이벤트가 발생(dwell 5.0초 → 10.01초로 자연스럽게 증가) — 로직이 일관되게 동작함을 확인.

## Failure Cases / Analysis
- 5초 Threshold에서는 6건이 잡히지만, 이 영상은 **횡단보도(사람이 지나가는 통로)**라서 5초 정도는 정상적으로 걸어서 통과하는 시간과 겹칠 수 있다 — 실제로는 오탐(단순 보행자를 배회로 오인)일 가능성이 있다.
- 20초/30초에서는 이벤트가 0건인데, 이는 로직이 안 도는 게 아니라 **이 장소(횡단보도) 자체가 원래 오래 머무는 곳이 아니기 때문**이다 — Loitering 기능은 카지노의 캐셔/금고 앞, 출입구처럼 "원래 오래 머물면 안 되는 구역"에 적용해야 의미가 있다는 것을 실측으로 재확인했다(지침 15 원칙과 일치).
- 단순 Dwell Time만으로는 "정상적으로 천천히 걷는 사람"과 "제자리에 서 있는 사람"을 구분하지 못한다 — 지침 15의 개선 후보(Movement Distance 추가)가 실제로 필요함을 확인.

## FC-004 개선 (EXIT 유실 수정)
`ROIStateMachine.forget_track()`이 Track이 INSIDE 상태에서 사라질 때 EXIT 이벤트를 강제로 발생시키도록 수정. Unit Test 2개 추가(`test_forget_track_emits_exit_when_was_inside`, `test_forget_track_emits_nothing_when_was_outside`)로 검증 — PAR-002 참고.

## Decision
- 이 영상(횡단보도)에는 5초 Threshold가 과민, 20초 이상이 현실적이라고 판단. 다만 카지노 캐셔/금고 앞 같은 실제 적용 구역에서는 Threshold를 다시 튜닝해야 한다.
- Movement Distance를 추가한 개선된 Loitering 로직은 Stretch로 남기고, Core MVP는 Dwell Time 단일 기준으로 확정한다.

## Next Action
Metadata Pipeline 구현 — 지금까지 만든 Track/BestShot/Event 결과를 하나의 스키마로 통합
