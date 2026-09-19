# EXP-005 Intrusion Detection — Polygon ROI + State Machine ✅ 실행 완료

## Goal
Polygon ROI에 Bottom Center 판정 + State Machine(OUTSIDE↔INSIDE)을 적용해 중복 없는 Intrusion(ENTER/EXIT) 이벤트를 만든다.

## Hypothesis
"Track이 ROI 안에 있는 모든 프레임을 이벤트로 카운트"하는 순진한(Naive) 방식은 이벤트가 폭발적으로 많아질 것이고, State Machine으로 전이 시점만 잡으면 크게 줄어들 것이다.

## Dataset
crowded_intersection_1080p.webm, 300 frame

## Environment
MacBook Air M3, YOLO11n + ByteTrack

## Configuration
- ROI: 횡단보도 영역을 감싸는 사다리꼴 Polygon (실제 카지노 "통제 구역 진입 감지" 데모로 시뮬레이션)
- 판정 기준점: Bounding Box Bottom Center (지침 13 후보 중 채택 — 사람이 서 있는 바닥 위치에 가장 가까움)
- Track 상태 정리: 10프레임 이상 미관측 시 상태 삭제 (장시간 실행 메모리 누적 방지)

## Baseline
Naive 방식: ROI 안에 있는 모든 (track, frame) 쌍을 이벤트로 카운트

## Result

| 방식 | ENTER | EXIT | 총 이벤트 |
|---|---|---|---|
| **Baseline (Naive, 중복 방지 없음)** | - | - | **630** |
| **State Machine (개선)** | 16 | 7 | **23** |

**중복 억제율: 96.3%** ((630-23)/630)

스냅샷(frame_0090.jpg)으로 육안 확인: ROI(빨간 사다리꼴) 안의 3명은 빨간 박스(INSIDE), 우측 바깥 인물은 초록 박스(OUTSIDE)로 정확히 구분됨.

## Failure Cases
- ENTER(16) > EXIT(7)로 비대칭 — Track이 ROI 안에 있는 상태에서 화면 밖으로 나가거나 Tracking이 끊기면(예: Occlusion), `forget_track()`이 상태만 지우고 EXIT 이벤트를 별도로 발생시키지 않기 때문에 "들어간 기록은 있는데 나간 기록이 없는" Track이 다수 생김. 300프레임(약 12.5초)이라는 짧은 관찰 창의 영향도 있음(아직 밖으로 안 나갔을 수도 있음).
- 이 비대칭은 실제 VMS 검색 관점에서 "이 사람이 언제 나갔는지" 질의에 답 못하는 문제로 이어질 수 있음 — 05. Failure Cases에 등록.

## Analysis
State Machine 방식은 가설대로 이벤트를 96.3% 줄였다 — 이 정도 규모의 중복이면 실제 관제 시스템에서는 알림 폭탄(Alert Fatigue)으로 이어져 관제 인력이 진짜 이벤트를 놓치게 만들 수 있다. 다만 Track 소실 시 EXIT를 유실하는 문제가 있어, "이 사람이 확실히 나갔다"를 보장하려면 추가 설계가 필요하다(예: 일정 시간 미관측 시 강제 EXIT 이벤트 발생).

## Decision
이번 Baseline은 이 상태로 확정한다. Track 소실 시 강제 EXIT 처리는 Phase 7 Loitering(Dwell Time 계산)과 함께 재설계하는 것이 효율적이라고 판단해 다음 단계로 미룬다.

## Next Action
Phase 6 Line Crossing 진행. Track 소실 시 EXIT 처리 개선은 백로그(FC-004)로 등록.
