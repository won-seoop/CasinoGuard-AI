# EXP-006 Line Crossing Detection ✅ 실행 완료

## Goal
Virtual Line(두 점)을 기준으로 이전/현재 좌표의 부호 변화를 감지해 방향 있는 Line Crossing 이벤트를 만든다.

## Dataset
crowded_intersection_1080p.webm, 300 frame

## Configuration
가상선: 횡단보도를 가로지르는 선분(도로/인도 경계 근사). 판정 기준점: Bottom Center. 부호(side) 판정: 외적(cross product) 부호.

## Result
| 방향 | 이벤트 수 |
|---|---|
| A_TO_B | 1 |
| B_TO_A | 2 |
| **합계** | **3** |

스냅샷(frame_0090.jpg)으로 가상선 위치와 Bounding Box를 육안 확인 — 선 배치는 적절했음.

## Failure Cases / Analysis
- 300프레임(약 12.5초) 동안 이벤트가 3건뿐으로 적게 나왔다. 원인: (1) 많은 Track이 이미 선의 한쪽에서 시작해(카메라에 가까운 전경 인물들이 초반부터 선 아래에 위치) "최초 관측이면 이벤트 없음" 규칙에 의해 카운트되지 않음 (2) FC-002(배경 밀집 군중 Detection 실패)로 인해 애초에 추적되지 않는 사람이 많아 Crossing 자체가 기록될 기회가 적음.
- 즉 Line Crossing 이벤트 수가 적은 것은 로직 버그가 아니라 상류 단계(Detection 커버리지)의 한계가 그대로 전이된 것으로 판단 — FC-002와 동일한 근본 원인 계열.

## Decision
Line Crossing 로직 자체는 Unit Test(7개 통과)로 정확성을 확인했으므로 Baseline으로 확정. 이벤트 수가 적은 근본 원인(Detection 커버리지)은 Edge Optimization 단계에서 함께 개선.

## Next Action
Phase 7 Loitering Detection — EXP-005에서 발견한 FC-004(Track 소실 시 EXIT 유실)를 함께 개선
