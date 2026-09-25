# EXP-018 BestShot Position Score 개선 (FC-003 수정 검증) ✅ 실행 완료

## Goal
FC-003(EXP-004에서 발견)에서 지적된 BestShot Position Score의 "경계 근접 페널티"
문제를 실제로 수정하고, 수정 전/후를 통제된 Ground Truth로 정량 비교한다.

## Hypothesis
기존 구현(margin_ratio=2% 근접 여부)은 "경계 근접"과 "실제로 잘림(clipping)"을 혼동하고
있어, 실제로는 안 잘렸는데도 경계 근처에 있다는 이유만으로 낮은 점수(0.3)를 주는 False
Positive가 많이 발생할 것이다. bbox가 이미지 경계에 실제로 닿았는지(boundary_eps=2px,
Detector 좌표 반올림 오차만 흡수)를 보는 방식으로 바꾸면 이 False Positive가 크게
줄어들면서도, 실제로 잘린 경우를 놓치는 False Negative는 거의 늘지 않을 것이다.

## Problem
EXP-004 Failure Cases에서 발견: Track 1의 최종 선택 프레임은 Position Score 0.3점을
받았지만 육안으로는 실제로 잘리지 않은 좋은 사진이었다. 원인은
`position_score()`가 bbox와 프레임 경계 사이 거리가 margin_ratio(2%) 이내면 실제
잘림 여부와 무관하게 무조건 0.3점을 주는 이진 로직이었기 때문이다(`05. Failure Cases`
FC-003 참고).

## Dataset
원본 Wikimedia crowded 영상은 이 원격 세션에서 접근 불가(EXP-010/014~017과 동일한
network 제약, egress 정책상 commons.wikimedia.org/raw.githubusercontent.com 등
403). 대신 ultralytics 패키지 내장 실제 사진 `bus.jpg`에서 YOLO11n으로 실제 검출된
사람 1명(conf=0.878, 경계에서 충분히 떨어진 온전한 전신 crop, 192×504px)을 잘라내
"알려진 크기의 사람 패치"로 사용했다. 이 패치를 810×1080 배경(사람 없는 하늘/지붕
영역 타일링, EXP-017과 동일한 배경 생성 방식) 위 다양한 위치에 합성해 **기하학적으로
정확한 Ground Truth**(이 프레임에서 사람이 실제로 몇 px 잘렸는지)를 만들었다 — 육안
라벨링이 아니라 합성 좌표로 계산되므로 오차가 없다.

- 수평 슬라이드: y 중앙 고정, x_off를 -152→770까지 8px 간격으로 이동 (192개 프레임)
- 수직 슬라이드: x 중앙 고정, y_off를 -464→1040까지 8px 간격으로 이동 (113개 프레임)
- 각 합성 프레임에 실제 YOLO11n(conf=0.25, imgsz=640)을 다시 돌려 검출 bbox를 얻고,
  그 bbox를 legacy/new position_score에 그대로 입력했다(production과 동일 경로).

## Environment
원격 세션 컨테이너, Python 3.11, ultralytics 8.4.162(YOLO11n), OpenCV(headless), CPU.

## Configuration
- Legacy: `position_score_margin_legacy(bbox, W, H, margin_ratio=0.02)` — 기존 EXP-004 설정 그대로
- New: `position_score(bbox, W, H, boundary_eps=2.0)` — 경계에 실제로 닿은 변의 개수로
  등급화 (0개: 1.0 / 1개: 0.6 / 2개 이상: 0.2)
- Ground Truth: `gt_min_gap_px < 0` → 실제로 잘림. 분석용 버킷 3개
  (`clipped` / `near_edge_not_clipped`(0~20px, legacy margin과 겹치는 구간) / `far_from_edge`)

## Baseline
Legacy(margin 근접 기반) position_score를 Baseline으로 삼는다 — EXP-004에서 실제
사용된 방식 그대로.

## Result

총 305프레임 중 301프레임에서 YOLO가 사람을 검출(4프레임은 거의 전부 가려져 미검출,
분석에서 제외). 합성 Ground Truth 기준 151프레임이 "실제로 잘림", 150프레임이 "안 잘림"
(그중 12프레임이 legacy margin과 직접 겹치는 near-edge 구간).

| Metric | Legacy (margin 2%) | New (boundary_eps 2px) |
|---|---|---|
| Accuracy | 0.9635 (290/301) | 0.9867 (297/301) |
| Precision (flag=clipped) | 0.9317 | 0.9803 |
| Recall (flag=clipped) | 1.0 | 0.9933 |
| False Positive (n=301) | 11 | 3 |
| False Negative (n=301) | 0 | 1 |
| **near_edge_not_clipped 버킷 False Positive Rate** (n=12) | **0.917 (11/12)** | **0.25 (3/12)** |
| clipped 버킷 Recall (n=150) | 1.0 (150/150) | 0.993 (149/150) |

FC-003이 실제로 지적한 문제(안 잘렸는데 경계 근처라 낮은 점수)에 해당하는
`near_edge_not_clipped` 버킷에서 False Positive Rate가 91.7%→25.0%로 크게 감소했다.
전체 Accuracy도 96.3%→98.7%, Precision도 93.2%→98.0%로 개선됐다.

## Failure Cases
- New 방식의 잔여 False Positive 3건은 모두 `gt_min_gap_px == 0`(패치가 캔버스 경계와
  정확히 맞닿은, "잘린 것도 안 잘린 것도 아닌" 경계값) 프레임이었다 — YOLO bbox가
  boundary_eps(2px) 이내로 경계에 닿게 검출되어 "닿음"으로 판정됨. 이는 Ground Truth
  정의 자체가 애매한 경계 케이스(실제로 딱 프레임 끝에 서 있는 사람)이므로 완전한
  버그라기보다는 boundary_eps의 정의상 한계로 본다.
- New 방식의 유일한 False Negative 1건(실제 14px 잘렸는데 만점)은 YOLO 검출 bbox의
  오른쪽 경계가 진짜 경계보다 약 2.0~2.1px 안쪽에서 끝나(Detector 좌표 회귀 오차가
  boundary_eps=2px 문턱과 거의 정확히 맞물림) "안 닿음"으로 판정된 단일 프레임이었다.
  301프레임 중 1건으로 전체 결과에 미치는 영향은 작지만, boundary_eps를 너무 타이트하게
  잡으면 이런 경계 케이스가 발생할 수 있음을 한계로 기록한다.

## Analysis
가설이 실측으로 확인됐다: "경계 근접"과 "실제로 잘림"을 구분하는 것이 core 문제였고,
legacy 방식은 근접 구간(near_edge_not_clipped)에서 91.7%나 되는 극단적으로 높은 False
Positive Rate를 보였다 — 즉 "경계 근처에 서 있기만 해도 거의 항상 낮은 점수"였다는
뜻이다. New 방식은 이 비율을 4분의 1 수준으로 낮췄고, 대신 아주 미세한 경계 케이스
(gt_gap==0, boundary_eps 문턱과 거의 일치하는 1프레임)에서만 오차가 남았다 — legacy의
21.6px(1080 기준)/16.2px(810 기준) 넓은 마진과 달리, 남은 오차의 크기가 2px 안팎으로
수십 배 작아졌다.

## Decision
`position_score()`를 boundary_eps(기본 2px) 기반 "실제 경계 접촉 여부 + 닿은 변 개수"
방식으로 교체한다(`src/bestshot/scorer.py`). 기존 margin 기반 구현은
`position_score_margin_legacy()`로 이름을 바꿔 비교/회귀용으로 남기고, production
경로(`bestshot_score()`)는 새 `position_score()`를 사용하도록 이미 연결되어 있었다
(함수명만 교체하면 되도록 설계).

**검토한 대안**
- **대안 A: margin_ratio를 더 작게 튜닝**(예: 2%→0.2%) — 검토했으나 채택하지 않음.
  여전히 "거리 기반 근접" 이진 임계값이라는 같은 구조적 한계를 갖고 있고, 어떤 값을
  골라도 Ground Truth 없이는 임계값의 근거가 없다(단지 문제가 발생하는 구간의 폭을
  줄일 뿐, 개념적으로 "근접"과 "잘림"을 여전히 혼동한다). 값을 얼마나 줄여야 하는지도
  이미지 해상도마다 달라 일반화되지 않는다.
- **대안 B(채택): boundary_eps 기반 실제 접촉 판정 + 닿은 변 개수로 등급화** — 위
  Ground Truth 실험으로 근접 구간 False Positive Rate가 91.7%→25.0%로 줄어드는 것을
  직접 측정해 확인했고, boundary_eps=2px는 이미지 해상도와 무관하게 "Detector 좌표
  반올림 오차만 흡수"한다는 명확한 근거가 있어 채택했다.

## Regression Test
`tests/test_bestshot_scorer.py`에 5개 테스트 추가(기존 6개 전부 무수정 통과 — 기존
`test_position_score_penalizes_edge_touching_box`도 x1=0(경계에 정확히 닿음)이라
새 로직에서도 그대로 낮은 점수를 받아 하위 호환): near-edge-not-clipped가 만점을
받는지, 닿은 변 개수에 따라 등급이 달라지는지, boundary_eps 허용 오차, legacy와의
직접 비교(같은 bbox에 대해 legacy=0.3, new=1.0). 전체 테스트 스위트
(`pytest tests/ -q`) 100 passed, 7 skipped — 기존 기능 회귀 없음 확인.

## Next Action
Person Re-ID는 OSNet 등 pretrained weight 확보 가능성을 사전 확인 후 진행 검토.
FC-002(배경 밀집 군중 Detection 실패)는 Wikimedia 원본 영상 접근이 복구되는 세션에서
재검토.
