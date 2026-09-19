# EXP-002 Person Detection Baseline — YOLO11n (계획, 미실행)

## Goal
Pretrained YOLO11n의 Person Detection Baseline 성능을 측정한다.

## Hypothesis
YOLO11n(conf=0.4, imgsz=640)은 COCO 유사 일반 씬에서 합리적인 F1을 보이지만, CrowdHuman의 Occlusion 상황과 ExDark의 저조도 씬에서는 Recall이 뚜렷하게 낮아질 것이다.

## Problem
아직 없음 — Baseline 구축 전 단계.

## Dataset
- COCO person subset — sanity check용
- CrowdHuman — Occlusion / 밀집 씬 평가
- ExDark / NightOwls — 저조도 Failure Case 평가

## Environment
MacBook Air M3 (CPU/MPS), Ultralytics YOLO11n pretrained weight

## Configuration
- Baseline: conf=0.4, imgsz=640, NMS 기본값
- 실험 예정 스윕: confidence [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], image size [640, 960, 1280]

## Baseline
YOLO11n pretrained, conf=0.4, imgsz=640

## Result

**Confidence Threshold Sweep** (coco128, person class만, IoU=0.5 매칭, GT 있는 이미지 61/128장)

| Confidence | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.2 | 177 | 51 | 77 | 0.7763 | 0.6969 | 0.7344 |
| 0.3 | 154 | 24 | 100 | 0.8652 | 0.6063 | 0.7130 |
| 0.4 | 141 | 9 | 113 | 0.9400 | 0.5551 | 0.6980 |
| 0.5 | 125 | 5 | 129 | 0.9615 | 0.4921 | 0.6510 |
| 0.6 | 110 | 2 | 144 | 0.9821 | 0.4331 | 0.6011 |
| 0.7 | 97 | 1 | 157 | 0.9898 | 0.3819 | 0.5511 |

→ F1은 conf=0.2에서 최고(0.7344). 다만 이 결과는 coco128(일반 씬)만 반영한 값이며 카지노처럼 밀집/저조도인 환경에서는 다를 수 있음(아래 Analysis).

**Video FPS/Latency** (conf=0.4, warm-up 5프레임 제외, MacBook Air M3)

| Video | imgsz | FPS | P50 | P95 |
|---|---|---|---|---|
| normal_480p | 640 | 23.44 | 40.1ms | 56.7ms |
| normal_480p | 960 | 10.69 | 91.3ms | 111.7ms |
| normal_480p | 1280 | 6.06 | 151.6ms | 210.2ms |
| crowded_1080p | 640 | 27.42 | 33.9ms | 51.1ms |
| crowded_1080p | 960 | 13.70 | 69.7ms | 93.9ms |
| crowded_1080p | 1280 | 7.99 | 119.3ms | 156.9ms |

## Failure Cases
- **imgsz 1280에서 FPS 6대로 급락**: 원본 mp4의 원본 FPS(24~30)를 못 따라감 → EXP-001에서 예상했던 "실시간성 병목은 Detection 단계에서 나타날 것"이라는 가설이 실제로 확인됨.
- **1차 실행 시 모델 Warm-up 오염**: 처음 측정했을 때 normal_480p가 crowded_1080p보다 느리게 나왔는데(P95 87ms vs 46ms), 이는 normal_480p를 먼저 측정해서 모델/백엔드 초기화 비용이 첫 영상 측정에 몰린 것이 원인이었다. Warm-up 5프레임을 추가하고 재측정하니 두 영상의 순서 효과가 줄었으나(P95 87→57ms), 여전히 crowded_1080p가 근소하게 더 빠른 결과가 남음 — 정확한 원인은 확인하지 못함(추가 조사 필요, 추측으로 단정하지 않음).
- **Small Object / Occlusion / Low Light 정량 평가는 이번 실행에서 스킵**: CrowdHuman(~수GB), ExDark(~수백MB) 원본 데이터셋은 이번 세션에서 다운로드하지 않았다. 대신 실제 확보한 crowded_1080p 영상(다수 인원, 부분 Occlusion 포함)으로 정성적 확인만 진행. 정량 평가(Precision/Recall)는 다음 확장 과제로 남긴다 — 숫자를 만들어내지 않는다는 원칙(지침 31)에 따라 미실행으로 명시.

## Analysis
1. Confidence를 높일수록 Precision은 크게 오르고(0.78→0.99) Recall은 크게 떨어진다(0.70→0.38) — 전형적인 Precision/Recall Trade-off. F1 기준으로는 낮은 Confidence(0.2)가 유리하지만, 카지노처럼 오탐(False Positive)이 관제 인력 부담으로 직결되는 환경에서는 F1보다 Precision을 우선할 근거가 있을 수 있음(추후 Threshold 결정 시 재검토 대상).
2. imgsz를 640→960→1280으로 올릴수록 FPS가 23→11→6으로 거의 선형에 가깝게 감소 → **Accuracy(큰 imgsz일수록 작은 객체 검출에 유리)와 Latency의 Trade-off**가 실제로 관측됨(지침 24 Edge Optimization에서 다룰 핵심 축).
3. EXP-001의 "Video I/O는 병목이 아니다"라는 결론과 달리, Detection 단계(특히 imgsz 1280)에서는 원본 FPS를 못 따라가는 것이 실측으로 확인됨 → Phase 3 Tracking부터는 실시간성을 계속 추적 관리해야 한다.

## Decision
- Baseline Threshold는 일단 conf=0.4, imgsz=640으로 유지한다 (Precision 0.94로 오탐이 적고, FPS 23으로 원본 FPS(24~30)에 근접). imgsz 960/1280은 "Edge Optimization Phase"에서 Accuracy 개선이 필요할 때 재검토.
- CrowdHuman/ExDark 정량 평가는 지금 당장 다운로드하지 않고, Core MVP(Phase 3~7)를 먼저 완성한 뒤 시간이 남으면 Stretch로 진행한다(지침 7 Core MVP 우선 원칙).

## Next Action
Phase 3 Multi Object Tracking Baseline(ByteTrack)으로 진행
