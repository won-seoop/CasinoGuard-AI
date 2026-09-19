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
(미실행 — Phase 2 구현 후 채움)

## Failure Cases
환경별 확인 예정: Small Object, Occlusion(CrowdHuman), Crowded Scene, Low Light/Backlight(ExDark)

## Analysis
(미실행)

## Decision
(미실행)

## Next Action
Phase 2 Person Detection 코드 구현 → 본 실험 실제 실행 → Threshold/Image Size 스윕 진행
