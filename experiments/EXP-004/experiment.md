# EXP-004 BestShot — Track별 대표 프레임 선택 ✅ 실행 완료

## Goal
Track마다 가장 좋은 대표 프레임(BestShot)을 자동으로 고르는 로직을 만들고, Random Baseline 및 Confidence-only 방식과 비교한다.

## Hypothesis
Confidence만으로 대표 프레임을 고르면(대안 A), 사람이 카메라에 너무 가까워 잘리거나 흔들린(Motion Blur) 프레임이 높은 Confidence를 받아 선택될 수 있다. Sharpness/Size/Occlusion/Position을 함께 반영한 종합 점수(대안 B)가 더 식별 가능한 대표 이미지를 고를 것이다.

## Dataset
crowded_intersection_1080p.webm, 300 frame, ByteTrack으로 추적된 Track 중 관측치 8개 이상인 32개 Track

## Environment
MacBook Air M3, YOLO11n + ByteTrack, conf=0.4, imgsz=640

## Configuration
BestShot Score = Confidence + Sharpness + Size + Occlusion + Position (5개 서브점수, 초기 가중치 모두 1.0 — 아직 튜닝 전)
- Sharpness: Laplacian 분산 (0~1 정규화)
- Size: bbox 면적 비율 (프레임의 2% 이상이면 만점)
- Occlusion: 같은 프레임 내 다른 박스와의 최대 IoU를 Proxy로 사용 (1 - max_IoU)
- Position: bbox가 프레임 경계(margin 2%)에 닿으면 0.3, 아니면 1.0

## Baseline
Random Frame 선택 (seed=42)

## Result

32개 Track에 대해 3가지 방식(Baseline/A/B)으로 대표 프레임을 선택하고 Image Grid로 저장 (`results/EXP-004/bestshot_grid/track_*.jpg`).

**정성적 비교 — Track 1** (관측치 243개, 가장 뚜렷한 차이 사례):
- Baseline(Random, frame 6): 화면 밖으로 나가는 중인 분홍 셔츠 인물 — 몸 일부만 보임
- A(Confidence-only, frame 67, conf=0.944): 매우 가까운 거리에서 촬영된 파란 셔츠 뒷모습 클로즈업 — 얼굴/특징 식별 어려움
- B(BestShot Score, frame 206, total=4.069): 표범무늬 원피스 + 가방을 든 인물 전신이 뚜렷하게 보임 — 세 방식 중 가장 식별하기 좋은 이미지

**정성적 비교 — Track 28** (관측치 113개): 세 방식 모두 비슷한 뒷모습 이미지를 선택 — 이 Track은 애초에 프레임 전체에서 화질/구도 변화가 크지 않아 세 방식의 차이가 뚜렷하지 않음.

## Failure Cases
- Track 1의 BestShot 선택 프레임(206)은 실제로는 Position Score가 0.3(경계 근접 페널티)이었는데도 다른 서브 점수(Sharpness 0.968, Size 1.0, Occlusion 0.946)가 높아 최종적으로 선택됨. 육안으로는 실제로 잘리지 않은 좋은 사진이었다 — 즉 현재 Position Score의 "경계 근접 여부만 보는 margin 방식"은 다소 거칠어서(실제 잘림 여부가 아니라 근접 여부만 봄) False Penalty를 줄 수 있음(이번 케이스는 다행히 다른 점수로 상쇄됨). → 05. Failure Cases에 등록.
- Confidence-only 방식(A)이 오히려 가장 안 좋은 사진을 고른 사례(Track 1)가 확인됨 — Confidence는 "사람이 있다는 확신도"이지 "사진 품질"과는 무관하다는 것을 실측으로 확인.

## Analysis
Confidence만으로는 BestShot을 고르기 부족하다는 가설이 Track 1에서 명확하게 확인됨(정성적). 다만 Track 28처럼 프레임 간 품질 차이가 작은 경우에는 세 방식의 차이가 거의 없어, 모든 Track에서 일관되게 우월한 것은 아니다. Position Score의 "경계 근접 페널티"는 근접 여부만 보고 실제 잘림 정도를 안 보는 한계가 있다.

## Decision
BestShot Score(가중치 동일 1.0)를 최종 대표 프레임 선택 방식으로 채택한다. Position Score는 다음 개선 대상으로 남긴다 (경계 근접이 아니라 실제 bbox가 이미지 경계에서 잘렸는지 여부로 개선 검토).

## Next Action
Phase 5 Intrusion Detection 진행. BestShot Position Score 개선은 별도 개선 실험(EXP 번호 추후 배정)으로 분리.
