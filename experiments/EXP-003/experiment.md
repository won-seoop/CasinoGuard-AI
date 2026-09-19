# EXP-003 Multi Object Tracking Baseline — ByteTrack ✅ 실행 완료

## Goal
Ultralytics 내장 ByteTrack(`bytetrack.yaml`)을 이용해 Person Tracking Baseline을 측정한다.

## Hypothesis
혼잡한 씬(crowded_1080p)에서는 Occlusion과 사람 간 교차로 인해 한적한 씬(normal_480p)보다 Track이 자주 끊기거나(재등장) 새로운 ID가 더 많이 발급될 것이다.

## Problem
공식 GT가 있는 MOT17은 압축 파일이 5.8GB(Content-Length 확인)로 이번 세션에서는 다운로드하지 않음. 정확한 ID Switch/IDF1/HOTA는 계산할 수 없음을 솔직하게 명시.

## Dataset
직접 확보한 영상 2건(normal_480p, crowded_1080p) — 공식 GT 없음.

## Environment
MacBook Air M3, Ultralytics YOLO11n + ByteTrack(bytetrack.yaml)

## Configuration
conf=0.4, imgsz=640, classes=[person], 200 frame 처리, 30프레임마다 스냅샷 저장

## Baseline
Detection(EXP-002 동일 설정) + `model.track(persist=True, tracker="bytetrack.yaml")`

## Result

| Video | Frames | 고유 Track ID 수 | 최대 동시 Track | 평균 동시 Track | 10프레임+ 재등장 이벤트 | FPS |
|---|---|---|---|---|---|---|
| normal_480p | 200 | 11 | 5 | 3.24 | 0 | 16.87 |
| crowded_1080p | 200 | 41 | 8 | 5.31 | 7 | 24.03 |

- "재등장 이벤트"는 공식 ID Switch가 아니라, 같은 ID가 10프레임 이상 사라졌다가 다시 나타난 횟수를 세는 간단한 자체 Proxy 지표다. Track Fragmentation 후보를 스크리닝하는 용도로만 쓰고, 공식 MOT 지표로 대체하지 않는다(지침 31).

## Failure Cases

**FC-002 배경 밀집 군중에서 Detection/Tracking이 아예 발생하지 않음** (crowded_1080p, frame_0060/frame_0090 스냅샷으로 실제 확인)
- 현상: 매장 입구 근처에 30명 이상 밀집한 배경 군중은 박스가 전혀 생성되지 않음. 전경의 크고 뚜렷한 보행자(id:14, 16, 41 등)만 안정적으로 Detection·Tracking됨.
- 재현 조건: crowded_1080p.webm, imgsz=640, conf=0.4
- 원인 가설: (1) imgsz=640으로 다운스케일되면서 배경 인물의 픽셀 크기가 Detection 최소 해상도 이하로 작아짐 (2) 인물 간 심한 Occlusion으로 개별 경계가 거의 없음
- 확인 방법: frame_0060.jpg / frame_0090.jpg를 직접 육안 확인 — 배경 군중 전체가 박스 없이 통째로 비어 있음을 확인
- 검토한 대안: (A) imgsz를 960/1280으로 올려 작은 객체 인식률 개선 (B) 지금 단계에서는 손대지 않고 Phase 2 EXP-002에서 이미 확인한 imgsz-FPS Trade-off 기록만 남기고 Edge Optimization 단계에서 재검토
- 최종 처리: (B) 채택. Core MVP(Tracking → BestShot → Intrusion 등)를 먼저 완성하는 것이 우선이므로 지금 imgsz를 올리지 않는다. 다만 이 현상은 카지노 게이밍 플로어처럼 카메라에서 먼 배경까지 밀집한 상황에서 실제로 발생할 수 있는 문제이므로 05. Failure Cases에 등록해 추후 Edge Optimization/카메라 설치각도 논의에 재사용한다.

**Track ID 안정성(정성적)**: frame_0060 → frame_0090(약 1.25초 경과) 사이, 전경 인물 id:14/16/41은 ID가 유지됨. id:49/53(화면 좌측 끝)은 화면 밖으로 나가며 사라짐 — 정상 동작.

## Analysis
- crowded_1080p에서 고유 ID 수(41)가 normal_480p(11)보다 훨씬 많은 것은 실제로 사람이 더 많이 등장한 영향과, 재등장 이벤트(7건)로 인한 ID 재발급이 섞여 있어 원인을 분리하지 못함 — GT 없이는 정확히 분리 불가(정직하게 한계로 남김).
- 가장 뚜렷한 문제는 ID Switch보다 **작은 객체가 아예 검출조차 안 되는 것**이었다 — Tracking 알고리즘 이전에 Detection 단계의 한계가 Tracking 품질에 그대로 전이된다는 것을 확인.

## Decision
Phase 3은 이 상태로 Baseline으로 확정하고 다음 단계(BestShot)로 진행한다. Track 품질 개선(BoT-SORT 비교, buffer 파라미터 튜닝)은 GT 확보 여부와 함께 별도 실험으로 분리해 재검토한다.

## Next Action
Phase 4 BestShot — 지금 확보한 Track 결과(id별 bbox 시퀀스)를 이용해 대표 프레임 선택 로직 구현
