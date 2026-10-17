# EXP-039 Person Re-ID FC-014 - Segmentation Mask 기반 Gray World White Balance 검증

## Goal
EXP-038이 Gamma-Background+White Balance(FC-013(b) warm_cast/cool_cast 개선)를 적용한
뒤 발견한 FC-014(Occlusion Top-1이 Baseline/Candidate/Fusion 세 방법 모두에서 일관되게
악화)를 완화할 수 있는지 검증한다. Roadmap(01. Architecture & Roadmap) EXP-037 Next
Action 7-(a')가 제안한 방법("White Balance의 Gray World 평균 계산을 person Segmentation
Mask 픽셀에만 한정")을 실제로 구현해 Occlusion Top-1 저하가 줄어드는지 실측한다.

## Hypothesis
person Segmentation Mask(yolo11n-seg.pt, crop 단위 직접 추론 - EXP-028/029와 동일 방식)로
Gray World 평균을 person 픽셀에만 한정하면, Occlusion 패치(중립 회색 사각형)가 평균
계산에서 제외되어 Occlusion Top-1 저하가 완화될 것이다.

경계 가설(Roadmap 자신이 EXP-038 Next Action에 이미 적어둔 의심): Occlusion 패치는 crop
내부, 즉 사람의 실루엣 범위 안에 있으므로 Segmentation 모델이 "가려진 사람"도 하나의
person Instance로 인식해 패치 영역까지 Mask에 포함시킬 수 있다 - 이 경우 masking은 패치를
걸러내지 못하고 효과가 없다(오히려 Segmentation 추론 비용만 추가). 가설 확인/반증 중
어느 쪽이 나오든 추측 없이 코드 레벨 직접 측정으로 확정한다.

## Problem
EXP-038에서 Gamma-Background+WB가 FC-013(b)(warm_cast/cool_cast)를 사실상 해결
(88.9~100%)했지만, 그 과정에서 Occlusion Top-1이 Baseline -17.78pp/Candidate
-4.44pp/Fusion -13.33pp로 일관되게 악화됐다(FC-014). 원인은 EXP-038이 이미 코드로
확인했다: White Balance(Gray World)가 crop 전체 채널 평균을 "중립 회색" 기준으로
쓰는데, Occlusion 패치 자체가 완벽한 무채색(120,120,120)이라 crop의 20~30%를 차지하면
평균을 인위적으로 중립 쪽으로 끌어당긴다. 카지노 CCTV에서 사람 간 교차/가림은 가장
흔한 시나리오이므로, 조명 Cast를 고치려다 Occlusion 강건성을 희생하는 것은 받아들일
수 없는 Trade-off다(EXP-038 Decision).

## Dataset
- Gallery: coco128 실제 person crop 45명(`results/EXP-027/crops`, EXP-027/036/037/038과
  완전히 동일한 집합 - 실행 시 재탐지 결과로 검증됨).
- Query: EXP-036/037/038과 완전히 동일한 Protocol(RNG seed 42, 동일 호출 순서) - 조명
  4종(low_light/strong_light/warm_cast/cool_cast) + 기하 변형 3종(occlusion/bbox_jitter/
  flip) = 45명 x 7 variant = 315개 Query.
- coco128 원본 이미지는 이번 세션에서 egress 허용 경로
  (`https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`)로
  재다운로드함(EXP-027/035/038이 확인한 동일 경로, 403 없이 접근 성공).

## Environment
- 원격 자동화 세션(claude-sonnet-5). 이번 세션은 컨테이너가 완전히 새로 시작돼
  Python 패키지가 전혀 설치돼 있지 않은 상태였다(`cv2`/`torch`/`ultralytics` 전부
  ImportError) - `python3.11 -m pip install ultralytics opencv-python-headless fastapi
  uvicorn pytest httpx`로 재설치(ultralytics 8.4.178, torch 2.14.1, opencv-python-headless
  5.0.0.93, pytest 9.1.1 - EXP-038과 동일 버전대).
- `yolo11n.pt`/`yolo11n-cls.pt`/`yolo11n-seg.pt`는 ultralytics 공식 GitHub Release에서
  이번 세션에 자동 재다운로드됨(캐시 없음, 컨테이너 신규 시작).

## Configuration
- `src/attributes/color.py` 신규 추가:
  - `white_balance_gray_world_masked(crop_bgr, mask, gain_min=0.3, gain_max=3.0,
    min_mask_fraction=0.05)`: Gray World 채널 평균을 `mask==True` 픽셀만으로 계산.
    mask 유효 픽셀이 5% 미만이면 `white_balance_gray_world()`(전체 픽셀)로 폴백
    (`classify_person_attributes_with_mask_relaxed`의 폴백 설계와 동일).
  - `gamma_background_then_white_balance_masked(crop_bgr, background_bgr, mask, ...)`:
    `gamma_background_then_white_balance()`와 동일 순서(감마 먼저 -> WB 나중)이지만
    WB 단계만 masked 버전으로 교체.
- `scripts/run_exp039_reid_wb_masked_gray_world.py`: EXP-038과 동일한 배경 추출/Query
  구성 로직을 재사용하되, 두 Pipeline을 나란히 실행 - `gamma_background_plus_wb`
  (EXP-038 재현, Masking 없음) vs `gamma_background_plus_wb_masked`(신규, Mask는
  `segment_largest_instance_mask(segmenter, img_bgr)`로 **변형이 이미 적용된 이미지
  자체**(occlusion/조명/jitter/flip 반영 후)에서 직접 추론 - 실제 운영에서 BestShot
  crop 하나에 Segmentation을 돌리는 상황과 동일한 조건).
- 임베딩: `color_histogram_embedding`(Baseline), `YoloBackboneEmbedder`(Candidate,
  yolo11n-cls.pt backbone), 50:50 Score Fusion(`rank_gallery_fused`).

## Baseline
EXP-038의 `gamma_background_plus_wb`(Masking 없음)를 이 실험에서 그대로 재실행해
재현성을 확인함(Overall Top-1 Baseline 0.6857/Candidate 0.7492/Fusion 0.7778, occlusion
Top-1 Baseline 0.6222/Candidate 0.7333/Fusion 0.7556 - EXP-038 기록값과 완전히 일치).

## Result

| 조건 | Masking 없음 (EXP-038 재현) | Mask 기반 (EXP-039, 신규) | 변화 |
|---|---|---|---|
| **Baseline (Color Histogram)** |||
| Overall Top-1 | 0.6857 | 0.7079 | **+2.22pp** |
| occlusion Top-1 | 0.6222 | **0.7333** | **+11.11pp** |
| low_light Top-1 | 0.2222 | 0.2667 | +4.44pp |
| strong_light Top-1 | 0.2222 | 0.2222 | 0 |
| warm_cast Top-1 | 0.9111 | 0.9111 | 0 |
| cool_cast Top-1 | 0.8889 | 0.8667 | -2.22pp |
| bbox_jitter Top-1 | 0.9333 | 0.9556 | +2.22pp |
| flip Top-1 | 1.0000 | 1.0000 | 0 |
| **Candidate (YOLO Backbone)** |||
| Overall Top-1 | 0.7492 | 0.7492 | 0 |
| occlusion Top-1 | 0.7333 | **0.7778** | **+4.44pp** |
| low_light Top-1 | 0.3778 | 0.3556 | -2.22pp |
| strong_light Top-1 | 0.6444 | 0.6444 | 0 |
| warm_cast Top-1 | 1.0000 | 1.0000 | 0 |
| cool_cast Top-1 | 1.0000 | 1.0000 | 0 |
| bbox_jitter Top-1 | 0.6000 | 0.5778 | -2.22pp |
| flip Top-1 | 0.8889 | 0.8889 | 0 |
| **Fusion (0.5/0.5)** |||
| Overall Top-1 | 0.7778 | 0.8000 | **+2.22pp** |
| occlusion Top-1 | 0.7556 | **0.8000** | **+4.44pp** |
| warm_cast Top-1 | 0.9556 | 1.0000 | +4.44pp |
| cool_cast Top-1 | 0.8889 | 0.9333 | +4.44pp |

원본 수치: `results/EXP-039/summary.json`, 쿼리별 예측: `results/EXP-039/
per_query_predictions_{label}.csv`, 실행 중 측정한 Mask 점유율: `results/EXP-039/
mask_coverage.csv`.

### Mask가 실제로 Occlusion 패치를 제외했는지 직접 측정
실행 로그상 Mask 점유율(mask==True 비율)이 `gallery(변형 없음)=0.484` vs
`occlusion=0.333`로, Occlusion Variant에서 Mask가 평균 15.1pp 더 작았다. 이것만으로는
"Mask가 작아진 이유가 패치를 제외했기 때문"인지 확정할 수 없어(Segmentation 자체가
불확실해져 전체 Mask가 축소됐을 수도 있음), 각 Query의 정확한 Occlusion 사각형 좌표를
RNG 소비 순서를 그대로 재현해 복원하고, Mask와 "패치 영역"/"패치 밖(옷) 영역"의 겹침을
직접 계산했다(n=45, 보조 스크립트 - 세션 스크래치, 재현 가능):

| 영역 | Mask==True에 포함된 비율 (평균) |
|---|---|
| Occlusion 패치 영역 | **15.87%** |
| 패치 밖(옷이 보이는) 영역 | **37.81%** |

패치 영역의 Mask 포함률(15.87%)이 패치 밖 영역(37.81%)보다 22.0pp 낮다 - Segmentation
모델이 중립 회색 사각형을 "사람이 아닌 영역"으로 상당 부분 배제하고 있음을 코드 레벨로
직접 확인했다(두 수치 모두 100%가 아닌 것은 crop 자체가 Tight Crop이라 Segmentation이
배경/경계를 보수적으로 잘라내는 이 프로젝트의 기존 관찰(EXP-028)과 일치).

## Failure Cases
새 Failure Case는 발견되지 않았다. 다만 Masking이 경미한 Side Effect를 남겼다(사소한
수준, 모든 지표에서 상쇄됨):
- Baseline `cool_cast` Top-1 -2.22pp, Candidate `low_light`/`bbox_jitter` Top-1 각각
  -2.22pp. 모두 1개 Query(1/45 = 2.22pp 단위) 수준의 변동으로, 헤드라인 Overall Top-1이
  세 방법 모두 역전 없이 개선/유지된 것과 별개로 솔직하게 기록한다. 원인은 직접
  규명하지 않았다(n=45 각 Variant 45 Query 단위라 1건의 순위 변화가 2.22pp를 만드는
  통계적 노이즈 수준일 가능성이 있음, 추측으로 단정하지 않음).

## Analysis
**가설이 확인됐다** - Roadmap이 명시적으로 걸어둔 의심("Occlusion 패치가 Mask 내부에
있어 자동 해결 안 될 수 있음")과 반대로, Segmentation 모델은 중립 회색 사각형을 사람
실루엣의 일부로 보지 않고 상당 부분(패치 영역의 84.13%) 제외했다. 그 결과 Occlusion
Top-1이 세 방법 모두에서 일관되게 개선됐다(Baseline +11.11pp, Candidate +4.44pp,
Fusion +4.44pp) - EXP-038이 도입한 FC-014를 가장 직접적으로 겨냥한 수정이 실제로
그 문제를 완화했다.

더 중요하게, "평균 개선만 보고 끝내지 않는다"는 원칙대로 전체 Variant를 슬라이싱한
결과 FC-013(b)의 목표였던 warm_cast/cool_cast에도 새 Regression이 생기지 않았다
(Fusion은 오히려 +4.44pp씩 추가 개선, Baseline cool_cast만 -2.22pp의 경미한 변동).
세 가지 헤드라인 Overall Top-1(Baseline/Candidate/Fusion) 전부 역전 없이 개선되거나
유지됐다 - EXP-037/038이 채택 기준으로 쓴 것과 동일한 "모든 헤드라인 지표 지배"
조건을 통과한다.

이 결과는 "검증 전에는 가정하지 않는다"는 이 프로젝트의 원칙을 보여주는 사례이기도
하다 - Roadmap에 이미 적힌 합리적인 의심(Mask가 Occlusion을 포함할 것)이 실측으로
반증됐다. YOLO Segmentation이 지역적으로 "사람처럼 보이지 않는" 평평한 단색 영역을
낮은 확신도로 처리해 전체 Mask에서 자연스럽게 깎아낸 것으로 보인다(정확한 내부
메커니즘까지는 이번 실험에서 규명하지 않음 - 추측 아님을 명시).

## Decision
**`gamma_background_then_white_balance_masked`(Mask 기반 WB)를 Mask 없는
`gamma_background_then_white_balance`(EXP-038)보다 선호하는 다음 후보로 채택**한다 -
FC-014가 겨냥한 Occlusion Top-1이 세 방법 모두에서 역전 없이 개선됐고, FC-013(b)가
겨냥한 warm_cast/cool_cast에도 새 Regression이 없다(오히려 Fusion은 추가 개선).

다만 **production(`src/reid/`, `run_full_pipeline.py`)에는 여전히 연결하지 않는다** -
이유: (1) n=45 합성 Query의 근본 한계(FC-013 전체가 공유하는 제약)가 아직 해소되지
않았고, (2) 이번 Pipeline은 Segmentation 모델(yolo11n-seg.pt) 추론을 Re-ID 임베딩마다
추가로 호출해야 하므로(EXP-028 실측: Detector 단독 대비 1.29x), Re-ID가 실시간
파이프라인에 연결되기로 결정되기 전까지는 이 추가 비용을 FPS 예산에 반영할 필요가
없다.

## Next Action
1. (EXP-037 Next Action (a'') 이월) Baseline(Color Histogram)의 low_light/strong_light가
   Gamma+WB에서 추가 악화된 원인을 Gray World 게인 값을 직접 로깅해 규명할 것(이번
   실험에서도 규명하지 않았고 여전히 "추정"으로만 남아 있음).
2. Masking이 만든 경미한 Side Effect(Baseline cool_cast -2.22pp, Candidate
   low_light/bbox_jitter -2.22pp)가 통계적 노이즈인지 체계적 패턴인지, n을 늘려 재확인할
   필요가 있다(coco128 n=45는 Variant당 45 Query뿐이라 1건이 2.22pp를 만든다).
3. 원본 Wide-angle CCTV 영상 접근이 가능해지면 이번 합성 Query Protocol(EXP-036/037/038과
   동일)을 실제 같은 사람의 여러 Track으로 교체 검증(EXP-037 Next Action (b)와 동일,
   여전히 미해결).
4. Re-ID 자체를 production에 연결할지 결정되면, 이번에 채택한 Mask 기반 WB의
   Segmentation 추론 비용(Track당 1회 호출 가정 시 EXP-029의 "Track당 1회" 설계를
   그대로 재사용 가능)을 FPS/P95 Latency 관점에서 재측정할 것.
5. n=45 한계와 production 연결 여부가 모두 해소되기 전까지 MetadataStore/VMS 연결은
   보류.
