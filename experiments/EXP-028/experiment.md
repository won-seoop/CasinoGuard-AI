# EXP-028 Attribute Metadata 색상 분류 - Segmentation Mask 기반 영역 분리 (대안 C)

관련 실험: EXP-025(Baseline/A/B, PAR-016), EXP-026(FC-012 WB 재검토, PAR-017),
EXP-027(coco128 n=21 재검증 + 무채색 신호 분리도 분석, PAR-018)

## Goal
EXP-027 Next Action 1번("b/b_wb 모두에 공통된 저조도 무관 실패 원인[배경 Bleed, JPEG 압축
Noise]을 줄이는 더 근본적인 영역 분리 방법[예: Segmentation Mask]을 검토할 것 - 단순 HSV
필터링은 n=21 결과로 한계가 드러났다")과 01. Architecture & Roadmap 다음 Action("FC-012 다음
후보는 Segmentation Mask 등 더 근본적인 영역 분리 방법 검토")을 수행한다. method b/b_wb는
여전히 고정 비율 사각형(side_margin_ratio)으로 배경을 "근사적으로만" 제외한다 - person
Instance Segmentation Mask로 배경을 픽셀 단위로 정확히 제외하면 Hue 다수결 오염이 줄어들어
Accuracy가 개선되는지를 EXP-027과 동일한 n=21 Ground Truth로 검증한다.

## Hypothesis
1. 고정 비율 사각형 영역은 person의 실제 윤곽과 무관하게 배경을 일정 부분 포함한다
   (팔을 벌리거나 체형이 좁은 경우 side_margin_ratio=0.12로도 부족, 반대로 체형이 넓으면
   person 일부를 잘라낼 수도 있음). Segmentation Mask로 배경을 픽셀 단위로 제외하면
   (대안 C) method b보다 Overall/GT=black Accuracy가 오를 것이다.
2. 대안 C는 "배경 Bleed"라는 원인만 해결하고 FC-012가 겨냥하는 "조명 Cast로 인한 Hue
   왜곡"은 해결하지 못할 것이다 - 즉 GT=black→blue 오분류율(cool_cast/strong_light)은
   b_wb(White Balance 적용)보다 나아지지 않을 것이다. 두 원인이 독립적이라면 Mask(대안
   C)와 White Balance(EXP-026 대안 B)를 함께 적용(c_wb)하면 각각의 이득이 합쳐질 것이다.

## Problem
EXP-025~027에서 region_dominant_color(method b/b_wb)는 그림자/하이라이트/피부색 픽셀을
HSV 임계값으로 걸러내는 방식으로 배경 오염을 "완화"했지만, 좌우 고정 비율(side_margin_
ratio=0.12)로 자른 사각형 영역 자체가 person의 실제 체형/자세와 무관하므로 배경이 그보다
넓게 섞이면(예: 팔을 벌린 자세, 좁은 체형) 여전히 오염된다. EXP-027은 이 한계를 "더 근본적인
영역 분리 방법(Segmentation Mask)"이 다음 후보라고 명시적으로 이월했다.

## Dataset
EXP-027과 완전히 동일한 coco128 n=21 Ground Truth crop + 동일 5종 합성 조명(normal/
low_light/strong_light/warm_cast/cool_cast). 재현성 확인을 위해 coco128을 동일 경로
(`github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`)로 재다운로드하고,
동일 Detector(yolo11n.pt, conf=0.4, classes=[0])로 crop을 재검출해 GT key(`000000000086_0`
등) 21개가 모두 그대로 재현됨을 확인했다(`detect_crops_and_masks()`의 `missing` 검사가
에러 없이 통과).

## Environment
원격 자동화 세션, `.venv`(Python 3.11, 신규 생성) + opencv-python-headless, numpy,
ultralytics 8.4.173, lap, fastapi, pytest. CPU 전용. 신규 의존성 모델:
`yolo11n-seg.pt`(ultralytics 공식 Release Asset, egress 제약 없이 접근 가능, 5.9MB).

## Configuration
`src/attributes/color.py`에 대안 C 추가(기존 method b/b_wb 파라미터는 변경하지 않음):
- `split_upper_lower_by_mask()`: Segmentation Mask를 상/하로 나눔(세로 비율은 method b와
  동일 head_skip_ratio=0.22/upper_end_ratio=0.55/foot_skip_ratio=0.08, 좌우 마진은 불필요
  - 마스크 자체가 배경을 이미 제외).
- `classify_person_attributes_with_mask()`: 마스크로 분리한 영역에 method b와 동일한
  region_dominant_color 필터링+Hue 다수결을 적용. `min_mask_fraction=0.05` 미만이면(세그멘
  테이션 실패) method b와 동일한 고정 비율 사각형으로 Fallback.
- `classify_person_attributes_with_mask_and_wb()`: 대안 C 앞에 EXP-026의
  `white_balance_gray_world()`를 추가(c_wb, 가설 2 검증용).

Mask 생성(`scripts/run_exp028_attribute_segmentation_mask.py`): `yolo11n-seg.pt`로 이미지
전체에서 person Instance Mask를 얻고, 같은 이미지에서 `yolo11n.pt`로 얻은 Detector bbox와
IoU(threshold=0.5)로 매칭해 crop 좌표로 자른다 - Detector와 Segmenter의 Box 순서가 다를 수
있어 이름(GT key)은 Detector 기준으로 고정하고 Segmenter 결과를 거기에 맞춰 매칭한다.

## Baseline
EXP-027이 n=21로 측정한 수치(README/Roadmap 기준): Overall Accuracy baseline 15.2% / A
15.7% / B 26.2%(normal 조명만 33.3%) / B_wb 32.4%. 이 실험은 이 네 가지 수치를 변경하지
않고(method b/b_wb 코드 미변경, 재실행으로 동일 수치 재확인) 그 위에 C/C_wb를 추가로
비교한다.

## Result
`results/EXP-028/summary.json` (n=21 crop x 5 lighting x upper+lower = 210 셀/method).
Segmentation Mask는 21/21 GT crop 모두에서 매칭됨(IoU>=0.5, Fallback 발생 0건 - "마스크
자체의 효과"만 순수하게 측정됨):

| Method | Overall Acc | Normal 조명만 | GT=black Acc | GT=black→blue(strong_light) | GT=black→blue(cool_cast) |
|---|---|---|---|---|---|
| baseline | 15.2% | 7.1% | 22.7% (25/110) | 0.0% | 72.7% |
| a | 15.7% | 7.1% | 22.7% (25/110) | 4.6% | 72.7% |
| b (production) | 26.2% | 33.3% | 39.1% (43/110) | 18.2% | 59.1% |
| b_wb (실험용) | 32.4% | 35.7% | 49.1% (54/110) | 22.7% | 22.7% |
| **c (대안 C, Mask만)** | **35.2%** | **42.9%** | **52.7% (58/110)** | 18.2% | 50.0% |
| **c_wb (대안 C+WB)** | **38.1%** | 38.1% | **55.5% (61/110)** | 27.3% | **22.7%** |

추가 비용(`measure_inference_overhead()`, coco128 20장 평균): Detector(yolo11n.pt) 71.2ms/
장, Detector+Segmenter(yolo11n-seg.pt) 91.6ms/장(오버헤드 1.29x), 모델 크기 5.61MB→6.18MB
(Segmenter 추가 +0.57MB).

## Failure Cases
- 없음(21/21 GT crop 모두 Segmentation Mask가 매칭돼 Fallback 경로가 전혀 실행되지
  않았다 - coco128처럼 person이 crop의 대부분을 차지하는 Close-up 위주 Dataset에서는
  Segmentation 실패율이 낮았다는 뜻이다. 다만 카지노 CCTV의 Wide-angle/작은 사람에서는
  이 비율이 그대로 유지되지 않을 수 있어 Decision에 한계로 기록한다).

## Analysis
**가설 1(Mask가 배경 Bleed를 줄여 Accuracy를 개선한다)은 확인됐다.** 대안 C(Mask만, WB
없음)는 Overall Accuracy(35.2%), Normal 조명(42.9%), GT=black Accuracy(52.7%) 세 지표
모두에서 기존 최고 기록이던 b_wb(각각 32.4%/35.7%/49.1%)를 넘어섰다 - **White Balance(색
보정) 없이 순수하게 "영역을 더 정확하게 잘랐다"는 것만으로 조명 보정 기법보다 더 큰
개선**을 낸 것이다. 이는 EXP-025~027이 줄곧 가정했던 "주요 실패 원인은 조명 Cast"라는
전제보다, 실제로는 "고정 비율 사각형이 배경/비-옷 픽셀을 구조적으로 포함한다"는 원인이
최소한 동등하거나 더 크게 기여했음을 시사한다.

**가설 2(Mask는 Cast 문제를 해결하지 못한다, WB와는 독립적이다)도 확인됐다.** 대안 C
단독의 GT=black→blue 오분류율은 strong_light 18.2%(= method b와 동일), cool_cast 50.0%
(b_wb의 22.7%보다 나쁨) - **Mask만으로는 FC-012의 표적 실패(Cast로 인한 Hue 왜곡)를 전혀
개선하지 못했다.** 반면 c_wb(Mask+WB)는 strong_light 27.3%(오히려 약간 악화), cool_cast
22.7%(b_wb와 동일 - WB가 cool_cast를 처리하는 능력은 그대로 유지)이면서 Overall/GT=black
Accuracy는 모든 method 중 최고치(38.1%/55.5%)를 기록했다. 즉 **두 기법은 서로 다른 원인
(배경 Bleed vs 채널 Cast)을 해결하므로 독립적으로 합산된다** - c_wb가 c와 b_wb 각각의
장점을 모두 가져간 것으로 해석된다(단, strong_light의 Cast 문제는 EXP-026에서 이미 "노출/
대비 손실이라 WB로 해결 안 됨"이라고 결론 낸 것과 일치하게, c_wb에서도 여전히 해결되지
않았다 - FC-012는 이번에도 완전히 해결되지 않는다).

**Edge Trade-off**: Segmenter 추가는 추론 시간을 1.29배(71ms→92ms/image, CPU) 늘리고
모델 용량을 0.57MB 늘린다. 카지노 환경처럼 다채널 CCTV에서 Attribute를 매 Frame이 아니라
BestShot 확정 시점(Track당 1회)에만 계산한다면(지침 12/19 설계 의도와 일치) 이 오버헤드는
실시간 파이프라인 FPS에 영향을 주지 않을 가능성이 높다 - 다만 이번 실험은 그 통합까지는
측정하지 않았다(Next Action).

## Decision
1. **대안 C+WB(c_wb)를 "현재까지 측정된 최고 성능의 실험적 방법"으로 기록한다.** 그러나
   production 기본값(`method="b"`)은 변경하지 않는다 - EXP-027이 n=21에서도 정한 production
   전환 임계치(예: normal 조명 50%+)에 c_wb(38.1%)도 여전히 못 미치고, 여전히 n=21의 단일
   Dataset·coco128(지상 스냅샷 각도)만으로 측정된 결과라는 한계가 EXP-027 Decision과
   동일하게 적용된다. **Attribute를 MetadataStore/VMS API에 연결하는 작업도 계속 보류한다.**
2. **FC-012는 이번에도 미해결로 유지한다.** c_wb가 cool_cast 오분류율을 b_wb 수준(22.7%)
   으로 유지했을 뿐 더 낮추지는 못했고, strong_light는 오히려 소폭 악화(22.7%→27.3%)됐다 -
   Segmentation Mask는 FC-012가 겨냥한 "조명 Cast" 문제 자체에는 기여하지 않는, 별개의
   개선축이라는 것이 이번 실험의 핵심 결론이다.
3. **대안 C의 코드(`classify_person_attributes_with_mask`, `classify_person_attributes_
   with_mask_and_wb`)는 src/에 유지하되 method 이름("c"/"c_wb")은 실험용으로만 노출한다**
   (EXP-026의 b_wb와 동일한 원칙 - "검증됐지만 아직 production 임계치에 못 미치는 방법은
   코드에는 남기고 기본값은 바꾸지 않는다").
4. Segmentation 실패율이 0%로 측정된 것은 coco128이 Close-up 위주이기 때문일 가능성이
   높다는 한계를 정직하게 기록한다 - 카지노 CCTV Wide-angle/작은 사람 crop에서는 Segmenter가
   사람을 아예 놓치거나 다른 사람과 Mask가 섞일 수 있다(Occlusion). 이번 실험은 그 경로
   (`min_mask_fraction` Fallback)를 Unit Test로만 검증했고 실제 Failure Case로는 관찰하지
   못했다.

## Next Action
1. 원본 영상(Wide-angle CCTV) 접근이 가능해지면 Segmentation Mask 매칭 실패율(IoU<0.5)이
   실제로 발생하는지, 그때 Fallback이 적절히 동작하는지 재검증할 것.
2. c_wb를 Track당 1회(BestShot 확정 시점)만 호출하는 방식으로 `run_full_pipeline.py`에
   연결했을 때 실제 FPS/P95 Latency 영향을 측정할 것(이번 실험은 오프라인 n=21 Batch만
   측정, 실시간 파이프라인 통합 Overhead는 미측정).
3. FC-012의 strong_light 오분류(노출/대비 손실)는 WB/Mask 둘 다로 해결되지 않음이 EXP-026/
   028에서 반복 확인됐다 - 다음 후보는 Gamma 보정 등 노출 복원 기법이거나, "강한 인공조명
   환경에서는 Attribute 신뢰도를 낮게 보고한다"는 설계 전환(EXP-027이 제안한 "판단 불가"
   개념과 결합) 검토.
