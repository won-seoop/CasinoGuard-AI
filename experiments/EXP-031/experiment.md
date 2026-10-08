# EXP-031 FC-012 "구간 단위 낮은 신뢰도 표시"로의 설계 전환 - Frame 노출/Cast 신호 검증

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021)

## Goal
01. Architecture & Roadmap 다음 Action 3번: 네 가지 pixel-level 보정(White Balance,
Segmentation Mask, Gamma-self, Gamma-background)이 모두 FC-012(검정 옷이
strong_light/cool_cast에서 blue로 오분류)를 부분적으로만 개선했고, EXP-030 Decision은
다음 후보를 "구간 단위 낮은 신뢰도 표시로의 설계 전환"으로 명시했다. 이 실험은 그
신호를 Frame 전체(crop이 아님)의 HSV 평균 밝기로 설계하고, 실제로 Attribute 정확도와
상관관계가 있는지 실측으로 검증한다.

## Hypothesis
1. Frame 전체의 평균 밝기(frame_mean_brightness)는 crop 자체 밝기와 달리 "옷
   색상(albedo)"과 섞이지 않으므로(EXP-030이 adaptive_gamma_correct를 기각한 confound가
   생기지 않는다), 이 값이 정상 범위를 벗어난 구간(low_light/strong_light)을 "낮은
   신뢰도"로 표시하면 표시하지 않은(unflagged) 구간의 Accuracy가 전체 평균보다 높아질
   것이다.
2. Threshold를 이 Dataset의 실측 분포로 정하면(Alt B, 지침 22 원칙) 임의로 고른
   고정값(Alt A)보다 더 정확하게 "진짜 비정상 노출"만 표시할 것이다.
3. 노출 신호만으로는 FC-012의 또 다른 축(warm_cast/cool_cast, 채널 Cast)을 겨냥하지
   못하므로, 채널 Cast 편차(frame_channel_cast_deviation)를 추가로 결합하면(Alt C)
   unflagged subset의 Accuracy가 더 개선될 것이다.

## Problem
EXP-026/028/030이 실측한 네 가지 pixel-level 보정은 모두 일부 조건만 개선하고
다른 조건을 악화시켰다(EXP-030 Decision 참고). 틀린 색상을 고치는 대신, "이 구간의
색상은 신뢰할 수 없다"고 표시하는 쪽으로 설계를 바꾸면 적어도 VMS 검색 결과가
사용자를 오도하지 않을 수 있다는 것이 이번 실험의 전제다. 이 전제 자체(Frame
밝기/Cast 편차가 실제로 Accuracy와 상관관계가 있는가)를 검증 없이 구현하면
EXP-027이 경계한 "검증 없는 기능을 먼저 만드는" 실수가 된다.

## Dataset
EXP-025/027/028/030과 완전히 동일한 coco128(실제 COCO train2017) n=21 person crop
Ground Truth + 동일 5종 합성 조명(normal/low_light/strong_light/warm_cast/cool_cast,
`apply_lighting()` 동일 구현). coco128을 동일 경로(`github.com/ultralytics/assets/
releases/download/v0.0.0/coco128.zip`)로 재다운로드하고 동일 Detector(yolo11n.pt,
conf=0.4, classes=[0])로 GT key 21개가 모두 재현됨을 확인했다. Frame 밝기/Cast 측정은
crop이 아니라 GT 21개가 가리키는 고유 원본 이미지 17장(여러 GT crop이 같은 이미지를
공유하는 경우 중복 제외) 전체에 대해 수행한다.

## Environment
원격 자동화 세션, 컨테이너가 매 세션 새로 초기화되므로 `.venv`(Python 3.11)를 신규
생성하고 opencv-python-headless, numpy, ultralytics, lap, fastapi, pytest를
재설치했다(EXP-026~030과 동일 구성, data/raw·models/·.venv는 지침 33에 따라 Git에
커밋되지 않으므로 매 세션 재생성이 정상). CPU 전용.

## Configuration
`src/attributes/color.py`에 순수 함수 2개 추가:
- `frame_mean_brightness(frame_bgr)`: Frame 전체 HSV V 채널 평균.
- `classify_exposure_level(mean_v, low_threshold, high_threshold)`: "low_light"/
  "strong_light"/"normal" 분류.
- `frame_channel_cast_deviation(frame_bgr)`: 채널 평균이 중립 회색에서 벗어난 정도
  (white_balance_gray_world와 같은 Gray World 가정, 보정이 아니라 측정값만 반환).

`src/attributes/confidence.py`(신규): `TrackExposureAccumulator` - DwellCounter와
동일한 패턴으로 Track이 관측되는 동안 노출 비정상 프레임 수를 누적하고,
`confidence(track_id, min_abnormal_fraction)`로 "low"/"high"를 반환한다. 이 실험
결과에 따라(아래 Result/Decision) production에는 연결하지 않고 메커니즘만 Unit
Test로 검증한다.

`scripts/run_exp031_attribute_confidence_flag.py`: 3개 Threshold 세트 비교.
- Alt A(고정 휴리스틱): V<85/V>170 (이 Dataset을 보지 않고 고른 값).
- Alt B(Dataset 기반): normal 조건 17장의 Frame 평균 밝기 P10/P90.
- Alt C(결합): Alt B(노출) OR Cast 편차 P90 초과.

## Baseline
플래그 없이 전부 신뢰(현재 production과 동일 상태): method "b" Overall Accuracy
26.19%(210셀), GT=black Accuracy 39.09%(110셀) — EXP-030과 완전히 동일 수치(코드
미변경 재확인).

## Result
`results/EXP-031/summary.json`. normal 조건 17장의 실측 분포: Frame 평균 밝기
mean=107.35 std=25.11 min=52.69 max=164.86 → Alt B Threshold(P10/P90)=80.96/130.81.
Frame Cast 편차 mean=0.1719 std=0.1382 → Alt C 추가 Threshold(P90)=0.3604.

| Threshold 세트 | Flagged Rate | Unflagged Accuracy | Flagged Accuracy | Baseline(전부 신뢰) |
|---|---|---|---|---|
| Alt A(고정 85/170) | 46.7% | **22.3%** | 30.6% | 26.2% |
| Alt B(Dataset 기반) | 65.7% | **15.3%** | 31.9% | 26.2% |
| Alt C(노출+Cast 결합) | 75.2% | **11.5%** | 31.0% | 26.2% |

조건별 Flagged Rate(Alt B): normal 28.6%, low_light 100%, strong_light 90.5%,
warm_cast 57.1%, cool_cast 52.4% — 의도한 대로 low_light/strong_light는 거의 전부
Flag됐다.

**세 Threshold 세트 모두 가설과 정반대 방향이다**: unflagged(고신뢰로 표시한 구간)의
Accuracy가 flagged(저신뢰로 표시한 구간)보다 오히려 낮다. Alt C(결합)로 Flag 범위를
넓힐수록 이 역전이 더 심해졌다(22.3%→15.3%→11.5%, flagged_rate가 커질수록 unflagged
Accuracy가 더 떨어짐).

## Failure Cases
unflagged subset의 조건별 구성(cell 수, Alt B 기준)을 직접 확인했다: normal 15,
low_light **0**, strong_light 2, warm_cast 9, cool_cast 10 (총 72 unflagged 중
low_light가 전부 Flag돼 0개 남음). EXP-030이 측정한 method "b"의 조건별 Accuracy(참고:
normal 33.3%, **low_light 40.5%**, strong_light 21.4%, warm_cast 16.7%, cool_cast
19.0%)와 겹쳐 보면 원인이 명확하다 — unflagged subset은 이 classifier가 가장 낮은
Accuracy를 보이는 세 조건(normal 33.3%, warm_cast 16.7%, cool_cast 19.0%)으로 채워져
있는 반면, 노출 신호가 가장 강하게 Flag하는 low_light는 오히려 이 classifier가 가장
**잘** 맞히는 조건(40.5%)이었다. 즉 "Frame 밝기가 정상 범위를 벗어난 구간 = 낮은
신뢰도"라는 가정 자체가, 이 특정 classifier(method b)의 실제 실패 패턴과 반대
방향이다.

## Analysis
가설 1·2(노출 신호가 confound 없이 작동하는가)는 메커니즘 자체로는 맞았다 —
frame_mean_brightness는 실제로 crop 밝기와 달리 옷 색상에 좌우되지 않는다(Unit Test로
확인). 하지만 "Frame 밝기 이탈 = Attribute 오분류 가능성 높음"이라는 더 중요한
전제가 이 classifier에는 성립하지 않았다. 원인은 method b의 오분류가 노출(밝기)
자체보다 Hue 분류 단계의 다른 약점(그림자/하이라이트 필터, 피부색 필터, Hue bin
다수결)에 더 크게 좌우되기 때문으로 보인다 — `img*0.35`(low_light)는 모든 픽셀의
V를 균일하게 낮추기만 해서 오히려 VAL_BLACK 임계값 기반 판정이 쉬워지는 조건을
우연히 만들었을 수 있는 반면, warm_cast/cool_cast(채널 비율만 바뀜, 밝기는 거의
그대로)는 Hue 자체를 왜곡시켜(EXP-026/027의 분석과 일치) 더 큰 오분류를 만든다 —
그런데 Hue 왜곡은 Frame 밝기로는 전혀 감지되지 않는다.

가설 3(Cast 편차를 추가하면 개선)도 기각됐다 - 오히려 결과가 더 나빠졌다. 원인은
frame_channel_cast_deviation 자체가 EXP-027이 이미 확인한 패턴(achromatic/chromatic
신호가 분리되지 않음)을 Frame 단위에서도 반복한 것으로 보인다: 실제 COCO 사진은
합성 조명을 가하지 않은 normal 조건에서도 이미 채널 편차가 넓게 퍼져 있어
(mean=0.172, std=0.138) cool_cast(mean=0.206)와 거의 겹친다 — 실제 사진에는 원래도
다양한 실내외 조명 색온도가 섞여 있어, "중립 회색이 아니면 Cast"라는 Gray World
가정이 normal 조건 자체에서도 깨진다.

## Decision
Frame 노출/Cast 기반 "구간 단위 신뢰도 표시" 설계는 **세 가지 Threshold 변형
모두에서 기각**한다 — unflagged subset이 flagged subset보다 항상 낮은 Accuracy를
보여, 신뢰도 표시가 오히려 거꾸로 작동한다. production(run_full_pipeline.py,
MetadataStore)에는 연결하지 않는다 — 검증되지 않은(오히려 반증된) 신호를 "신뢰도"로
노출하면 VMS 사용자가 실제로는 덜 믿어야 할 구간을 더 믿게 만드는 역효과가 난다
(EXP-027이 achromatic confidence 신호를 보류한 것과 동일한 원칙).

`frame_mean_brightness`/`classify_exposure_level`/`frame_channel_cast_deviation`
함수와 `TrackExposureAccumulator` 클래스는 실패 사례 보존 목적으로 `src/`에 남긴다
(지침 38) - 메커니즘(밝기/Cast 측정, Track별 누적 로직) 자체는 올바르게 동작하며
Unit Test로 검증됐고, 다른 목적(예: EXP-030의 배경 기반 Gamma 추정)에는 여전히
유효하게 쓰일 수 있다. 다만 "이 신호 = 낮은 신뢰도"라는 특정 가설은 반증됐으므로
그 용도로는 쓰지 않는다는 점을 docstring에 명시했다.

FC-012는 이제 pixel-level 보정 4종(White Balance, Segmentation Mask, Gamma-self,
Gamma-background) + confidence 신호 후보 2종(achromatic 채도/BGR std - EXP-027,
Frame 노출/Cast - EXP-031) 총 6가지 접근이 모두 부분효과 또는 완전 기각으로
종료됐다. 다음 후보는 이 classifier(method b, 그리고 production이 실제로 쓰는
c_wb)의 오분류가 **무엇에 의해** 발생하는지를 먼저(해결책을 설계하기 전에) 더
직접적으로 진단하는 쪽이어야 한다 - 예: 오분류된 각 셀의 Hue bin 다수결 과정 자체를
cell 단위로 들여다보는 Error Analysis, 또는 합성 조명이 아닌 실제 다양한 조명의
원본 CCTV 영상(접근 가능해지는 세션에서)으로 이 합성 Benchmark 자체의 대표성을
재검증하는 것.

## Next Action
1. FC-012는 백로그로 유지하되, "Frame 밝기/Cast 기반 confidence" 경로도 이제
   반증으로 종료됐음을 기록한다(Notion 05. Failure Cases).
2. 다음 세션은 pixel-level 보정이나 confidence 신호를 더 추가하기 전에, method
   b/c_wb의 오분류 110개 GT=black 셀을 직접 조건×Hue bin 단위로 Error Analysis해
   "무엇이 실제로 오분류를 일으키는가"(Hue 경계값, 필터 비율, 피부색 제외 범위 등)를
   먼저 진단할 것 - 지금까지 6가지 접근이 모두 "그럴듯한 가설"에서 출발해 실측으로
   반증되는 패턴이 반복되고 있다.
3. 원본 영상(실제 CCTV, 역광/스포트라이트가 실제로 분리되는 장면) 접근이 가능해지는
   세션에서 이 합성 Benchmark(coco128 n=21 + apply_lighting) 자체의 대표성도
   재검증할 것 - EXP-030 Next Action 2와 동일하게 이월.
