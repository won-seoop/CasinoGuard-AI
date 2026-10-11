# EXP-034 val_min=0 실제 파이프라인 직접 검증 + sat_min Sweep

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021), EXP-031(PAR-022), EXP-032(PAR-023), EXP-033(PAR-024)

## Goal
EXP-033 Next Action 1번(Roadmap 다음 Action 6번, 최우선): "val_min=0(그림자 하한
필터 완전 제거)을 `classify_person_attributes_with_mask_and_wb_relaxed(val_min=0)`
로 직접 검증할 것. 동시에 sat_min(채도 하한)도 같은 방식으로 낮춰보는 Sweep을
추가해 '그림자 제거 필터' 전체의 가치를 재검토할 것"을 수행한다.

## Hypothesis
1. EXP-033 섹션 5의 val_min Sweep(사후 분석, method "b" 단독 근사치)이 보인 단조
   추세(val_min을 낮출수록 Accuracy가 계속 오름)가 production이 실제로 쓰는
   Mask+WB 전체 경로(`classify_person_attributes_with_mask_and_wb_relaxed`)에서도
   그대로 유지되고, val_min=0이 Sweep 범위 내 최고일 것이다.
2. sat_min도 val_min과 동일한 메커니즘(진짜 어두운 옷이 필터에 걸려 사라짐)으로
   작동하므로, 낮출수록 단조적으로 Accuracy가 개선될 것이다.

## Problem
EXP-033은 val_min=10(대안 A)을 production(c_wb) 후보로 채택했지만, 이 결정에
쓰인 "val_min을 더 낮추면 계속 좋아진다"는 근거(사후 Sweep)는 Mask/White Balance를
빼고 `region_dominant_color`만 단독으로 돌린 근사치였다 - 실제 production이 쓰는
`classify_person_attributes_with_mask_and_wb_relaxed`(Segmentation Mask + Gray
World WB를 모두 포함)에서 같은 추세가 유지되는지는 아직 측정하지 않았다. 또한
`region_dominant_color`의 두 번째 필터 파라미터인 sat_min(채도 하한)은 지금까지
어떤 실험에서도 기본값(25)에서 건드려 본 적이 없다.

## Dataset
EXP-027/028/030/031/032/033과 완전히 동일한 coco128(실제 COCO train2017) n=21
person crop Ground Truth(EXP-032/033이 교정한 GT 라벨 오류 2건 반영) + 동일 5종
합성 조명(normal/low_light/strong_light/warm_cast/cool_cast).

## Environment
원격 자동화 세션, 컨테이너가 매 세션 새로 초기화되므로 `.venv`(Python 3.11)를
신규 생성하고 opencv-python-headless/numpy/ultralytics/lap/fastapi/pytest/httpx를
재설치했다. `github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`과
`v8.4.0`의 `yolo11n.pt`/`yolo11n-seg.pt`가 이번 세션도 egress 정책상 정상 접근
가능함을 재확인했다.

## Configuration
`src/attributes/color.py`에 `sat_min` 파라미터를 신규 노출(기존 함수 동작은
전혀 바꾸지 않음, 기본값 25로 EXP-033 당시와 동일):
- `classify_person_attributes_with_mask_relaxed(val_min=10, sat_min=25)`: 기존에는
  val_min만 노출했고 sat_min은 `region_dominant_color`의 기본값(25)으로 고정돼
  있었다. sat_min을 호출부에서 지정할 수 있도록 매개변수를 추가하고
  `region_dominant_color`/`region_dominant_color_from_pixels`로 그대로 전달한다.
- `classify_person_attributes_with_mask_and_wb_relaxed`는 이미 `**kwargs`로
  내부 함수에 전달하는 구조라 코드 변경 없이 sat_min을 받는다.

`scripts/run_exp034_val_min_zero_sat_min_sweep.py`: EXP-033과 동일한 Detector+
Segmenter IoU 매칭, 동일 GT, 동일 합성 조명으로 21 crop × 42 cell × 5 조명 =
210 cell 전수에 대해 `classify_person_attributes_with_mask_and_wb_relaxed`를
직접 호출한다(근사 없이 Mask+WB 전체 포함).
1. val_min Sweep: {35(baseline), 30, 25, 20, 15, 10(EXP-033 선택), 5, 0}을
   sat_min=25(기존 기본값) 고정 하에 측정 -> Overall/GT=black Accuracy 최고를
   `best_val_min`으로 선정.
2. sat_min Sweep: `best_val_min`을 고정한 뒤 sat_min in {25(baseline), 20, 15,
   10, 5, 0}을 측정.
Unit Test 4개 추가(`tests/test_attribute_color.py`): sat_min 파라미터가 실제로
전달되는지(기본값 유지 시 결과 불변, 0으로 낮추면 결과가 바뀜), White Balance
변형에도 그대로 전달되는지, 그리고 아래 Result 3)에서 발견한 비단조 메커니즘을
결정론적 합성 픽셀로 고정하는 회귀 테스트.

## Baseline
`classify_person_attributes_with_mask_and_wb_relaxed(val_min=35, sat_min=25)`
(EXP-029/030/031/032가 쓰는 production c_wb와 완전히 동일한 이분법 Baseline):
Overall 41.43%(87/210), GT=black 60.00%(60/100) - EXP-033의 c_wb 수치와 정확히
일치(코드 비변경 확인).
EXP-033이 채택한 대안 A(val_min=10, sat_min=25 암묵): Overall 49.05%, GT=black
75.00%, 비검정 GT(n=110) 24.55%.

## Result
`results/EXP-034/{predictions.csv, summary.json}`.

### 1) 가설 1은 확인됐다 - 실제 Mask+WB 파이프라인에서도 val_min=0까지 단조 개선이 유지된다

| val_min | Overall | GT=black | 비검정 GT(n=110) |
|---|---|---|---|
| 35(baseline) | 41.43% | 60.00% | 24.55% |
| 30 | 44.76% | 67.00% | 24.55% |
| 25 | 46.67% | 70.00% | 25.45% |
| 20 | 47.14% | 71.00% | 25.45% |
| 15 | 48.57% | 73.00% | 26.36% |
| 10(EXP-033 선택) | 49.05% | 75.00% | 25.45% |
| 5 | 49.52% | 77.00% | 24.55% |
| **0** | **50.48%(+9.05pp)** | **79.00%(+19.00pp)** | 24.55%(+0.0pp) |

EXP-033의 근사(method "b" 단독)가 예측한 대로, 실제 production 경로(Mask+WB 포함)
에서도 변곡점 없이 val_min=0까지 단조 개선되며, val_min=10(EXP-033 선택)보다도
Overall +1.43pp/GT=black +4.00pp 더 높다. 비검정 GT는 Baseline과 동일(24.55%,
부작용 없음) - EXP-033이 확인한 "val_min 완화는 비검정 GT를 희생시키지 않는다"는
결론이 val_min=0까지도 그대로 유지된다.

조건별로도 val_min=10 대비 4/5 조건이 개선되거나 동일하다(normal 54.76%→57.14%,
low_light 45.24%→47.62%, strong_light 38.10%→38.10% 동일, cool_cast 52.38%→
57.14%). warm_cast만 소폭 하락(54.76%→52.38%, -2.38pp) - EXP-033이 val_min=35
대비 발견한 유일한 역전(low_light, -2.38pp)과 같은 급의 미미한 Trade-off다.

### 2) 가설 2는 반증됐다 - sat_min은 단조가 아니라 비단조(U자형)로 작동한다

val_min=0 고정 하에 sat_min을 낮춘 결과(Overall/GT=black):

| sat_min | Overall | GT=black | 비검정 GT(n=110) |
|---|---|---|---|
| 25(baseline) | 50.48% | 79.00% | 24.55% |
| 20 | **50.95%(+0.47pp)** | 79.00%(+0.0pp) | **25.45%(+0.9pp)** |
| 15 | 50.48%(+0.0pp) | 76.00%(-3.0pp) | 27.27%(+2.7pp) |
| 10 | 50.48%(+0.0pp) | 76.00%(-3.0pp) | 27.27%(+2.7pp) |
| 5 | 49.52%(**-0.95pp, 유일한 역전**) | 74.00%(-5.0pp) | 24.55%(+0.0pp) |
| 0 | 50.95%(+0.47pp) | 77.00%(-2.0pp) | 27.27%(+2.7pp) |

sat_min=5~15 구간이 양쪽 극단(25, 0)보다 GT=black Accuracy가 낮다(76~74% vs
79%/77%) - 특히 sat_min=5는 Overall Accuracy 자체가 Baseline(sat_min=25)보다도
낮아지는 유일한 역전이다. "낮출수록 좋아질 것"이라는 가설 2는 명확히 반증됐다.

**sat_min=20이 모든 측정 지표에서 Baseline(25)을 약하지만 순수하게 지배한다**
(Overall +0.47pp, GT=black 동일, 비검정 GT +0.9pp, 조건별 역전 없음 - warm_cast만
52.38%→54.76% 개선, 나머지 4개 조건 완전히 동일) - 반면 sat_min=0은 Overall은
동률(50.95%)이지만 GT=black을 77%로 2pp 깎는다(아래 3번 원인 분석 참고).

### 3) 원인 분석: sat_min 비단조의 메커니즘을 셀 단위로 직접 추적
`results/EXP-034/predictions.csv`에서 sat_min={25,20,15,10,5,0} 중 적어도 하나가
다른 값들과 다른 정답/오답 판정을 내는 GT=black cell은 210개 중 6개뿐이었다. 그중
대표 사례 둘을 診diagnostic 함수로 직접 추적했다:

- `000000000536_0` upper(normal/warm_cast/cool_cast 공통): sat_min=25/20에서는
  "black"(정답)이었다가 sat_min=15에서 "red"로, sat_min=10 이하에서는 "gray"로
  오답이 된다 - sat_min을 낮출수록 필터를 통과하는 저채도 Noise 픽셀이 늘면서
  Hue 다수결의 median이 조금씩 오염되다가(S<30 지점에서 classify_hsv_pixel 자체의
  gray 분기로 넘어감).
- `000000000634_0` lower(warm_cast/cool_cast): sat_min=25/20/15/10에서는 "black"
  (정답)이지만 **sat_min=5에서만** "gray"(오답)로 떨어졌다가 **sat_min=0에서
  다시 "black"(정답)으로 복귀**한다. 결정론적 합성 픽셀로 재현한 메커니즘(신규
  Unit Test `test_sat_min_relaxation_is_not_monotonic_middle_value_can_be_worse_
  than_both_extremes`): 진짜 어두운 다수(S 낮음, V<45)와 중간 채도의 소수 오염이
  섞여 있을 때, 필터가 엄격하면(sat_min=25) 오염이 필터에 걸려 제외되고 min_
  filtered_fraction 폴백이 발동해 전체 픽셀 중 진짜 어두운 다수가 이기지만,
  필터를 "딱 중간만" 완화하면(sat_min=15/10/5) 오염이 필터를 통과하면서도
  진짜 어두운 다수는 여전히 통과하지 못해(S가 더 낮으므로) 오염이 다수결을
  가져간다. 필터를 완전히 버리면(sat_min=0) 진짜 어두운 다수까지 전부 포함되어
  수(count) 기준으로 다시 다수를 차지해 정답으로 복귀한다. 이 "중간 지대에서만
  걸리는 함정"이 U자형 비단조 패턴의 정확한 메커니즘이다.

## Failure Cases
새 Failure Case로 등록하지 않는다(FC-012의 연속 분석으로 유지, EXP-032/033과
동일 판단). "필터 Threshold를 낮출수록 좋아질 것"이라는 직관이 두 번째 파라미터
(sat_min)에서는 들어맞지 않고 비단조로 작동한다는 것 자체가, "한 파라미터에서
확인한 단조 추세를 다른 파라미터에도 그대로 가정하지 않는다"는 재사용 가능한
방법론적 교훈이다.

## Analysis
가설 1(val_min 단조 추세가 Mask+WB 전체 경로에서도 유지)은 확인됐다 - EXP-033의
근사(method "b" 단독)가 실제 production 경로의 동작을 정확하게 대표했다. 이는
"단순화된 근사로 먼저 추세를 확인하고, 그 추세를 실제 경로에서 재확인한다"는
접근이 이번에는 안전했다는 뜻이지만, 가설 2의 결과를 보면 이 안전함이 항상
보장되지는 않는다는 것도 함께 보여준다.

가설 2(sat_min 단조 개선)는 명확히 반증됐다. 원인은 val_min과 sat_min이 필터링
대상으로 삼는 Noise의 성격이 다르기 때문이다 - val_min(그림자 제거)이 낮추는
Threshold는 "진짜 옷(어두운 색)과 그림자 Noise"를 가르는 경계가 한쪽 방향(더 어두운
쪽)으로만 이동하지만, sat_min(채도 하한)을 낮추는 것은 "원래 필터에 걸리던 저채도
오염 픽셀"을 점진적으로 들여오는 과정에서, 어느 지점에 진짜 옷보다 수가 적지만
0은 아닌 오염 집단이 먼저 통과해 다수결을 왜곡시키는 중간 지대가 생긴다. 그 중간
지대를 지나 완전히 필터를 버리면(0) 진짜 옷의 절대 다수(count)가 다시 우세해져
U자형으로 복귀한다. 요컨대 val_min 완화는 "경계 이동"이고 sat_min 완화는 "오염
혼입 비율의 비선형적 변화"라서 서로 다른 동역학을 가진다.

## Decision
**val_min=0을 EXP-033의 val_min=10보다 우선하는 새 후보로 채택한다** - 실제
Mask+WB 파이프라인에서 Overall/GT=black Accuracy 모두 val_min=10을 앞서고(+1.43pp/
+4.00pp), 비검정 GT에 부작용이 없으며(24.55%→24.55%), 조건별로도 4/5가 개선되거나
동일하다(warm_cast만 -2.38pp의 미미한 역전).

**sat_min은 25(기존 기본값)에서 20으로만 낮춘다** - 20은 모든 측정 지표에서
Baseline을 순수하게 지배하는 유일한 값이다(Overall +0.47pp, GT=black 동일,
비검정 GT +0.9pp, 조건별 역전 없음). sat_min=0은 Overall은 동률이지만 GT=black을
2pp 깎고(위 원인 분석의 "중간 오염" 메커니즘이 sat_min=0에서는 사라지지만 다른
cell에서는 여전히 구조적으로 민감함), sat_min=5~15는 명확히 더 나쁘다 - 이
비단조성 자체가 "더 낮은 값을 무조건 선호하지 않는다"는 근거다.

**새 최종 후보: `classify_person_attributes_with_mask_and_wb_relaxed(val_min=0,
sat_min=20)`** - Overall 50.95%(Baseline 41.43%에서 +9.52pp, EXP-033의 val_min=10
선택에서 +1.90pp), GT=black 79.00%(+19.00pp/+4.00pp), 비검정 GT 25.45%(+0.90pp,
부작용 없음).

**단, 이번에도 production 기본값(method="b"/"c_wb")으로 즉시 승격하지 않는다.**
EXP-027/028/030/033이 이미 확립한 이유(n=21 작은 Dataset, 조명 변형이 synthetic,
GT 라벨 재검증이 완전히 끝나지 않음)가 그대로 적용된다. `classify_person_
attributes_with_mask_and_wb_relaxed`는 sat_min 파라미터가 추가된 상태로 코드에
남기지만(지침 38, 실패하지 않은 실험이므로 보존), `run_full_pipeline.py`/
`MetadataStore`는 바꾸지 않는다.

## Next Action
1. EXP-032/033이 계속 이월한 GT 라벨 21개 crop × 42개 영역 전체 재검증은 여전히
   이월이다 - val_min=0/sat_min=20까지 밀어붙인 개선이 "진짜 옷 색을 더 잘 잡아서"
   인지 "GT 라벨 자체의 애매한 경계 사례를 우연히 더 맞춰서"인지는 라벨 재검증
   없이는 확정할 수 없다. 이제 두 Sweep(val_min, sat_min)을 모두 거쳤으므로,
   다음 세션은 더 이상 이 필터 파라미터를 추가로 손대지 말고 반드시 이 재검증
   부터 수행해야 한다(이 항목이 이제 3번째로 이월됨).
2. Mask 매칭 실패율(coco128은 Close-up 위주라 0%)과 원본 CCTV(Wide-angle)에서의
   재검증도 계속 이월.
3. val_min/sat_min 2차원 Grid를 교차로 Sweep하면 더 나은 조합이 있을 가능성은
   있으나, 두 파라미터 모두 개별 최적점(0, 20)을 이미 찾았고 n=21 Dataset에서는
   추가 Grid Search가 과적합 위험만 키운다고 판단해 더 진행하지 않는다 - Next
   Action 1번(GT 재검증)이 선행되지 않는 한 더 정교한 Hyperparameter Tuning은
   의미가 없다.
