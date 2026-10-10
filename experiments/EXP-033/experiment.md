# EXP-033 min_filtered_fraction 폴백 경로 재설계 - 대안 A(val_min 완화) vs 대안 B(단계적 폴백)

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021), EXP-031(PAR-022), EXP-032(PAR-023)

## Goal
EXP-032 Next Action 1번: "min_filtered_fraction 폴백 경로 재설계 - 이분법적 폴백
(필터 통과 <30% -> 전부 포기) 대신 완화된 필터 기준 또는 부분 폴백 전략을 Alt A/B로
비교할 것"을 수행한다.

## Hypothesis
1. (대안 B가 이길 것) EXP-032가 오분류의 35~40%를 `min_filtered_fraction` 폴백
   경로(필터 통과 <30% -> skin exclusion까지 포함해 필터를 완전히 버림)로 특정했으므로,
   그 경로만 정교하게 고치는 대안 B(val_min을 35->15->0으로 단계적으로만 완화해
   skin exclusion은 항상 유지)가 전체를 거칠게 건드리는 대안 A(val_min을 그냥
   35->10으로 낮춤)보다 부작용(비검정 GT 오염) 없이 더 낫거나 같을 것이다.
2. 둘 다 적용해도 fallback이 전혀 발동하지 않는 cell의 Accuracy는 변하지 않을
   것이다(대안 B는 설계상 그 cell을 건드리지 않고, 대안 A도 "fallback 경로 자체"를
   겨냥한 수정이므로).

## Problem
EXP-032가 Hue 다수결 중간값을 직접 진단해 찾은 구체적 단서: `region_dominant_color`의
`min_filtered_fraction`(기본 0.3) 폴백이 필터 통과 픽셀 비율이 30% 미만이면 필터를
완전히 버리고 원본 전체 픽셀(배경/피부색 포함)로 되돌아간다. 이 폴백이 GT=black
오분류의 35~40%(method b 67건 중 27건, c_wb 49건 중 17건)를 차지한다. PAR-016이 이
폴백을 도입한 이유(진짜 검정 옷이 S/V 필터에 전부 걸려 사라지는 문제 방지)는 여전히
유효하지만, "필터를 버린다"는 이분법 자체가 또 다른 오분류 경로가 되고 있었다.

## Dataset
EXP-027/028/030/031/032와 완전히 동일한 coco128(실제 COCO train2017) n=21 person
crop Ground Truth + 동일 5종 합성 조명(normal/low_light/strong_light/warm_cast/
cool_cast). EXP-032가 직접 확인한 GT 라벨 오류 2건(`000000000086_0` upper:
black->white, `000000000634_0` upper: black->gray)을 이번 실험의 기준 GT에 처음부터
반영했다(과거 EXP-027/028/030/031 기록 수치는 당시 GT 기준으로 건드리지 않는다,
지침 38).

## Environment
원격 자동화 세션, 컨테이너가 매 세션 새로 초기화되므로 `.venv`(Python 3.11)를 신규
생성하고 opencv-python-headless/numpy/ultralytics/lap/fastapi/pytest/httpx를
재설치했다. `github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`과
`v8.4.0`의 `yolo11n.pt`/`yolo11n-seg.pt`가 이번 세션도 egress 정책상 정상 접근
가능함을 재확인했다(Wikimedia 등은 재확인하지 않음 - 이 실험은 coco128만 필요해
영향 없음).

## Configuration
`src/attributes/color.py`에 신규 추가(기존 공개 함수는 전혀 바꾸지 않음):
- `region_dominant_color_graded_fallback_diagnostic()`/`_graded_fallback()`/
  `_graded_fallback_from_pixels()`: 대안 B. `val_min_steps=(35, 15, 0)`을 순서대로
  시도해 처음으로 `min_filtered_fraction`을 넘기는 단계를 선택한다. `sat_min`·
  `val_max`·skin exclusion은 모든 단계에서 그대로 유지된다. 모든 단계가 기준을
  못 넘기면 가장 완화된 단계(val_min=0)의 결과를 쓰고, 그마저도 0픽셀이면(순수
  무채색 영역, S 자체가 sat_min 미달) 기존 함수와 동일하게 완전 원본으로 복귀한다.
- `classify_person_attributes(method="b_relaxed")`: 대안 A. method b와 영역 분리는
  동일, `region_dominant_color`의 `val_min`만 35->10으로 낮춘 가장 단순한 수정
  (폴백 설계 자체는 이분법 그대로 - "진짜 fallback 경로"가 아니라 1차 필터
  자체를 완화한다는 점에서 대안 B와 설계 철학이 다르다).
- `classify_person_attributes(method="b_graded")`: 대안 B를 공개 API에 연결.
- `classify_person_attributes_with_mask_relaxed`/`_and_wb_relaxed`,
  `classify_person_attributes_with_mask_graded`/`_and_wb_graded`: production이
  실제로 쓰는 Mask+WB 경로(c_wb, PAR-020)에 대안 A/B를 각각 연결한 변형.
- 단위 테스트 23개 추가(`tests/test_attribute_color.py`) - 특히
  `test_rescues_dark_garment_from_skin_contamination_that_flips_binary_fallback`는
  어두운 옷이 여러 Hue로 흩어지고 피부색이 30% 섞이면 이분법 폴백이 "orange"로
  오분류하지만 단계적 폴백은 "black"을 보존한다는 것을 결정론적 합성 픽셀로 고정한
  회귀 테스트다.

`scripts/run_exp033_fallback_redesign.py`: method b 계열(b/b_relaxed/b_graded)과
c_wb 계열(c_wb/c_wb_relaxed/c_wb_graded) 6가지를 21 crop × 42 cell × 5 조명 =
210 cell 전수에 대해 측정한다. EXP-027 방식의 Overall Accuracy(GT 색상 전체)와
EXP-026/032 방식의 GT=black Accuracy(FC-012가 겨냥한 부분집합)를 함께 낸다.

## Baseline
method b(binary fallback, val_min=35): Overall 27.62%(58/210), GT=black
42.00%(42/100) - EXP-032의 GT-corrected 수치와 정확히 일치(코드 비변경 확인).
method c_wb(production, PAR-020): Overall 41.43%(87/210), GT=black
60.00%(60/100) - 역시 EXP-032 GT-corrected 수치와 일치.

## Result
`results/EXP-033/{predictions.csv, summary.json}`.

### 1) 가설 1은 반증됐다 - 대안 B(더 정교한 설계)가 대안 A(더 단순한 수정)보다 전부 나빴다

| Method | Overall Accuracy | GT=black Accuracy |
|---|---|---|
| b (baseline) | 27.62% | 42.00% |
| b_relaxed (대안 A) | **36.19%** (+8.57pp) | **59.00%** (+17.00pp) |
| b_graded (대안 B) | 27.62% (+0.00pp) | 46.00% (+4.00pp) |
| c_wb (production, baseline) | 41.43% | 60.00% |
| c_wb_relaxed (대안 A) | **49.05%** (+7.62pp) | **75.00%** (+15.00pp) |
| c_wb_graded (대안 B) | 36.67% (**-4.76pp, 역전**) | 59.00% (-1.00pp) |

대안 B(production c_wb_graded)는 Overall Accuracy가 오히려 Baseline보다 낮아졌다 -
"더 정교하게 폴백만 고친다"는 설계가 실제로는 더 거친 수정(val_min을 그냥 낮추는
것)보다 못했다.

### 2) 조건별로도 대안 A는 거의 전부 개선, 대안 B는 거의 전부 악화(production 기준)

| 조건 | c_wb | c_wb_relaxed | c_wb_graded |
|---|---|---|---|
| normal | 42.86% | **54.76%** | 40.48% |
| low_light | 47.62% | 45.24%(-2.38pp, 유일한 소폭 역전) | 45.24% |
| strong_light | 35.71% | **38.10%** | 23.81% |
| warm_cast | 40.48% | **54.76%** | 35.71% |
| cool_cast | 40.48% | **52.38%** | 38.10% |

대안 A는 low_light에서만 2.38pp의 미미한 역전이 있고 나머지 4개 조건 전부 개선된다.
대안 B는 low_light를 포함해 5개 조건 전부 Baseline보다 낮다.

### 3) 원인 분석: 대안 A의 이득은 "폴백 경로"가 아니라 "1차 필터 자체"에서 나왔다
`b_used_fallback`(method b가 binary fallback을 발동했는지)으로 슬라이싱:

| 구간 | n | b_acc | b_relaxed_acc | b_graded_acc |
|---|---|---|---|---|
| fallback 미발동 (used_fallback=False) | 118 | 10.17% | **22.03%(+11.9pp)** | 10.17%(+0.0pp) |
| fallback 발동 (used_fallback=True) | 92 | 50.00% | 54.35%(+4.4pp) | 50.00%(+0.0pp) |
| [GT=black] fallback 미발동 | 45 | 13.33% | **44.44%(+31.1pp)** | 13.33%(+0.0pp) |
| [GT=black] fallback 발동 | 55 | 65.45% | 70.91%(+5.5pp) | **72.73%(+7.3pp)** |

대안 B(b_graded)는 설계 그대로 fallback 미발동 cell(전체의 56%)을 전혀 건드리지
않는다(개선 0.0pp, 당연함 - 로직상 손대지 않는 구간이다). GT=black만 놓고 fallback
발동 구간 안에서는 대안 B가 대안 A보다 오히려 근소하게 더 낫다(72.73% > 70.91%) -
EXP-032가 겨냥한 "폴백 경로 안의 GT=black 오분류"라는 narrow한 목표는 대안 B가
실제로 달성했다.

그런데 대안 A의 개선은 그 narrow한 목표 바깥, 즉 **fallback이 전혀 발동하지 않은
cell**(45개 GT=black cell, 전체 GT=black의 45%)에서 압도적으로 크게(+31.1pp) 나왔다
- EXP-032는 이 구간을 "이미 잘 작동하는 cell"로 분류해 분석 대상에서 제외했지만,
실제로는 이 구간의 Baseline Accuracy 자체가 13.33%로 매우 낮았다. val_min=35는
fallback을 피할 만큼은(≥30% 통과) 통과시키면서도, 통과한 그 30%대 픽셀 집합
자체가 실제 옷 색을 대표하지 못하는 경우가 많았다 - "폴백이 발동했는가"와 "필터링된
결과가 옷을 대표하는가"는 별개의 문제였다.

### 4) 비검정 GT에 대한 부작용 확인
대안 A가 GT=black을 과도하게 많이 예측해 비검정 GT를 희생시키는 것은 아닌지 확인:

| Method | 비검정 GT(n=110) Accuracy |
|---|---|
| b | 14.55% |
| b_relaxed | 15.45%(+0.9pp) |
| b_graded | 10.91%(-3.6pp) |
| c_wb | 24.55% |
| c_wb_relaxed | 25.45%(+0.9pp) |
| c_wb_graded | 16.36%(**-8.2pp**) |

대안 A는 비검정 GT도 소폭 개선(또는 최소 악화 없음)한다. 대안 B는 오히려 비검정
GT를 크게 악화시킨다(c_wb 기준 -8.2pp) - GT=black-in-fallback에서 얻은 좁은 이득
(+7.3pp)이 비검정 cell에서 잃은 손실(-8.2pp)로 상쇄되고도 남아, 전체 Accuracy가
역전되는 정확한 메커니즘이다.

### 5) val_min 추가 Sweep (사후 분석, production 결정에는 사용하지 않음)
대안 A/B를 결정하기 전에 미리 고른 val_min=10(대안 A)이 우연히 좋은 값이었는지
확인하기 위해 method b만으로 val_min={35,30,25,20,15,10,5,0}을 사후에 스윕했다:

| val_min | Overall | GT=black |
|---|---|---|
| 35(baseline) | 27.62% | 42.00% |
| 30 | 30.00% | 47.00% |
| 25 | 30.95% | 49.00% |
| 20 | 33.33% | 53.00% |
| 15 | 34.29% | 55.00% |
| 10(대안 A) | 36.19% | 59.00% |
| 5 | 38.10% | 63.00% |
| 0 | 38.57% | 64.00% |

val_min을 낮출수록 단조롭게(변곡점 없이) Accuracy가 계속 오른다 - val_min=0(그림자
하한 필터를 완전히 제거)이 스윕 범위 내 최고다. 이는 "그림자 제거용 val_min 하한"
이라는 설계 전제 자체가 이 Dataset에서는 순 이득이 아니라 순 손실에 가깝다는 뜻이다
(제거되는 진짜 그림자 Noise보다 함께 제거되는 진짜 검정 옷 픽셀이 더 많다). val_min=10
은 실험 전에 미리 고른 값(결과를 보고 고른 값이 아님)이라 Decision에서 대안 A/B
비교의 근거로는 그대로 쓰지만, 다음 세션은 이 단조 추세의 극단(val_min=0, 하한
필터 완전 제거)을 직접 다음 후보로 검증해야 한다(Next Action 참고).

## Failure Cases
새 Failure Case로 등록하지 않는다(FC-012의 연속 분석으로 유지, EXP-032와 동일
판단). 다만 "오분류를 Error Analysis로 정확히 진단했어도(EXP-032), 그 진단이
가리키는 narrow한 수정(대안 B)이 더 거친 수정(대안 A)보다 못할 수 있다"는 패턴은
이 프로젝트의 다른 실험(EXP-030/031이 가설을 반증한 패턴)과 같은 계열의 교훈이다.

## Analysis
가설 1(대안 B가 이길 것)은 명확히 반증됐다. 원인은 설계 범위의 불일치다 - EXP-032의
Error Analysis는 "폴백이 발동한 cell에서 무엇이 틀렸는가"만 봤고, 그 분석은
정확했다(대안 B가 그 narrow 구간에서는 실제로 대안 A를 이긴다, 72.73% > 70.91%).
하지만 Accuracy 손실의 더 큰 원천은 애초에 "폴백이 발동하지 않은" cell에 숨어
있었다 - val_min=35가 fallback을 피할 만큼은 통과율을 넘기면서도, 그 통과한
픽셀 집합의 대표성 자체가 나빴다. 대안 B는 설계상 이 구간을 전혀 건드리지 않으므로
(fallback 미발동 cell에서 개선 0.0pp, 측정으로 확인) 이 손실을 회복할 수 없었고,
narrow한 이득이 비검정 GT에 대한 부작용(-8.2pp)으로 상쇄되면서 전체 Accuracy가
Baseline보다 낮아졌다.

가설 2(fallback 미발동 cell은 두 대안 모두 영향 없을 것)는 대안 B에 대해서는
맞았지만(+0.0pp, 설계대로), 대안 A에 대해서는 틀렸다 - 애초에 대안 A는 "폴백
경로"가 아니라 1차 필터 자체를 바꾸므로 모든 cell에 영향을 준다. 이 가설이 틀린
방식 자체가 이번 실험의 핵심 발견이다: "오분류가 집중된 경로를 찾아 그 경로만
고친다"는 직관적으로 안전해 보이는 전략이, 문제의 진짜 범위(1차 필터의 Threshold
자체)보다 좁게 겨냥했기 때문에 실패했다.

## Decision
**대안 A(val_min 완화)를 대안 B(단계적 폴백)보다 선택한다** - Overall/GT=black
Accuracy 모두에서 대안 A가 대안 B를 크게 앞서고(production 기준 Overall +12.4pp,
GT=black +16.0pp), 비검정 GT에 부작용도 없다(오히려 소폭 개선). 대안 B는 narrow한
설계 목표는 달성했지만 더 큰 손실 구간을 건드리지 못해 전체적으로는 Baseline보다
나쁘다 - production에 연결해서는 안 된다.

**단, 대안 A도 production 기본값(method="b"/"c_wb")으로 즉시 승격하지 않는다.**
EXP-027/028/030이 이미 확립한 것과 동일한 이유(n=21 작은 Dataset, 조명 변형이
synthetic, GT 라벨 재검증이 완전히 끝나지 않음)가 이번에도 그대로 적용된다. 추가로
이번 실험 자체가 val_min=0까지 단조 개선이 이어짐을 보여 "val_min=10이 최적"이라고
확정할 근거도 아직 없다 - 다음 세션에서 val_min=0(하한 필터 완전 제거)을 직접
검증해야 production 후보로 확정할 수 있다. `classify_person_attributes_with_mask_
and_wb_relaxed`는 이번 세션까지 측정된 가장 강력한 FC-012 개선 후보로 코드에
남기지만(지침 38, 실패하지 않은 실험이므로 보존), `run_full_pipeline.py`/
`MetadataStore`는 바꾸지 않는다.

## Next Action
1. val_min=0(그림자 하한 필터 완전 제거)을 `classify_person_attributes_with_mask_
   and_wb_relaxed(val_min=0)`로 직접 검증할 것 - 이번 실험의 사후 Sweep이 단조
   추세의 끝(val_min=0)이 스윕 범위 내 최고임을 보였으므로, 이것이 다음으로 손댈
   가장 근거 있는 지점이다. 동시에 sat_min(채도 하한)도 같은 방식으로 낮춰보는
   스윕을 추가해 "그림자 제거 필터" 전체의 가치를 재검토할 것(val_min만으로는
   아직 sat_min의 기여를 분리하지 못했다).
2. EXP-032 Next Action 2(GT 라벨 21개 crop × 42개 영역 전체 재검증)는 여전히
   이월이다 - val_min=0까지 밀어붙이는 것이 Accuracy를 올리는 것이 "진짜 옷 색을
   더 잘 잡아서"인지 "GT 라벨 자체의 애매한 경계 사례(네이비/블랙 경계 등)를
   우연히 더 맞춰서"인지는 라벨 재검증 없이는 확정할 수 없다.
3. Mask 매칭 실패율(coco128은 Close-up 위주라 0%, EXP-028 Next Action과 동일
   이월)과 원본 CCTV(Wide-angle)에서의 재검증도 계속 이월.
