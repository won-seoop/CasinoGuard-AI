# EXP-026 FC-012 재현 + White Balance / Achromatic Shortcut 대안 비교

## Goal
EXP-025(PAR-016)가 남긴 FC-012 — Attribute 색상 분류(method "b")에서 검정 옷이
`strong_light`/`cool_cast` 합성 조명 조건에서 반복적으로 `blue`로 오분류되는 문제 —
의 원인을 더 구체적으로 규명하고, 두 가지 대안으로 실제 개선 여부를 측정한다.

## Hypothesis
> HSV의 S(채도)는 R/G/B의 "상대적" 차이이므로, 어두운 픽셀(V가 작은 검정 옷)일수록
> 조명 Cast가 만든 작은 절대 채널 차이도 S를 크게 부풀린다. Hue 분류 이전에 (A) 무채색
> 여부를 raw BGR 채널 spread로 먼저 판정하거나 (B) Gray World White Balance로 채널
> Cast 자체를 줄이면, FC-012가 겨냥하는 "검정 GT 셀이 blue로 분류되는" 비율이 줄어들
> 것이다.

## Problem
`results/EXP-025/predictions.csv`의 `cool_cast`/`strong_light` 행을 보면 GT가
`black`인 상/하의 셀 중 다수가 `blue`로 분류된다(예: `zidane_0 cool_cast upper/lower`,
`bus_0 strong_light upper/lower`). 01. Architecture & Roadmap과 PAR-016에 FC-012로
등록되어 있으나 원인은 "카메라가 조명 색을 반영하는 물리적 한계"로만 추정되어 있었고
실제 HSV 수치를 근거로 검증되지는 않았다.

## Dataset
EXP-025가 이미 생성해 Git에 저장한 `results/EXP-025/lighting_variants/*.jpg`
(ultralytics 번들 `zidane.jpg`/`bus.jpg`에서 YOLO11n(conf=0.3)으로 뽑은 crop 5개 ×
조명 변형 5종: normal/low_light/strong_light/warm_cast/cool_cast)와 동일한 수동
Ground Truth(EXP-025와 동일)를 그대로 재사용했다. 새로 YOLO를 실행하지 않아 이번
세션의 네트워크 제약과 무관하게 완전히 재현 가능하다(지침 30 Regression Test 취지:
동일 Dataset으로 Before/After 비교).

## Environment
원격 자동화 세션(egress: pypi/GitHub Release Assets만 허용). 로컬 CPU, Python
3.11 + OpenCV-headless, 네트워크 미사용(이미 저장된 이미지 파일만 읽음).

## Configuration
- `src/attributes/color.py`에 `white_balance_gray_world(gain_min=0.3, gain_max=3.0)`
  신규 추가.
- `classify_person_attributes()`에 실험용 `method="b_wb"` 추가(Gray World WB →
  기존 method "b" 파이프라인). production 기본값인 `method="b"`는 변경하지 않았다
  (Decision 참고).
- 대안 A는 production 코드에 추가하지 않고 `scripts/run_exp026_fc012_white_balance.py`
  안에 `region_dominant_color_achromatic_shortcut(spread_thresh=12)`로만 구현
  (threshold는 8/12/16 grid search에서 모두 동률 최고치였고, 그중 가장 보수적인 값을
  선택했다 - Analysis 참고).

## Baseline
EXP-025의 production 방식: `classify_person_attributes(method="b")`
(영역 분할 + 그림자/하이라이트/피부색 필터 + Hue 다수결, White Balance 없음).
저장된 variant 이미지를 다시 읽어 측정했기 때문에(JPEG 재압축 영향) EXP-025의
원본 숫자(0.40)와 완전히 동일하지는 않다(0.44) — 이번 실험 내부의 Baseline/대안 A/
대안 B 세 방식은 모두 동일하게 저장된 이미지를 읽으므로 셋 사이의 비교는 유효하다.

## Result
`results/EXP-026/summary.json`, `results/EXP-026/predictions.csv`.

전체 50셀(5 crop × 5 조명 × 상/하) 기준:

| Metric | Baseline | 대안 A (BGR spread shortcut) | 대안 B (Gray World WB) |
|---|---|---|---|
| 전체 Accuracy | 0.44 | 0.42 | **0.48** |
| 조명 안정성(non-normal 평균) | 0.425 | 0.40 | **0.475** |

FC-012가 실제로 겨냥하는 "GT가 black인 셀"(35개: crop 7개 × 조명 5종)만 따로 슬라이싱:

| Metric | Baseline | 대안 A | 대안 B |
|---|---|---|---|
| 검정 GT 셀 Accuracy | 0.4286 (15/35) | 0.40 (14/35) | **0.4286 (15/35, 동일)** |
| 검정 GT 셀 → blue 오분류율 | 0.3143 | 0.3143 | 0.4571 |

검정 GT 셀 Accuracy를 조명별로 쪼개면:

| 조건 | Baseline Acc | 대안 B Acc | Baseline blue율 | 대안 B blue율 |
|---|---|---|---|---|
| normal | 0.5714 | 0.4286 | 0.14 | 0.43 |
| low_light | 1.0 | 0.8571 | 0.00 | 0.00 |
| strong_light | 0.0 | 0.0 | 0.43 | **1.00** |
| warm_cast | 0.4286 | 0.4286 | 0.14 | 0.43 |
| cool_cast | **0.1429** | **0.4286** | 0.86 | 0.43 |

### pytest
`tests/test_attribute_color.py`에 White Balance 책임(Cast 축소, gain clip, 빈 영역
처리)과 `method="b_wb"` 유효성 + `method="b"` 비영향(회귀 방지) 테스트 6개 신규. 전체
회귀 185 passed(기존 179 + 신규 6) / 7 skipped, 회귀 없음.

## Failure Cases
- **대안 B(Gray World WB)는 검정 GT 셀 자체의 Accuracy를 전혀 개선하지 못했다**
  (15/35 → 15/35, 완전히 동일). `cool_cast`에서는 크게 좋아졌지만(0.14→0.43) `normal`
  에서 오히려 나빠졌고(0.57→0.43), `strong_light`에서는 Accuracy는 그대로(0.0)인
  채로 오분류가 더 획일적으로 `blue`에 쏠렸다(0.43→1.00). 즉 조건별로 오분류 패턴만
  재배치됐을 뿐, "검정 옷이 틀리는 빈도" 자체는 줄지 않았다.
- 전체 평균 Accuracy가 오른 이유(0.44→0.48)는 검정 셀이 아니라 **진짜 파란색/흰색
  옷(bus_1/bus_2의 blue 하의)이 더 안정적으로 분류됐기 때문**이다(非검정 15셀 기준
  7/15→9/15). 이는 이 실험이 원래 겨냥한 문제와 무관한 부수 효과다.
- 대안 A(BGR spread shortcut)는 모든 threshold(8~50)에서 Baseline보다 낫거나
  같은 적이 없었고(최고 0.42 < 0.44), 검정 GT 셀 Accuracy도 오히려 떨어졌다
  (0.4286→0.40, bus_0 low_light lower가 새로 틀림). raw BGR spread는 조명 밝기 자체에
  의존하므로(밝을수록 절대 채널 차이도 커짐) "Cast로 생긴 무채색 오염"과 "원래
  채도가 있는 옷"을 안정적으로 구분하는 신호가 되지 못했다.

## Analysis
가설은 부분적으로 틀렸다. Gray World White Balance는 채널 간 "상대적" Cast를 줄이는
데는 실제로 효과가 있었다(`cool_cast`에서 검정 셀 Accuracy 0.14→0.43 확인). 그러나
FC-012가 기술하는 현상("검정 옷이 blue가 됨")은 `cool_cast` 하나의 조건에서만
주로 관찰됐을 뿐, `strong_light`(과노출/대비 감소)에서는 애초에 채널 간 "비율"
문제가 아니라 전체적인 대비 손실이 원인이라 WB가 손을 댈 수 있는 영역이 아니었다 —
오히려 WB가 가벼운 Cast를 "증폭"시켜 이전에는 `green`/`gray`로 흩어져 있던 오분류를
`blue`로 더 획일화시켰다(strong_light blue율 0.43→1.00). `normal`(Cast가 전혀 없는
조건)에서도 WB가 crop 내부의 비대칭적 명암(얼굴/배경 잔여물 등)을 실제 색이 있는
것처럼 과대 해석해 Accuracy를 끌어내렸다(0.57→0.43). 결과적으로 "검정 GT 35셀"
전체로 보면 정확히 상쇄되어 Accuracy가 변하지 않았다(15/35 고정) — **평균(전체
Accuracy)만 보면 개선처럼 보이지만, 이 실험이 실제로 겨냥한 실패 모집단(검정 GT
셀)으로 Metric을 슬라이싱하지 않았다면 "FC-012를 고쳤다"고 잘못 결론 내릴 뻔했다**
(지침 36: 의미 있는 Metric 선택, 지침 38: 숫자로 성과를 과장하지 않는다).

대안 A(raw BGR spread 기반 무채색 Shortcut)는 가설의 또 다른 축("S 대신 절대 채널
차이를 보면 될 것")이었으나, 측정 결과 절대 채널 차이도 조명 밝기 자체에 선형적으로
비례해 커지므로(Cast가 없어도 밝은 영역은 spread가 커진다 - 원래 정상 조명 데이터에서도
spread 20~50대가 흔함) 무채색/유채색을 가르는 안정적인 임계값이 존재하지 않았다.

## Decision
두 대안 모두 production 기본값(`classify_person_attributes(method="b")`)으로
승격하지 않는다. `white_balance_gray_world()`와 `method="b_wb"`는 코드에 남겨
Unit Test로 동작을 고정하되(향후 더 큰 Dataset에서 재평가할 수 있도록), 호출하지
않으면 기존 파이프라인에 영향이 없다. FC-012는 "미해결"로 유지하되 원인 설명을
구체화한다: 단일 White Balance나 단순 채도/Spread Threshold로는 해결되지 않고,
① `cool_cast`류(채널 비율 Cast)와 ② `strong_light`류(대비/노출 손실)는 서로 다른
원인이라 서로 다른 처치가 필요하다는 것이 이번 실험의 핵심 결론이다. Roadmap이
후보로 적어뒀던 "무채색 확신도 점수 + 낮은 확신도는 unknown 처리"가 다음에 시도할
만한, 아직 검증되지 않은 방향으로 남는다(바로 적용하지 않는 이유: 이번 실험에서
"무채색 여부 판정" 자체가 신뢰할 수 있는 신호를 찾지 못했으므로, confidence 점수를
설계하기 전에 더 크고 다양한 n으로 그 신호 자체를 먼저 검증해야 한다).

## Next Action
1. FC-012는 `cool_cast`/`strong_light`를 더 이상 하나의 원인으로 묶지 않는다 —
   Failure Case 기록을 두 하위 유형으로 분리해 다음에 재검토할 때 혼동하지 않도록
   한다.
2. 원본 영상 접근이 가능한 세션에서 더 크고 색상 다양성이 높은 실제 crop을 확보하면,
   "무채색 신호"(S, raw spread, 또는 다른 후보)가 Dataset 크기와 무관하게 안정적인지
   먼저 검증한다 — 이번 n=5(그중 검정 GT는 crop 3개뿐)로는 Threshold 하나를
   확정하기에 근본적으로 부족하다(지침 36, 38).
3. Attribute를 MetadataStore/VMS API에 연결하는 작업은 계속 보류한다(EXP-025
   Decision 재확인 — 이번 실험도 그 결론을 바꿀 만한 근거를 만들지 못했다).
