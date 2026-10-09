# EXP-032 GT=black 오분류 Error Analysis - Hue 다수결 단위 직접 진단

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021), EXP-031(PAR-022)

## Goal
EXP-031 Next Action: "pixel-level 보정이나 confidence 신호를 더 추가하기 전에, method
b/c_wb의 오분류 GT=black 110셀을 직접 조건×Hue bin 단위로 Error Analysis해 '무엇이
실제로 오분류를 일으키는가'를 먼저 진단할 것"을 수행한다. 새 보정을 설계하지 않고,
기존 `region_dominant_color`(Hue 다수결) 로직의 중간값(필터 폴백 여부, 다수결 1/2위
Hue bin의 득표 격차, 최종 HSV)을 직접 들여다본다.

## Hypothesis
1. FC-012는 "GT=black이 strong_light/cool_cast에서 blue로 오분류된다"고 프레이밍했지만,
   실제 오분류의 예측 색상을 전수 조사하면 blue 외에 orange/red/gray 등 다른 색으로도
   상당수 틀릴 것이다(FC-012의 프레이밍이 실패 양상을 과소 일반화했을 것).
2. PAR-016이 도입한 `min_filtered_fraction` 폴백(필터 통과 픽셀이 30% 미만이면 원본
   전체 픽셀로 되돌아감)이 여전히 오분류의 상당 부분을 차지할 것이다 - 폴백 자체가
   "진짜 검정을 필터가 전부 걸러내는 문제"를 막기 위해 추가됐지만, 폴백된 원본 픽셀에는
   배경/피부색이 그대로 섞여 있어 또 다른 오분류 경로가 될 수 있다.
3. 조명을 전혀 가하지 않은 normal 조건에서도 상당수가 오분류된다면, 이는 4번의 보정
   시도(WB/Mask/Gamma, 모두 "조명 문제"를 겨냥)가 애초에 해결할 수 없는 범위의 오류가
   존재한다는 뜻이다.

## Problem
FC-012를 겨냥한 pixel-level 보정 4종(White Balance/Segmentation Mask/Gamma-self/
Gamma-background, EXP-026/028/030) + confidence 신호 후보 2종(achromatic 채도·BGR std/
Frame 노출·Cast, EXP-027/031)이 전부 부분효과 또는 완전 반증으로 끝났다. 매번 "그럴듯한
가설"(조명을 보정하면 나아질 것이다, 밝기가 이상하면 신뢰도가 낮을 것이다)에서 출발해
실측으로 반증되는 패턴이 반복됐다(EXP-031 Decision이 명시적으로 지적). 이 실험은 가설을
먼저 세우고 검증하는 순서를 뒤집어, 틀린 예측 자체를 구조적으로 들여다봐서 가설을
데이터에서 끌어낸다.

## Dataset
EXP-027/028/030/031과 완전히 동일한 coco128(실제 COCO train2017) n=21 person crop
Ground Truth + 동일 5종 합성 조명(normal/low_light/strong_light/warm_cast/cool_cast).
coco128을 동일 경로(`github.com/ultralytics/assets/releases/download/v0.0.0/
coco128.zip`)로 재다운로드하고 동일 Detector(yolo11n.pt, conf=0.4)+Segmenter
(yolo11n-seg.pt, EXP-028과 동일 IoU 매칭)로 GT key 21개가 모두 재현됨을 확인했다.
GT=black 셀은 22개(upper/lower 합산) × 5 조명 조건 = 110셀/method(EXP-031이 인용한
수치와 일치).

## Environment
원격 자동화 세션, 컨테이너가 매 세션 새로 초기화되므로 `.venv`(Python 3.11)를 신규
생성하고 opencv-python-headless, numpy, ultralytics, lap, fastapi, pytest를
재설치했다(이전 세션들과 동일 구성). CPU 전용. 이번 세션은 `github.com/ultralytics/
assets/...`, `pypi.org`가 egress 정책상 정상 접근 가능함을 재확인했다(Wikimedia 등은
여전히 차단 상태로 별도 확인하지 않음 - 이 실험은 coco128만 필요해 영향 없음).

## Configuration
`src/attributes/color.py`에 순수 함수(기존 로직과 100% 동일 동작, 반환값만 dict로
확장) 추가:
- `region_dominant_color_diagnostic()`: `region_dominant_color()`의 내부 로직을 그대로
  수행하되 `color`/`used_fallback`/`filtered_fraction`/`dominant_bin`/
  `dominant_bin_count`/`runner_up_bin`/`runner_up_bin_count`/`median_h`/`median_s`/
  `median_v`를 모두 반환한다. `region_dominant_color()`는 이 함수를 감싸는 thin
  wrapper로 리팩터링했다(기존 `TestRegionDominantColor` 전체가 변경 없이 통과함으로
  동작 비변경 확인).
- `region_dominant_color_diagnostic_from_pixels()`: 세그멘테이션 마스크 픽셀 집합용.
- `classify_person_attributes_with_mask_diagnostic()` /
  `classify_person_attributes_with_mask_and_wb_diagnostic()`: production이 실제로
  쓰는 c_wb 경로에 대해 region별 `used_mask`(마스크 사용 vs 사각형 폴백)까지 포함한
  진단을 반환한다.

`scripts/run_exp032_attribute_black_error_analysis.py`: GT=black 110셀 전수에 대해
method "b"(실험용)와 method "c_wb"(production이 실제로 `run_full_pipeline.py`에
연결한 방식, PAR-020)의 진단을 수집하고, 조건별/오분류 예측색별/폴백 여부별/margin별로
집계한다.

## Baseline
EXP-031이 인용한 그대로: 플래그/진단 없이 method "b" GT=black Accuracy 39.09%(원 GT
기준, EXP-030/031과 완전히 동일한 수치로 재확인 - 코드 로직 비변경).

## Result
`results/EXP-032/{error_rows.csv, error_rows_gt_corrected.csv, summary.json, crops/}`.

### 1) FC-012의 "항상 blue" 프레이밍은 과소 일반화였다
오분류된 셀의 예측 색상 분포(원 GT 기준, method b n=67 오분류 / c_wb n=49 오분류):

| 예측 색상 | method b | method c_wb(production) |
|---|---|---|
| blue | 20 (29.9%) | 22 (44.9%) |
| orange | 14 (20.9%) | 0 |
| red | 13 (19.4%) | 8 (16.3%) |
| gray | 12 (17.9%) | 13 (26.5%) |
| white | 3 | 5 |
| green | 3 | 0 |
| yellow/pink/purple | 각 1 | 1(purple) |

method b는 blue가 전체 오분류의 29.9%에 불과하고, orange(20.9%)·red(19.4%)·
gray(17.9%)가 비슷한 비중으로 섞여 있다. production이 쓰는 c_wb는 blue 비중이
44.9%로 더 높지만 여전히 절반을 넘지 않는다 - "검정이 blue로 틀린다"는 FC-012의
프레이밍은 실제 실패의 절반도 설명하지 못한다.

### 2) 조명을 전혀 가하지 않은 normal 조건에서도 절반 가까이 틀린다
| 조건 | method b Accuracy | c_wb Accuracy |
|---|---|---|
| normal(조명 미변형) | 54.55% | 59.09% |
| low_light | 77.27% | 90.91% |
| strong_light | **9.09%** | 18.18% |
| warm_cast | 27.27% | 54.55% |
| cool_cast | 27.27% | 54.55% |

normal 조건(어떤 합성 조명 변형도 가하지 않은 원본 그대로)에서도 method b는 거의
절반(45.45%), c_wb도 40.91%가 틀린다. 즉 4번의 조명 보정 시도(WB/Mask/Gamma)가
애초에 손댈 수 없는 범위의 오류가 전체 오류 중 상당 비중을 차지한다 - strong_light가
가장 심각하지만(9.09%), normal 자체도 이미 "좋다"고 부를 수 있는 수준이 아니다.

### 3) 오분류의 약 35~40%가 `min_filtered_fraction` 폴백 경로에서 발생한다
- method b: 67건 중 27건(40.3%)이 `used_fallback=True`(필터 통과 픽셀 <30% → 원본
  전체 픽셀로 되돌아간 경우).
- method c_wb: 49건 중 17건(34.7%)이 폴백 경로.

PAR-016이 이 폴백을 도입한 이유는 "진짜 검정 옷이 S/V 필터에 전부 걸려 사라지는" 문제를
막기 위해서였다(EXP-025). 그 목적 자체는 여전히 유효하지만(필터 없이 그냥 믿으면 안
됨), 되돌아간 원본 전체 픽셀에 배경/피부색이 다시 섞여 들어가면서 그 자체가 또 다른
오분류 경로가 되고 있음을 처음으로 정량 확인했다.

### 4) 오분류는 대부분 "확신에 찬 오답"이지 "근소한 차이"가 아니다
Hue 다수결 1위/2위 bin의 득표 비율(margin)이 0.6 미만(2위가 1위의 40% 이상)인
"근소한 margin" 오분류는 method b 19.4%(13/67), c_wb 20.4%(10/49)뿐이다. 나머지
약 80%는 1위 Hue bin이 2위를 크게 앞서는, 즉 "거슬리는 소수 픽셀이 박빙으로 다수결을
흔든 것"이 아니라 "다수 픽셀 자체가 이미 검정이 아닌 다른 Hue로 보였다"는 뜻이다 -
threshold를 미세 조정하는 접근(SAT_LOW/VAL_BLACK 경계값 조정 등)으로는 이 80%를
거의 고치지 못한다.

### 5) GT 라벨 자체에 육안으로 확인되는 오류 2건을 발견해 교정 재측정했다
normal 조건 오분류 10건(method b)을 `results/EXP-032/crops/*.jpg`로 직접 육안
재확인한 결과, 2건은 classifier 오류가 아니라 **GT 라벨 오류**였다:
- `000000000086_0` upper: 주석은 "어두운 자켓"이지만 실제 사진(흑백 모터사이클 사진)
  속 남성은 밝은 크림색 셔츠를 입고 있다 - classifier 예측(white)이 맞고 GT(black)가
  틀렸다.
- `000000000634_0` upper: 그레이스케일 스케이트보더 사진에서 상의는 연회색 티셔츠,
  검정은 하의(반바지)뿐인데 상/하의 모두 black으로 라벨링됐다.

(경계 사례로 `000000000113_0` lower/`000000000572_1` upper는 원 주석에도 "네이비"라고
적혀 있지만 실내 저조도 사진에서 육안으로도 거의 검정에 가까워 **교정하지 않고
그대로 두었다** - 과대 교정 방지.)

이 2건만 고쳐 같은 파이프라인을 재측정하면:

| Metric | 원본 GT | GT 교정 후(2셀) | 변화 |
|---|---|---|---|
| method b 전체 Accuracy | 39.09% | 42.00% | +2.91pp |
| method b normal Accuracy | 54.55% | 60.00% | +5.45pp |
| c_wb 전체 Accuracy | 55.45% | 60.00% | +4.55pp |
| c_wb normal Accuracy | 59.09% | 65.00% | +5.91pp |

GT 셀 110개 중 단 2개(1.8%)를 고쳤을 뿐인데 normal 조건 Accuracy가 5.5~5.9pp
움직였다 - GT 품질이 측정치에 실질적 영향을 주지만, 교정 후에도 normal Accuracy가
60~65%에 머무는 것으로 보아 **GT 오류는 전체 문제의 일부일 뿐, 대부분(폴백 경로
35~40%, 확신에 찬 오분류 80%)은 classifier 로직 자체의 한계**임을 확인했다.

## Failure Cases
새 Failure Case로 분류하지 않고 FC-012의 하위 분석으로 기록한다(근본 원인이 FC-012와
같은 Hue 다수결 로직을 공유하기 때문). 다만 이번에 **GT 데이터셋 자체의 라벨 오류**라는
새로운 실패 유형을 처음 발견했다 - 과거 4개 실험(EXP-027/028/030/031)이 전부 이
오염된 GT로 측정됐지만, 그 실험들이 보고한 "상대적 순위"(baseline<a<b<b_wb, Mask+WB
결합이 최고 등)는 2셀 교정으로도 뒤바뀌지 않아(절대 수치만 소폭 이동) 과거 PAR의
결론 자체는 유효하다.

## Analysis
가설 1(FC-012 프레이밍 과소 일반화)과 가설 2(폴백 경로가 상당 비중)는 실측으로
확인됐다. 가설 3(normal 조건도 상당수 오분류)도 확인됐다 - 오히려 예상보다 심각해서
(45~41% 오분류), 지금까지의 모든 보정 시도(조명 특화)가 가진 "이론적 최대 효과"
자체가 작았다는 것을 시사한다. 즉 EXP-026/028/030이 strong_light/cool_cast에서만
부분 개선을 보이고 Overall Accuracy는 거의 못 올린 이유가 단순히 "보정이 미흡해서"가
아니라 "애초에 고칠 수 있는 오류의 비중이 작아서"였을 가능성이 높다.

GT 라벨 오류 발견은 이 프로젝트의 측정 방법론에 대한 메타적 교훈이다 - 지침 31은 GT를
"지어내지 말라"고 하지만, 한 번 만든 GT를 재검증 없이 4번 재사용하는 과정에서 생긴
오류(사람의 실수)는 지어낸 것과는 다른 문제이고, 작은 n(21개 crop)일수록 각 라벨
오류가 측정치에 미치는 영향(1셀=0.9pp)이 커서 더 위험하다.

## Decision
**방향 전환을 확정한다**: pixel-level 보정이나 confidence 신호를 더 추가하지 않는다.
네 가지 보정(WB/Mask/Gamma-self/Gamma-background)과 두 가지 confidence 신호
(achromatic 채도/BGR std, Frame 노출/Cast)가 전부 부분효과 또는 반증으로 끝난 이유가
이번 실험으로 설명된다 - 오분류의 다수(normal 조건 40%+, 폴백 경로 35~40%, 확신에
찬 오분류 80%)가애초에 "조명을 보정하면 고쳐지는" 범위 밖에 있었다.

다음 세션이 시도해야 할 것은 새 보정이 아니라 **filter 폴백 경로 자체의 재설계**다 -
`min_filtered_fraction` 폴백이 오분류의 35~40%를 차지한다는 것은, "필터를 버리고
원본으로 돌아간다"는 이분법적 설계(PAR-016)보다 더 정교한 전략(예: 필터 기준을
완화해 더 많은 진짜 옷 픽셀을 통과시키거나, 폴백 시에도 배경/피부색 제외를 유지하는
부분 폴백)이 필요할 수 있다는 구체적 단서다. 또한 GT 라벨 재검증(21개 crop 전체를
상/하 42개 영역 단위로 재확인)도 다음 세션 후보로 남긴다 - 이번에 2건만 확인했지만
전체를 보지 못했다.

이번 실험 자체는 production 코드(`run_full_pipeline.py`, `MetadataStore`)를 바꾸지
않는다 - 진단 함수만 `src/attributes/color.py`에 추가했고(기존 함수는 thin wrapper로
리팩터링, 동작 비변경을 Regression Test로 고정), 기존 공개 API(`region_dominant_color`,
`classify_person_attributes_with_mask*`)의 동작은 전혀 바뀌지 않았다.

## Next Action
1. `min_filtered_fraction` 폴백 경로 재설계 - 이분법적 폴백(필터 통과 <30% -> 전부
   포기) 대신 완화된 필터 기준 또는 부분 폴백 전략을 Alt A/B로 비교할 것(이번 실험이
   오분류의 35~40%를 이 경로로 특정했으므로, 다음으로 손댈 가장 근거 있는 지점이다).
2. GT 라벨 21개 crop × 42개 영역(상/하) 전체를 재검증해 추가 라벨 오류가 있는지
   확인할 것(이번에는 normal 조건 오분류 10개만 샘플로 봤다).
3. 원본 영상(실제 CCTV) 접근이 가능해지는 세션에서 이 synthetic Benchmark(coco128 +
   apply_lighting) 자체의 대표성도 재검증할 것(EXP-030/031 Next Action과 동일,
   이월 지속).
