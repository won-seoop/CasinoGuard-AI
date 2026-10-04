# EXP-027 Attribute Metadata 색상 분류 coco128 기반 n 확장 재검증 + 무채색 신호 분리도 분석

관련 실험: EXP-025(Attribute Metadata 색상 분류 Baseline/A/B, PAR-016), EXP-026(FC-012 재검토,
Gray World WB vs Achromatic Shortcut, PAR-017)

## Goal
01. Architecture & Roadmap 다음 Action에 반복적으로 남아 있던 "EXP-025 Attribute 색상 분류를
더 크고 색상 다양성 높은 실제 인물 crop으로 재검증(현재 n=5, 검정 편중 한계)"과 "FC-012 다음
후보(무채색 확신도 점수)를 설계하기 전에 더 큰 n으로 '무채색 신호' 자체가 안정적인지 먼저
검증"을 함께 수행한다.

## Hypothesis
1. n=5(zidane.jpg/bus.jpg 2장에서 나온 crop)로 측정한 EXP-025/026의 Accuracy는 Dataset이
   작고 편중되어 있어 실제보다 낙관적일 것이다 — 더 크고 다양한 crop에서는 method 간 순위
   (baseline < a < b < b_wb)는 유지되더라도 절대 Accuracy는 낮아질 것이다.
2. GT가 achromatic(black/white/gray)인 영역과 chromatic인 영역은 raw HSV 채도(S) 또는
   raw BGR 채널 표준편차로 어느 정도 분리될 것이다 — 그렇다면 Roadmap이 제안한 "낮은 확신도는
   unknown으로 표시" 설계가 성립할 수 있다.

## Problem
EXP-025의 Ground Truth는 ultralytics 패키지에 번들된 이미지 2장(zidane.jpg, bus.jpg)에서 나온
person crop 5개뿐이었다. 이 중 4개가 GT upper=black이었다(검정 옷 편중). n=5는 통계적으로
방법 간 우열을 신뢰성 있게 가를 수 없고, 당시 Decision에서도 "n=5 작은 Dataset·낮은 색상
다양성이라는 한계를 정직하게 기록"하며 이월했다. 또한 EXP-026은 FC-012(검정 옷이
strong_light/cool_cast에서 blue로 오분류)의 다음 대안으로 "무채색 확신도 점수"를 제안했지만,
그 신호(채도 S 등)가 실제로 achromatic/chromatic을 분리하는지는 전혀 검증되지 않은 상태였다.

## Dataset
- **coco128** (실제 COCO train2017 128장, 7MB). 이 원격 세션에서 Wikimedia·motchallenge.net은
  이번에도 403(egress 정책)으로 차단됨을 재확인했지만, `https://github.com/ultralytics/assets/
  releases/download/v0.0.0/coco128.zip`(ultralytics 패키지의 `coco128.yaml`에 명시된 공식
  다운로드 경로)은 200으로 정상 접근 가능함을 새로 확인했다 — 이전 세션들이 시도했던
  `v8.3.0/coco128.zip`(404)과 달리 실제 태그는 `v0.0.0`이었다.
- YOLO11n(conf=0.4, person class)으로 128장 전체에서 person을 검출하고 bbox 면적
  >=120x250px인 crop 45개를 얻었다(`results/EXP-027/crops/`).
- 이 중 상/하의가 모두 가려지지 않고 단일 색으로 판단 가능한 crop **21개**를 Read 도구로 직접
  확대해 보며 수동 라벨링했다(지침 31, `scripts/run_exp027_attribute_coco128.py`의
  `GROUND_TRUTH`). 꽃무늬/타이다이 등 다색 패턴이거나(예: `000000000036_0` 원피스,
  `000000000474_0` 타이다이 셔츠) 하반신이 테이블/의자에 가려 보이지 않는 crop(예:
  `000000000110_1`, `000000000241_0`)은 "정확하게 측정할 수 없는 Metric은 억지로 작성하지
  않는다" 원칙에 따라 제외했다 — 즉 45개 중 24개를 버린 것 자체가 실제 CCTV crop의 상당수가
  Attribute 라벨링에 부적합함을 보여주는 관찰이기도 하다.
- n=5 대비 색상 다양성이 늘었다(black 9쌍 외에 white/gray/red/pink/orange/blue 포함).
- EXP-025와 동일한 5가지 합성 조명(normal/low_light/strong_light/warm_cast/cool_cast)을
  동일 Config로 재적용해 Before/After가 유효하게 비교되도록 했다.

## Environment
- 원격 자동화 세션, `.venv`(Python 3.11, 신규 생성) + opencv-python-headless, numpy, ultralytics,
  lap, fastapi, pytest. CPU 전용.

## Configuration
변경 없음. `src/attributes/color.py`의 4가지 method(baseline/a/b/b_wb) 기본 파라미터를
그대로 사용(EXP-025/026과 동일 `head_skip_ratio=0.22, upper_end_ratio=0.55,
foot_skip_ratio=0.08, side_margin_ratio=0.12`, `region_dominant_color` 기본값).

## Baseline
EXP-025/026이 n=5에서 측정한 수치를 Baseline으로 삼는다(README 기준): Overall Accuracy
baseline 32% / A 28% / B 40%, B_wb 전체 평균 48%(EXP-026).

## Result

`results/EXP-027/summary.json` (n=21 crop x 5 lighting x upper+lower = 210 셀/method):

| Method | Overall Acc (n=21, 전체 조명) | Normal 조명만 | GT=black Acc | GT=black→blue 오분류 |
|---|---|---|---|---|
| baseline | 15.2% | 7.1% | 22.7% (25/110) | 16건 |
| a | 15.7% | 7.1% | 22.7% (25/110) | 17건 |
| b (production) | 26.2% | 33.3% | 39.1% (43/110) | 20건 |
| b_wb (실험용) | 32.4% | 35.7% | 49.1% (54/110) | 23건 |

조명 조건별 Accuracy(method b_wb 기준): normal 35.7%, low_light 40.5%, strong_light 23.8%,
warm_cast 31.0%, cool_cast 31.0%.

무채색 신호 분리도 분석(`results/EXP-027/achromatic_signal.csv`, 조명 보정 없는 원본 crop
기준, 상/하 영역 42개 = GT achromatic 34개 + chromatic 8개):

| 신호 | achromatic 평균 | achromatic 최댓값 | chromatic 평균 | chromatic 최솟값 | 분리됨? |
|---|---|---|---|---|---|
| HSV 채도 S | 64.46 | 158.89 | 121.29 | 62.92 | **아니오** (겹침) |
| raw BGR 채널 표준편차 | 55.26 | 89.92 | 56.54 | 36.74 | **아니오** (거의 동일 평균) |

## Failure Cases
- **coco128 person crop의 45개 중 24개(53%)가 상/하의 색상 판단 불가**: 다색 패턴(꽃무늬,
  타이다이), 좌식/프레임 밖으로 잘린 하반신, 배경 인물과의 Occlusion. 실제 CCTV Wide-angle
  영상에서도 유사한 비율로 "Attribute 판단 불가" crop이 나올 수 있음을 시사한다 — Attribute
  Metadata를 VMS에 노출할 때 "판단 불가"를 명시적으로 표현하는 설계가 필요하다는 근거가 됐다
  (아래 Decision 참고, 기존 설계에는 이 개념이 없음).
- **Baseline/A의 normal 조명 Accuracy가 7.1%로 n=5 때(32%/28%)보다 훨씬 낮음**: coco128 crop은
  zidane.jpg/bus.jpg보다 배경이 풍부한 실제 스냅샷이라, bbox 전체 평균(Baseline) 또는 단순
  상하 분할 평균(A)이 배경·잔디·가구 색으로 더 쉽게 끌려간다. 예: `000000000086_0`(검정 자켓
  남성)의 baseline 예측이 upper=gray, `000000000165_1`(검정 정장)은 upper=orange로 나왔다 —
  배경/피부/조명이 섞인 평균이 실제 옷 색과 무관한 값으로 수렴한 사례.
- **방법 B/B_wb도 GT=black 셀의 약 50~61%를 여전히 맞추지 못함**: n=5에서는 보이지 않던
  규모의 실패다. 오분류는 blue(가장 흔함)뿐 아니라 gray, orange 등으로도 분산되어, FC-012가
  설명하는 "조명 Cast → blue 오분류" 하나만으로는 전체 실패를 설명하지 못함을 확인했다(복합
  원인: 조명 Cast + 배경 Bleed + crop 해상도/압축 Noise).

## Analysis
**가설 1(절대 Accuracy 하락)은 확인됐다.** 모든 method에서 n=21 Accuracy가 n=5보다 낮다
(baseline 32%→15%, A 28%→16%, B 40%→26% [normal 조명 기준이면 40%→33%], B_wb 48%→32%).
그러나 **method 간 상대적 순위는 그대로 유지됐다**(baseline≈A < B < B_wb). 즉 EXP-025/026의
"B가 Baseline보다 낫다"는 핵심 결론 자체는 n 확장 후에도 뒤집히지 않았지만, 실제 운영
환경에서 기대할 수 있는 절대 품질 수준은 이전 기록(32~48%)보다 훨씬 낮게(15~32%) 다시
보정해야 한다. 이는 지침 31("정확하게 측정할 수 없는 Metric은 억지로 작성하지 않는다")과
지침 38(실패를 숨기지 않는다) 원칙에 따라 정직하게 기록한다.

**가설 2(무채색 신호 분리)는 기각됐다.** HSV 채도(S)와 raw BGR 채널 표준편차 모두 achromatic
영역과 chromatic 영역 사이에 뚜렷한 분리 구간이 없다 — 오히려 가장 채도가 높게 측정된 영역
(S=158.89)이 GT=black(`000000000572_0`, 어두운 니트)이었고, 가장 채도가 낮은 chromatic
영역(S=62.92)은 GT=pink(`000000000623_0`, 분홍 바지)였다. 원인은 이 영역들이 "옷 색 하나"가
아니라 배경 Bleed·하이라이트·그림자·JPEG 압축 Noise가 섞인 실제 픽셀 집합이기 때문이다 —
`region_dominant_color()`가 이미 S/V 극단값과 피부색을 걸러내려 시도하지만(EXP-025), 그 필터를
통과한 "대표 영역"조차 색상 간 신호가 깔끔히 갈리지 않는다는 뜻이다.

## Decision
1. **n=5 Dataset로 측정한 EXP-025/026의 수치는 "작은 Dataset에서의 상대 비교"로만 유효하고,
   절대 Accuracy로 재인용하지 않는다.** README/Notion에 이 n=21 재검증 결과를 함께 명시한다.
2. **method b(production 기본값)는 그대로 유지한다.** n이 늘어도 baseline/A보다 일관되게
   우수했고(normal 조명 33% vs 7%), b_wb보다는 낮지만(33% vs 36%) EXP-026에서 b_wb가
   low_light에 새로운 regression을 만든다는 이유로 production 승격을 보류했던 결정도 이번 n=21
   measurement에서 재확인됐다(b_wb의 low_light Accuracy 40.5%는 b의 40.5%와 동일 — 이득이
   없고, strong_light에서만 b_wb가 23.8%로 b의 21.4%보다 near-tie 수준 우위). 두 방법 다
   production 전환 임계치(예: "normal 조명에서 50% 이상")에 못 미쳐, **Attribute를
   MetadataStore/VMS API에 연결하는 작업은 계속 보류한다.**
3. **"무채색 확신도 점수" 기반 unknown fallback은 설계/구현하지 않는다.** Next Action에서
   제안됐던 방향이지만, 이번 실측으로 후보 신호(S, raw BGR std) 모두 achromatic/chromatic을
   분리하지 못함을 확인했다 — 구현해도 의미 있게 동작하지 않을 것이므로, 근거 없는 기능을
   추가하지 않는다(지침: "처음부터 가장 좋은 설정과 완성된 코드를 제공하여 문제를 없애버리지
   않는다"의 반대 방향 원칙 — 반대로 "검증 안 된 해결책을 성급히 만들지 않는다"에도 해당).
4. **"판단 불가" crop 비율(53%)은 신규 관찰로 기록만 하고 이번 실험 범위에서 처리하지
   않는다**(Next Action으로 이월) — classify_person_attributes()에 "unknown/low_confidence"
   반환 값을 추가하는 것은 결과 2/3의 결론(무채색 신호가 불안정)과 결합해 재설계가 필요한
   별도 작업이다.

## Next Action
1. Attribute Metadata를 MetadataStore/VMS API에 연결하기 전에, b/b_wb 모두에 공통된 저조도
   무관 실패 원인(배경 Bleed, JPEG 압축 Noise)을 줄이는 더 근본적인 영역 분리 방법(예: 사람
   Segmentation Mask 활용, bbox 전체가 아닌 옷 영역만)을 검토할 것 — 단순 HSV 필터링은 n=21
   결과로 한계가 드러났다.
2. 원본 영상(Wikimedia) 접근이 복구되면 CCTV 각도(Top-down/측면 Wide shot)에서의 Attribute
   Accuracy를 coco128(지상 스냅샷 각도)과 비교 검증할 것 — 카지노 CCTV는 사람 사진 각도와
   다르다.
3. FC-012는 여전히 미해결로 유지한다(채도 기반 신호로는 해결 방향을 찾지 못함 — 원인 자체가
   조명 Cast보다 "crop에 섞인 비-옷 픽셀 비율"에 더 가깝다는 가설을 다음 세션에서 검증).
