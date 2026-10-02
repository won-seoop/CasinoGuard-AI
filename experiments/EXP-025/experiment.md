# EXP-025 Attribute Metadata - 상의/하의 색상 분류 Baseline vs 대안 A vs 대안 B

관련 실험: EXP-004(BestShot, person crop 처리 선례), EXP-018(FC-003, 합성 Ground Truth로 측정한
선례)

## Goal
지침 19(Attribute Metadata)의 Core MVP 이후 Stretch Goal 중 아직 손대지 않은 "상의 색상,
하의 색상" 분류를 Baseline부터 구축하고, 카지노처럼 조명이 강하거나 변하는 환경에서 Attribute가
얼마나 안정적인지 실측한다.

## Hypothesis
1. Baseline(bbox 전체를 한 번에 평균)은 상/하의가 다른 색일 때 두 색이 섞여 어느 쪽도 아닌
   값으로 분류될 것이다.
2. 상/하 영역을 나누기만 해도(대안 A) 상/하 구분은 가능해지지만, 배경 Bleed·밝기 변화에는
   여전히 취약할 것이다.
3. 그림자/하이라이트 픽셀을 제거하고 Hue 다수결을 쓰면(대안 B) 조명 변화에 더 안정적일
   것이다.

## Problem
- 지금까지 Metadata Pipeline(EXP-008)은 track_id/class/confidence/bbox/zone/dwell_time만
  저장하고, 지침 16이 "추후 확장"으로 명시한 upper_color/lower_color 등 Attribute는 아직
  구현되지 않았다.
- VMS Search(지침 17)가 향후 "빨간 상의를 입은 사람" 같은 검색을 지원하려면 이 Attribute가
  필요하다.

## Dataset
- 이 원격 자동화 세션도 기존 세션들과 동일하게 egress 정책상 Wikimedia/raw.githubusercontent.com/
  archive.org가 전부 403/연결 차단(사전 `curl` 테스트로 재확인)되어 원본 공개 CCTV 영상에
  접근하지 못했다. `yolo11n.pt`는 GitHub Release Asset 경로로 정상 다운로드됨을 재확인했다.
- 대신 ultralytics 패키지에 번들된 실제 사진 2장(`zidane.jpg`, `bus.jpg`, 둘 다 실제 인물
  사진이며 합성 이미지가 아님)에 YOLO11n Person Detection(conf=0.3)을 돌려 실제 검출 bbox로
  crop 6개를 뽑았다. 그중 `bus_3`(심한 Occlusion으로 사람 여부가 Crop만으로는 불확실)은
  Ground Truth 라벨링에서 제외했다(지침 31: Ground Truth를 억지로 만들지 않는다) — 최종
  평가는 5개 crop.
- Ground Truth(상의/하의 색상)는 Read 도구로 각 crop 이미지를 직접 눈으로 보고 수동
  라벨링했다(`scripts/run_exp025_attribute_color.py`의 `GROUND_TRUTH` 딕셔너리에 근거 기록).
  한계: 5개 crop 중 4개가 검정 계열을 포함하고 있어(배경 사진들이 정장/코트 차림 위주) 색상
  다양성이 낮다 — 이는 솔직하게 명시하는 Dataset 한계이며, 실제 색상이 거짓으로 꾸며지지는
  않았다(지침 8/31).
- 조명 안정성을 측정하기 위해 각 crop에 합성 조명 변형 4종(저조도 ×0.35, 강한조명/역광
  ×1.9+25 클리핑, 난색 Cast 백열등 근사, 한색 Cast 형광등/역광 근사)을 추가로 적용해
  `normal` 포함 총 5개 조명 조건으로 평가했다 — 이 변형은 Baseline 조명(정상) 대비
  **상대적** 안정성을 보기 위한 것이며, 조명 조건 자체에 Ground Truth를 새로 만들지는
  않는다(동일 crop의 `normal` 라벨을 그대로 기준으로 삼는다).

## Environment
- 원격 자동화 세션, `.venv`(Python 3.11.15) 신규 생성
  (opencv-python-headless 5.0.0, numpy 2.4.6, scikit-learn, ultralytics, pytest, lap, fastapi).
- CPU 전용.

## Configuration
`src/attributes/color.py` 파라미터:
- 분류 임계값: `VAL_BLACK=45`, `SAT_LOW=30`, `VAL_WHITE=195`, Hue bin 경계는 표준 12색상환
  근사(빨강/주황/노랑/초록/파랑/보라/분홍 7개 유채색 + 검정/흰색/회색 3개 무채색).
- 대안 A: `upper_end_ratio=0.5` 고정 비율 2분할(머리 제외 없음).
- 대안 B: `head_skip_ratio=0.22`, `upper_end_ratio=0.55`, `foot_skip_ratio=0.08`,
  `side_margin_ratio=0.12`, `region_dominant_color(sat_min=25, val_min=35, val_max=245,
  hue_bin_width=10, exclude_skin=True, min_filtered_fraction=0.3)`.
- 이 임계값들은 5개 crop 각각의 정답을 맞히도록 역산한 것이 아니라(지침 38: 숫자 조작 금지),
  일반적인 HSV 색상 이론(저채도=무채색, 저명도=검정) 기준으로 먼저 정하고 실제 데이터로
  검증하는 순서로 진행했다 — 그 결과 아래 Result에서 보듯 모든 방식의 정확도가 기대보다
  낮게 나왔고, 이를 그대로 기록한다.

## Baseline
`whole_bbox_mean_color()`: bbox 전체 픽셀의 평균 BGR을 한 번 HSV로 변환해 분류. 상/하의
구분이 원천적으로 불가능하다(단일 색상만 출력). 평가 시 "상의 GT 또는 하의 GT 중 하나라도
맞으면 정답"으로 최대한 관대하게 채점했는데도 Accuracy가 낮았다(아래 Result).

## Result

### 1차 측정 (대안 B 최초 구현, `exclude_skin` 없음)
| 방식 | normal | low_light | strong_light | warm_cast | cool_cast | 전체 평균 | 조명변형 평균(안정성) |
|---|---|---|---|---|---|---|---|
| Baseline | 0.20 | 0.80 | 0.20 | 0.00 | 0.40 | 0.32 | 0.35 |
| 대안 A | 0.20 | 0.70 | 0.10 | 0.10 | 0.30 | 0.28 | 0.30 |
| 대안 B (최초) | 0.20 | 0.10 | 0.20 | 0.10 | 0.20 | 0.16 | 0.15 |

가설과 반대로 대안 B가 Baseline보다도 낮았다. `results/EXP-025/predictions.csv`의 `normal`
조건 행을 직접 조사해 원인을 분석했다(아래 Analysis).

### Analysis (원인)
1. **피부색 오염**: `zidane_0`/`zidane_1`는 전신이 아니라 얼굴+상반신 클로즈업 crop인데,
   `head_skip_ratio=0.22`만으로는 턱선·목·뻗은 손 등 피부 픽셀을 다 걸러내지 못했다. 상의가
   실제로는 검정 정장인데 대안 B가 `orange`/`green`/`red`로 잘못 분류한 원인을 추적한 결과,
   필터링(그림자/하이라이트 제거) 후 남은 소수 픽셀이 전부 피부색이었고 Hue 다수결이 그
   소수 픽셀 쪽으로 쏠렸다.
2. **필터-폴백 버그(코드 결함, 진짜 버그)**: `region_dominant_color()`의 S/V 필터
   (`sat_min=25`)는 저채도(무채색) 픽셀을 전부 제외하는데, 진짜 검정 옷(V가 매우 낮음)은
   채도도 0에 가까워 **이 필터에 의해 거의 항상 통째로 제외됐다**. 즉 "필터링 후 남은
   픽셀이 0개면 전체 픽셀로 fallback"하는 최초 구현은, 검정 옷(다수)이 필터에 걸려 사라지고
   피부색(소수)만 남는 경우를 전혀 포착하지 못했다 — 소수 피부색 픽셀이 0개가 아니므로
   fallback이 발동하지 않았다.

### 2차 측정 (버그 수정 후 재측정, 동일 조건)
`src/attributes/color.py`에 두 가지를 수정했다:
- `_is_skin_like()` 피부색 HSV 휴리스틱 필터 추가(`exclude_skin=True` 기본값).
- 필터링된 픽셀 수가 전체의 `min_filtered_fraction`(기본 0.3) 미만이면 필터를 신뢰하지 않고
  원본 전체 픽셀로 되돌아가도록 폴백 조건을 "개수가 0" → "비율이 낮음"으로 일반화.

| 방식 | normal | low_light | strong_light | warm_cast | cool_cast | 전체 평균 | 조명변형 평균(안정성) |
|---|---|---|---|---|---|---|---|
| Baseline | 0.20 | 0.80 | 0.20 | 0.00 | 0.40 | 0.32 | 0.35 |
| 대안 A | 0.20 | 0.70 | 0.10 | 0.10 | 0.30 | 0.28 | 0.30 |
| **대안 B (수정 후)** | **0.40** | 0.60 | **0.30** | **0.40** | 0.30 | **0.40** | **0.40** |

(`results/EXP-025/summary.json`, `results/EXP-025/predictions.csv`에 전체 raw 데이터 보존.)

### pytest
`tests/test_attribute_color.py` 26개 신규(HSV 분류 경계, 영역 분할 비율, Baseline 두 색
블렌딩 한계, 대안 B의 배경 Bleed/그림자 강건성, 피부색 소수 오염 방지, 필터 완전 실패 시
fallback 등). 전체 회귀 179 passed / 7 skipped(기존 skip 유지, 회귀 없음).

## Failure Cases
- 검정 옷이 `strong_light`/`cool_cast`(한색 Cast) 조건에서 반복적으로 `blue`로 오분류됨
  (zidane_0/1, bus_0 전 방식 공통). `bus_0`은 `normal` 조명에서도 실측 상의 영역 median
  Hue가 이미 파랑 계열(배경의 파란 버스 반사광으로 추정)이었다 — 카메라 센서가 주변 조명
  색을 그대로 반영하는 물리적 한계이며, 이 프로젝트의 필터링만으로는 해결 불가능하다
  (White Balance 보정이 필요, 범위 밖). → **FC-012로 신규 등록**.
- `bus_1`(크림색 퍼(fur) 재질 자켓)은 `normal`/`low_light`/`warm_cast`에서 계속 `orange`로
  오분류됐다. 질감이 있는 밝은 색 의류는 그림자/하이라이트 경계 픽셀이 필터를 통과해
  Hue 다수결을 왜곡시키는 것으로 추정(확정은 아님, 재질별 심층 분석은 범위 밖).

## Analysis
대안 B는 Baseline 대비 전체 평균(0.32→0.40)과 조명 안정성(0.35→0.40) 모두 개선했지만,
절대 수치 자체는 여전히 낮다(40%). 이는 (1) 평가 Dataset이 n=5로 매우 작고 색상 다양성이
낮으며, (2) 머리/발 비율을 고정값으로 가정하는 접근 자체가 클로즈업 vs 전신처럼 Framing이
다른 crop에 똑같이 적용되기 어렵고, (3) 재질(퍼/텍스처)에 따라 그림자/하이라이트 필터의
효과가 달라지기 때문으로 분석된다. "대안 B가 대안 A/Baseline보다 낫다"는 결론은 이번
Dataset에서는 성립하지만, 더 크고 다양한 Dataset 없이 일반화를 주장하지는 않는다(지침 36:
의미 있는 Metric만 선택, 지침 38: 숫자 조작 금지).

## Decision
대안 B(영역 분할 + 그림자/하이라이트/피부색 필터 + Hue 다수결)를 `classify_person_attributes()`
기본 구현(`method="b"`)으로 채택한다. 근거: 동일 조건에서 실측한 Accuracy와 조명 안정성이
Baseline/대안 A보다 모두 높고, 코드 결함(필터-폴백 버그)을 실제로 발견·수정하는 과정에서
설계가 더 견고해졌다. Metadata Pipeline에 Attribute를 바로 통합하지 않고 `src/attributes/`
독립 모듈로만 남긴다 — 지침 19: "Attribute Classification 자체가 프로젝트 중심이 되어서는
안 된다", 그리고 아직 실제 영상(원본 CCTV)에서 검증되지 않은 상태로 MetadataStore 스키마를
확장하는 것은 시기상조라고 판단했다.

## Next Action
1. 원본 영상 접근이 가능한 세션에서 더 크고 다양한(색상 다양성 높은) 실제 인물 crop으로
   재검증 — 이번 n=5 Dataset의 한계(검정 편중)를 해소해야 "대안 B가 Baseline보다 낫다"는
   결론을 더 신뢰할 수 있다.
2. FC-012(조명 색 Cast로 인한 검정→파랑 오분류)는 해결하지 않고 백로그로 남긴다 — 근본
   해결은 White Balance 보정 또는 무채색 확신도(achromatic confidence) 점수를 추가해
   낮은 확신도의 분류 결과는 Metadata에 "unknown"으로 남기는 방향이 유력해 보이나, 검증
   전이라 Decision에는 포함하지 않는다.
3. MetadataStore/VMS API에 upper_color/lower_color를 실제로 연결하는 작업은 위 1번
   검증 이후로 미룬다(지침 19 원칙 재확인).
