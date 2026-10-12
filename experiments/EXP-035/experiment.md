# EXP-035 Attribute Color GT 21 crop x 42 영역 전체 재검증 + 단일 진실 공급원 전환

관련 실험: EXP-027(PAR-018, GT 최초 n=21 라벨링), EXP-032(PAR-023, 오분류 110셀
중 10셀만 육안 재확인), EXP-033(PAR-024), EXP-034(PAR-025, 현재 production 후보
val_min=0/sat_min=20 확정)

## Goal

Roadmap 01. Architecture & Roadmap이 3회 연속 다음 세션으로 이월한 작업 —
"GT 라벨 21개 crop × 42개 영역 전체 재검증" — 을 실제로 수행한다. EXP-032는
"오분류로 보고된 셀의 일부(normal 조건 10셀)"만 육안 재확인했는데, 이 범위는
"classifier가 틀렸는데 GT도 우연히 같이 틀려서 '정답'으로 집계된 셀"을 원천적으로
볼 수 없다는 한계가 있었다. 이번 실험은 그 사각지대를 포함해 42개 라벨 전부를
처음부터 다시 검토한다.

## Hypothesis

1. 오분류로 보고되지 않은 나머지 32셀(42-10) 중에서도 추가 라벨 오류가 있을 것이다.
2. GROUND_TRUTH가 6개 스크립트(EXP-027/028/030/031/032/033/034)에 복사-붙여넣기
   되어 있으므로, EXP-032가 찾은 교정 2건이 일부 스크립트에는 전파되지 않았을
   것이다.
3. (사후 추가) 교정된 GT로 EXP-034의 최종 후보(val_min=0, sat_min=20)를 재측정하면
   Accuracy가 개선될 것이다 — EXP-032의 2건 교정 재측정에서 normal Accuracy가
   +5.5~5.9pp 개선됐던 선례를 따를 것이라는 가설.

## Problem

- EXP-034까지 production 후보(val_min=0, sat_min=20, c_wb_relaxed)의 Overall/GT=black
  Accuracy(50.95%/79.00%)는 GT 자체에 알려지지 않은 라벨 오류가 섞인 채로 측정된
  수치였다. GT 품질을 끝까지 검증하지 않은 상태에서 Threshold를 계속 튜닝하는 것은
  "틀린 정답지에 더 정확히 맞추는" 리스크가 있다.
- `GROUND_TRUTH` dict가 단일 진실 공급원 없이 6개 스크립트에 흩어져 있어, 한 곳의
  교정이 전체에 반영된다는 보장이 전혀 없었다(구조적 결함).

## Dataset

coco128 person crop 21개(EXP-027과 완전히 동일, `results/EXP-027/crops/*.jpg`에
이미 저장되어 있음) x upper/lower 2개 영역 = 42 라벨. 5종 합성 조명
(normal/low_light/strong_light/warm_cast/cool_cast, EXP-025부터 모든 실험과 동일)
적용 시 210개 (name, region, condition) 데이터 포인트.

## Environment

coco128 이미지(`/tmp/coco128_extract`, ultralytics GitHub Release Asset
`v0.0.0/coco128.zip`로 재다운로드 - 이 세션은 컨테이너가 매번 새로 시작되어 캐시가
없었다), YOLO11n(detector)/YOLO11n-seg(segmenter) 재다운로드, ultralytics 8.4.175.
이 세션은 egress 제약이 이전 로그들에 기록된 것(Wikimedia 등 외부 도메인 403)과
달리 GitHub Release Asset(coco128, YOLO 가중치)에는 접근 가능했다 - 이전
EXP-027/028 세션들이 이미 확인한 "ultralytics 공식 Release Asset 경로는
접근 가능하다"는 결론과 일치한다.

## Configuration

- `evaluation/attribute_color/coco128_ground_truth.py`(신규, 단일 진실 공급원):
  `GROUND_TRUTH`(교정 반영), `CORRECTIONS`(교정 로그, 근거 포함), `assert_consistent()`.
- 측정 config 2개: baseline(val_min=35, sat_min=25, EXP-033 이전 production 기본값),
  candidate(val_min=0, sat_min=20, EXP-034 최종 후보). 둘 다 `c_wb_relaxed`
  (Segmentation Mask + Gray World White Balance + region_dominant_color_relaxed).

## Baseline

EXP-034가 썼던 GT(= EXP-032의 2건 교정만 반영, 이번 EXP-035의 5건은 미반영) -
`scripts/run_exp035_ground_truth_audit.py`의 `GROUND_TRUTH_BEFORE`로 그대로 보존.

## Result

### 1) 42개 라벨 전체 육안 재검증 결과: 신규 오류 5건 추가 발견

`results/EXP-027/crops/*.jpg`를 3배 Upscale해 21개 crop x upper/lower 42개 라벨을
전부 원본 이미지와 대조했다. EXP-032가 이미 찾은 2건(086_0 upper, 634_0 upper)
외에 5건을 새로 발견했다:

| Crop | 영역 | 기존 GT | 교정 GT | 근거 |
|---|---|---|---|---|
| 000000000370_0 | upper | pink | white | 브로콜리를 든 여아의 상의(나시)는 흰색/크림색. 분홍은 하단 치마(lower)뿐인데 원 주석이 상/하를 뭉뚱그려 upper까지 분홍으로 잘못 라벨링 |
| 000000000415_0 | lower | white | blue | 테니스 선수(Federer)의 반바지는 선명한 로열 블루. 흰색 영역은 손목밴드/신발/양말뿐 |
| 000000000536_0 | upper | black | purple | 원피스가 와인/버건디 톤의 짙은 자주색 - 무채색(black)이 아니라 COLOR_BUCKETS의 purple에 더 가까움 |
| 000000000536_0 | lower | black | purple | 위와 동일한 원피스(상하 연결) |
| 000000000623_0 | lower | pink | black | 원 주석의 '핑크 바지'는 크롭에 겹친 인접 인물(분홍 옷 아동)의 옷이었다. 라벨 대상 인물(회색 가디건) 본인의 하의는 짙은 회색/검정 |

42개 중 7개(16.7%)가 교정 대상이었다 - Hypothesis 1 확인(추가 오류가 실제로
존재했다).

### 2) 구조적 결함 확인: GROUND_TRUTH 복사-붙여넣기로 인한 Drift

EXP-032가 교정한 2건(086_0 upper, 634_0 upper)이 실제로 어느 스크립트에
반영됐는지 `grep`으로 직접 확인했다:

| 스크립트 | 086_0 upper | 634_0 upper |
|---|---|---|
| EXP-027 | black (미반영) | black (미반영) |
| EXP-028 | black (미반영) | black (미반영) |
| EXP-030 | black (미반영) | black (미반영) |
| EXP-031 | black (미반영) | black (미반영) |
| EXP-032(자기 자신) | black (미반영) | black (미반영) |
| EXP-033 | white (반영됨) | gray (반영됨) |
| EXP-034 | white (반영됨) | gray (반영됨) |

EXP-032는 자기 자신의 스크립트에 있는 `GROUND_TRUTH` dict조차 고치지 않고,
`error_rows_gt_corrected.csv`라는 별도의 1회성 재채점 파일로만 교정 효과를
보여줬다(experiment.md 본문의 "2건만 고쳐 같은 파이프라인을 재측정하면" 표).
그 이후 EXP-033을 작성한 세션이 그 experiment.md를 읽고 자기 복사본에 수동으로
반영했을 뿐, 단일 소스가 없어 "한 곳을 고치면 전부 고쳐진다"는 보장이 전혀
없었다 - Hypothesis 2 확인.

### 3) 교정된 GT로 production 후보 재측정: Accuracy는 오히려 소폭 하락

`scripts/run_exp035_ground_truth_audit.py`로 candidate(val_min=0, sat_min=20)와
baseline(val_min=35, sat_min=25)을 교정 전/후 GT 양쪽으로 재측정했다
(`results/EXP-035/summary.json`):

| Config | Metric | 교정 전 | 교정 후 | 변화 |
|---|---|---|---|---|
| candidate(val_min=0, sat_min=20) | Overall Accuracy | 50.95% | 50.00% | **-0.95pp** |
| candidate | GT=black Accuracy | 79.00% | 78.95% | -0.05pp(거의 동일, N 변화) |
| candidate | 비검정 GT Accuracy | 25.45% | 26.09% | +0.64pp |
| baseline(val_min=35, sat_min=25) | Overall Accuracy | 41.43% | 41.43% | 0.00pp |
| baseline | GT=black Accuracy | 60.00% | 61.05% | +1.05pp |
| baseline | 비검정 GT Accuracy | 24.55% | 25.22% | +0.67pp |

Hypothesis 3(교정하면 Accuracy가 개선될 것)은 **반증**됐다 - candidate config는
오히려 소폭 하락했다. baseline config는 소폭 개선(GT=black +1.05pp)됐다는 점과
대비된다.

그래도 두 config의 **순위는 교정 후에도 바뀌지 않았다**: candidate가 Overall
(50.00%>41.43%)/GT=black(78.95%>61.05%)/비검정(26.09%>25.22%) 세 지표 모두에서
baseline을 역전 없이 지배한다(`candidate_still_beats_baseline_after_gt_correction:
true`). 다만 candidate의 GT=black 우위 폭은 19.00pp(교정 전)에서 17.90pp(교정 후)로
줄었다 - EXP-034가 측정한 candidate의 우위가 일부는 라벨 오류로 부풀려져 있었다.

## Failure Cases

없음(Classifier/Pipeline 코드는 전혀 바꾸지 않았다 - 이번 실험은 GT 데이터 품질과
그 측정 구조만 다룬다).

## Analysis

`results/EXP-035/predictions.csv`에서 GT가 바뀐 25개 행(5개 교정 cell x 5개 조명)
중 14개 행이 정오답 판정 자체가 뒤바뀌었다(candidate config 기준):

- **000000000536_0(8개 행, 상/하x4개 조명)**: 이전까지 "정답"으로 집계됐지만
  교정 후 "오답"이 됐다. 원인은 분명하다 - candidate config(val_min=0)는
  "그림자 하한 필터를 거의 제거"하는 설계라서 어두운/채도 있는 픽셀을 `black`으로
  분류하는 경향이 체계적으로 강하다(EXP-033/034 전체가 이 경향을 "검정 옷을
  더 잘 잡는다"는 의도로 추구해왔다). 536_0는 실제로는 와인색 원피스인데 GT가
  `black`으로 잘못 라벨링되어 있었던 탓에, classifier의 "일단 어두우면 black"
  이라는 체계적 편향이 **우연히 틀린 정답과 일치**해 "정답"으로 집계되고 있었다.
  라벨을 고치자 이 8개 셀에서 classifier가 실제로는 틀리고 있었다는 사실이
  드러났다.
- **000000000370_0 upper(strong_light 1개), 000000000623_0 lower(4개 조명)**:
  반대로 이전까지 "오답"으로 집계됐지만 교정 후 "정답"이 됐다(+5건) - classifier의
  예측 자체는 맞았는데 GT가 틀려서 부당하게 감점되고 있었다.
- 순효과: -8(536_0 손실) +6(370_0/623_0 이득) = **-2/210 cell** -> Overall
  Accuracy -0.95pp. 손실이 이득보다 컸던 이유는 536_0 하나가 상/하x5조명=10cell
  전부에 영향을 준 반면, 나머지 교정은 조명 조건에 따라 classifier 예측 자체가
  어차피 둘 다 틀려서(correct_before=correct_after=0) 무변화로 끝난 cell이
  섞여 있었기 때문이다(415_0 lower는 어느 조명에서도 pred가 white/blue 둘 중
  무엇과도 일치하지 않아 5개 행 전부 무변화).

이 결과가 주는 교훈은 EXP-026/EXP-032가 이미 확립한 "평균 Metric 개선만 보고
문제가 해결됐다고 결론 내리면 안 된다"(PAR-017)는 원칙의 연장선이지만 한 단계
더 날카롭다 - **GT 라벨 오류가 모델의 체계적 편향과 같은 방향으로 나 있으면,
그 오류는 무작위 노이즈가 아니라 그 편향을 검증해주는 것처럼 보이는 가짜 신호가
된다.** candidate config가 baseline보다 나은 것은 여전히 사실이지만(순위는
유지), EXP-034가 측정한 "candidate가 baseline을 얼마나 능가하는가"의 정확한
크기는 라벨 오류 1건(536_0) 때문에 부풀려져 있었다.

## Decision

1. `evaluation/attribute_color/coco128_ground_truth.py`를 신규 단일 진실
   공급원으로 채택한다. 앞으로 이 GT를 쓰는 모든 실험은 여기서 import하고,
   다시 복사-붙여넣기하지 않는다(재발 방지, PAR-026 9번 항목).
2. EXP-034의 **순위 결론(candidate > baseline)은 그대로 유지**한다 - GT 교정
   후에도 역전이 없다. production 기본값으로 **아직 승격하지 않는다**는 기존
   결정도 그대로 유지한다(n=21 한계는 여전히 유효하고, 이번 실험으로 오히려
   "Accuracy 절대 수치가 라벨 품질에 민감하다"는 사실이 추가로 확인됐기 때문에
   더더욱 섣불리 승격할 근거가 없다).
3. EXP-027/028/030/031/032의 기존 스크립트 안 `GROUND_TRUTH` 복사본은 **수정하지
   않는다** - 각 실험은 당시 실행된 그대로의 불변 기록이며, 그 시점에 어떤 GT로
   측정했는지 자체가 "라벨 오류가 결과에 어떤 영향을 줬는지" 분석할 때 필요한
   사료다. 대신 이 experiment.md와 `evaluation/` 모듈이 "현재 진실"을 가리키는
   단일 지점이 된다.

## Next Action

1. GT 교정이 전체 순위를 안 바꿨으므로 추가 GT 재검증의 ROI는 낮다. 다음 우선순위는
   n을 21보다 늘리는 것(별도 coco128 배치 또는 원본 CCTV 영상 접근 시)으로
   이동한다.
2. (EXP-029 Next Action, 계속 이월) 원본 Wide-angle CCTV 영상 접근이 가능한
   세션에서 `run_full_pipeline.py`를 실제로 실행해 Attribute 통합 오버헤드를
   재확인.
3. Person Re-ID 또는 People Counting(EXP-012) band_px의 MOT20-01 GT 대비 MAE
   측정을 다음 Stretch 후보로 고려.
