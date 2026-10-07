# EXP-030 FC-012 strong_light(노출 과다) 복원 - Adaptive Gamma Correction

관련 실험: EXP-025(Baseline/A/B, PAR-016), EXP-026(FC-012 White Balance, PAR-017),
EXP-027(coco128 n=21 재검증, PAR-018), EXP-028(Segmentation Mask, PAR-019),
EXP-029(Attribute 실시간 파이프라인 통합, PAR-020)

## Goal
01. Architecture & Roadmap 다음 Action 3번: FC-012는 EXP-026(White Balance)/EXP-028
(Segmentation Mask)로도 해결되지 않았고, 다음 후보로 "Gamma 보정 등 노출 복원 기법 또는
'판단 불가/낮은 신뢰도' 표시로의 설계 전환"을 명시했다. 이 실험은 전역 밝기(노출) 자체를
보정하는 Adaptive Gamma Correction을 실제로 구현·검증해, FC-012가 겨냥한 두 실패 조건
(strong_light/cool_cast) 중 지금까지 어떤 보정으로도 개선되지 않은 strong_light(오히려
WB/Mask를 추가할수록 18.2%→22.7%→27.3%로 악화)를 고칠 수 있는지 확인한다.

## Hypothesis
1. White Balance(EXP-026)는 "채널 간 상대적 Cast"를 보정하는 기법이라 strong_light
   (EXP-025 `img*1.9+25`, 모든 채널이 함께 밝아지는 전역 노출 문제)에는 구조적으로
   적용되지 않는다 - Gamma Correction(전역 밝기 보정)을 추가하면 strong_light의
   GT=black→blue 오분류율이 개선될 것이다.
2. (대안 A) crop(person bbox) 자신의 평균 밝기로 감마를 추정하면, crop 밝기가 "조명
   노출"과 "옷 색상(albedo)"이 섞인 신호이므로 진짜 검정 옷까지 함께 밝아져 Accuracy가
   악화될 것이다 - 이는 구현 전 설계 단계에서 예상한 리스크였다(아래 Decision 참고).
3. (대안 B) crop이 아니라 같은 Frame의 배경(person bbox 밖 영역)에서 감마를 추정하면,
   배경은 옷 색상과 무관하므로 노출만 더 순수하게 반영해 대안 A의 confound를 피할 수
   있을 것이다.

## Problem
EXP-026(PAR-017)과 EXP-028(PAR-019)이 실측한 GT=black→blue 오분류율(strong_light)은
b(18.2%) → b_wb(22.7%) → c_wb(27.3%)로 보정을 추가할수록 오히려 악화되고 있었다. 두
실험 모두 "채널 간 상대적 색 균형(White Balance)"이나 "배경 Bleed 제거(Segmentation
Mask)"를 겨냥했을 뿐, strong_light가 실제로 만드는 문제(전역 노출 과다로 모든 픽셀의
V가 함께 올라가 검정 옷의 V도 VAL_BLACK=45 임계값을 넘어서 버림)를 직접 겨냥한 적이
없었다.

## Dataset
EXP-027/028과 완전히 동일한 coco128(실제 COCO train2017) n=21 Ground Truth person crop +
동일 5종 합성 조명(normal/low_light/strong_light/warm_cast/cool_cast, `apply_lighting()`
동일 구현). 재현성 확인을 위해 coco128을 동일 경로(`github.com/ultralytics/assets/
releases/download/v0.0.0/coco128.zip`)로 재다운로드하고 동일 Detector(yolo11n.pt,
conf=0.4, classes=[0])로 crop을 재검출해 GT key 21개가 모두 그대로 재현됨을 확인했다.

## Environment
원격 자동화 세션, 컨테이너가 매 세션 새로 초기화되므로 `.venv`(Python 3.11)를 신규
생성하고 opencv-python-headless, numpy, ultralytics 8.4.174, lap, fastapi, pytest를
재설치했다(EXP-026~029와 동일 구성, data/raw·models/·.venv는 지침 33에 따라 Git에
커밋되지 않으므로 매 세션 재생성이 정상). CPU 전용.

## Configuration
`src/attributes/color.py`에 3개 함수 추가(기존 method b/b_wb 파라미터는 변경하지 않음):
- `estimate_gamma_from_reference(reference_bgr, target_mean=128.0, gamma_min=0.4,
  gamma_max=2.5)`: reference의 평균 밝기를 target_mean으로 맞추는 감마를 역산한다
  (`gamma = log(target_mean/255) / log(mean/255)`, 표준 Auto-Exposure 공식).
  white_balance_gray_world와 동일한 이유로 극단값(순수 흑/백)은 clip해 안전장치를 둔다.
- `apply_gamma(img_bgr, gamma)`: power law(`(img/255)^gamma * 255`)를 그대로 적용한다.
- `adaptive_gamma_correct(crop_bgr, ...)`: 대안 A. crop 자신의 밝기로 감마를 추정해
  자신에게 적용한다. **EXP-030에서 기각됨** - 아래 Result/Decision 참고.
- `adaptive_gamma_correct_from_background(crop_bgr, background_bgr, ...)`: 대안 B.
  감마를 crop이 아니라 같은 Frame의 배경(bbox 밖 영역)에서 추정한다.
- `classify_person_attributes()`에 `method="b_gamma"`(대안 A를 method b 앞에 추가),
  `"b_wb_gamma"`(Gamma→WB 순서로 결합) 추가.
- `scripts/run_exp030_attribute_gamma_correction.py`: `detect_person_crops_and_bboxes()`로
  원본 전체 이미지 경로+bbox를 함께 보존(대안 B는 crop 밖 배경이 필요해 EXP-027/028의
  "crop만 저장"으로는 부족함). `background_sample()`이 bbox를 제외한 Frame 나머지
  픽셀을 추출하고, `predict_b_gamma_bg()`가 그 배경 픽셀로 추정한 감마를 crop에 적용한
  뒤 method b와 동일한 상/하 분리·필터링·Hue 다수결을 그대로 적용한다
  (`BACKGROUND_MIN_PIXELS=5000` 미만이면 경고만 출력, 이번 n=21에서는 전부 통과).

## Baseline
EXP-027/028이 측정한 b/b_wb 수치(coco128 n=21, 동일 Dataset): Overall Accuracy
b=26.2%/b_wb=32.4%, GT=black strong_light→blue 오분류율 b=18.2%/b_wb=22.7%. 이 실험은
이 두 수치를 변경하지 않고(코드 미변경, 재실행으로 동일 수치 재확인) 그 위에 b_gamma/
b_wb_gamma/b_gamma_bg를 추가로 비교한다.

## Result
`results/EXP-030/summary.json` (n=21 crop x 5 lighting x upper+lower = 210 셀/method).

| Method | Overall Acc | Normal 조명만 | GT=black Acc | black→blue(strong_light) | black→blue(cool_cast) |
|---|---|---|---|---|---|
| b (production) | 26.2% | 33.3% | 39.1% | 18.2% | 59.1% |
| b_wb (실험용) | 32.4% | 35.7% | 49.1% | 22.7% | 22.7% |
| **b_gamma (대안 A, 기각)** | **8.1%** | **9.5%** | **7.3%** | 18.2% | **86.4%** |
| b_wb_gamma (A+WB 결합) | 10.0% | 11.9% | 7.3% | 31.8% | 36.4% |
| **b_gamma_bg (대안 B)** | 20.0% | 26.2% | 28.2% | **4.6%** | 68.2% |

조건별 Accuracy(accuracy_by_condition)도 함께 확인:

| Method | normal | low_light | strong_light | warm_cast | cool_cast |
|---|---|---|---|---|---|
| b | 33.3% | 40.5% | 21.4% | 16.7% | 19.0% |
| b_gamma | 9.5% | 4.8% | 16.7% | 7.1% | 2.4% |
| b_gamma_bg | 26.2% | **4.8%** | **35.7%** | 21.4% | 11.9% |

## Failure Cases
**대안 A(b_gamma)는 모든 조건에서(목표였던 strong_light까지 포함해 거의 개선 없음)
Accuracy가 붕괴했다.** 원인을 직접 측정해 확인했다(coco128 n=21, normal 조명, crop
자신의 평균 밝기 분포): GT=black crop들의 평균 밝기가 24.7~153.3까지 퍼져 있고, 이
범위가 흰 옷 GT crop들의 평균 밝기(103.98~153.25)와 그대로 겹친다. crop 자신의 평균
밝기를 target_mean(128)으로 강제 정규화하면 "검정 옷은 어둡다"는 분류 신호 자체가
지워진다 - 진짜 검정 옷(평균 24.7)은 밝게 끌어올려지고, 진짜 흰 옷(평균 153.25)은
어둡게 낮춰져 둘 다 비슷한 중간 밝기로 수렴한다. normal 조명(원래 보정이 필요 없는
조건)에서도 Accuracy가 33.3%→9.5%로 떨어진 것이 이 confound의 직접 증거다.

## Analysis
대안 B(b_gamma_bg, 배경에서 감마 추정)는 대안 A가 만든 confound를 정확히 해결했다 -
GT=black crop 자신은 그대로 두고 배경만 보고 노출을 판단하므로, strong_light의
GT=black→blue 오분류율이 18.2%(b) → **4.6%**(b_gamma_bg)로 지금까지(EXP-026/028 포함)
측정된 모든 방법 중 가장 낮아졌고 strong_light 조건별 Accuracy도 21.4%→35.7%로 개선됐다
- 가설 3이 지지됐다.

그러나 Overall Accuracy는 b_gamma_bg가 b보다 낮다(20.0%<26.2%), 원인은 low_light
조건의 급격한 악화(40.5%→4.8%)다. 이 합성 Benchmark의 low_light(`img*0.35`)와
strong_light(`img*1.9+25`)는 Frame 전체(배경+person)에 동일하게 적용되는 전역 변형이라,
배경과 person의 밝기가 항상 같은 방향·같은 비율로 움직인다. 즉 이 synthetic 설정에서는
"배경이 어두우면 person도 반드시 어둡다"가 항상 성립하므로, 배경 기준 감마 보정이
low_light 조건에서는 진짜 검정 옷(원래 어두운 것이 정상)까지 "노출이 부족하다"고 오판해
밝게 끌어올린다 - 대안 A와 같은 confound가 조건이 바뀌어도 완전히 사라지지 않고, 전역
균일 조명 변형 자체가 "배경과 옷 색상을 분리할 수 없게" 만든다.

반대로 실제 카지노/CCTV 환경에서 대안 B가 원래 겨냥하는 상황(역광, 스포트라이트 등
Person과 배경의 노출이 실제로 다른 경우)은 이 synthetic Benchmark로는 만들 수 없다 -
이번 결과는 "배경 기준 Gamma가 실제로 유효한 상황(노출이 분리된 경우)에서는 효과가
있다(strong_light 측정에서 확인)"와 "이 synthetic Benchmark 자체의 한계(전역 균일
변형이라 low_light에서는 배경과 옷이 분리되지 않는다)"를 함께 보여준다.

warm_cast/cool_cast(채널 Cast)에는 Gamma(밝기 보정)가 애초에 설계상 맞지 않는 문제라
b_gamma_bg도 cool_cast(68.2%)를 개선하지 못했다 - 이는 예상된 결과이며 가설 1이 전제한
"WB와 Gamma는 서로 다른 실패 원인을 겨냥한다"가 그대로 확인된 것이다.

## Decision
(A) crop 자신의 밝기로 감마를 추정 — **기각**. 실측으로 확인된 confound(crop 밝기가
옷 색상과 노출을 분리하지 못함) 때문에 모든 조건에서 Accuracy가 붕괴한다. 코드는
`adaptive_gamma_correct()`로 남기되(지침 38, 실패한 실험을 삭제하지 않는다), production
경로(`classify_person_attributes(method="b")`)에는 연결하지 않는다.

(B) 배경(Frame에서 person bbox 제외)에서 감마를 추정 — 목표였던 strong_light의
GT=black→blue 오분류율을 지금까지 가장 크게 개선했다(18.2%→4.6%)는 점에서 방향 자체는
타당하다고 판단하지만, **production으로 승격하지 않는다**. 이유: (1) 이 synthetic
Benchmark의 전역 균일 조명 변형 특성상 low_light에서 심각한 Regression(40.5%→4.8%)이
새로 생겼고, 이 Regression이 "실제로 틀린 방향"인지 "Benchmark 자체의 한계"인지를 현재
Dataset으로는 구분할 수 없다. (2) 실시간 파이프라인(EXP-029)에서 person bbox 밖 배경
픽셀까지 Frame에서 함께 들고 있어야 하므로 BestShot crop만 보관하는 현재 구조(PAR-004
이후 원본 프레임을 보관하지 않음)와 맞지 않아 통합 비용이 크다. Attribute는 여전히
MetadataStore/VMS API에 연결하지 않은 실험 단계이므로(EXP-027/028 Decision과 동일),
지금 production 코드를 바꾸는 것보다 더 분리 가능한(배경과 전경의 노출이 실제로 다른)
실제 CCTV 영상에서 재검증하는 쪽이 우선이다.

FC-012(검정 옷 strong_light/cool_cast 오분류)는 이제 **네 가지** 보정(White Balance,
Segmentation Mask, Gamma-self, Gamma-background)을 실측으로 검토했고, Gamma-background가
strong_light만 부분적으로 개선했을 뿐 어느 것도 전체 문제를 해결하지 못했다. 다음 후보는
Roadmap이 이미 명시한 대로 "판단 불가/낮은 신뢰도 표시로의 설계 전환"이 남은 유일한
미검토 방향이다 - 네 가지 서로 다른 pixel-level 보정이 모두 부분적 효과만 내는 것은,
이 문제가 "더 나은 보정 알고리즘"이 아니라 "일부 조건에서는 confidence 자체가 낮다는
것을 인정하는 설계"로 풀어야 한다는 쪽에 힘을 보탠다(단 EXP-027에서 이미 achromatic
confidence 신호 자체가 분리되지 않음을 확인했으므로, 그 신호가 아닌 다른 confidence
기준 - 예: 조명 조건 자체를 Frame 밝기로 감지해 "이 Track의 Attribute는 낮은 조명/과다
노출 구간에서만 관측됐다"는 구간 단위 플래그 - 가 다음 설계 후보다).

## Next Action
1. FC-012는 백로그로 유지하되, 네 번째 보정(Gamma-background)까지 기각/부분효과로
   종료되었음을 기록한다. 다음 후보는 pixel-level 보정이 아니라 "Track의 Attribute
   관측 구간 전체의 Frame 밝기 분포로 낮은 신뢰도 구간을 표시"하는 설계로 전환할 것
   (EXP-027의 achromatic confidence와는 다른 신호 - crop 자체가 아니라 Frame/배경
   밝기 기준).
2. 원본 영상(실제로 배경과 person의 노출이 분리되는 역광/스포트라이트 장면) 접근이
   가능해지는 세션에서 b_gamma_bg를 재검증할 것 - 이번 synthetic Benchmark는 전역
   균일 변형이라 "배경 기준 노출 추정"이 원래 겨냥하는 상황(노출이 분리된 경우)을
   만들 수 없었다.
3. Attribute의 MetadataStore/VMS API 연결은 계속 보류(EXP-027/028 Decision과 동일 -
   Overall Accuracy가 아직 production 임계치에 크게 못 미침).
