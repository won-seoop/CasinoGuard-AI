# EXP-038 Person Re-ID FC-013(b) warm_cast/cool_cast - Gamma-Background + White Balance 결합 검증

## Goal
FC-013(Person Re-ID 임베딩이 극단적 조명 변형에서 붕괴)의 두 갈래 중 (a) 전역 노출
붕괴(low_light/strong_light)는 EXP-037(배경 기반 Gamma 보정)로 부분 완화됐지만, (b)
채널 Cast(warm_cast/cool_cast)는 그대로 남아 있었다(오히려 2.2~8.9pp 소폭 악화). 이
실험은 Attribute 분류(FC-012)에서 채널 Cast를 겨냥해 이미 검증된 White Balance(Gray
World, EXP-026 대안 B)를 EXP-037이 채택한 Gamma-Background 뒤에 추가로 적용하면
warm_cast/cool_cast Top-1이 개선되는지, 그리고 그 과정에서 새 회귀가 생기지 않는지
Roadmap(01. Architecture & Roadmap) EXP-037 Next Action 7-(a)에 따라 실측한다.

## Hypothesis
Gamma(전역 노출)와 White Balance(채널 간 상대적 Cast)는 서로 다른 원인을 겨냥하므로
결합해도 서로 상쇄되지 않고 합쳐질 것이다 - warm_cast/cool_cast Top-1이 Gamma-Background
단독(EXP-037)보다 개선되고, 이미 Gamma-Background가 크게 개선한 low_light/strong_light
와 기하 변형(occlusion/bbox_jitter/flip)에는 새 Regression이 생기지 않을 것이다.

## Problem
EXP-037은 FC-013(a)(저노출/과다노출 붕괴)를 배경 기반 Gamma 보정으로 크게 완화했으나
(strong_light Top-1 YOLO Backbone 0.0%→66.7%), FC-013(b)(warm_cast/cool_cast)는 거의
그대로였고 오히려 소폭(2.2~8.9pp) 악화됐다. "한 Task(Attribute)에서 검증된 보정이 다른
Task(Re-ID)에도 그대로 통할 것"이라는 가정을 검증 없이 적용하면 EXP-031(Frame 노출/Cast
confidence 표시가 가설과 정반대로 반증된 사례)의 실수를 반복할 위험이 있어, 적용 전에
반드시 실측해야 한다(Roadmap이 명시한 "EXP-031이 반증한 패턴을 반복하지 않는다").

## Dataset
- Gallery: coco128 실제 person crop 45명(`results/EXP-027/crops`, EXP-027/036/037과
  완전히 동일한 집합 - 실행 시 재탐지 결과로 검증됨).
- Query: EXP-036/037과 완전히 동일한 Protocol(RNG seed 42, 동일 호출 순서) - 조명 4종
  (low_light/strong_light/warm_cast/cool_cast) + 기하 변형 3종(occlusion/bbox_jitter/
  flip) = 45명 x 7 variant = 315개 Query.
- coco128 원본 이미지는 이번 세션에서 `https://github.com/ultralytics/assets/releases/
  download/v0.0.0/coco128.zip`(egress 허용 경로, EXP-027이 확인한 경로)로 재다운로드함.

## Environment
- 원격 자동화 세션(claude-sonnet-5). Python 3.11, ultralytics 8.4.178, torch 2.14.1,
  opencv-python-headless 5.0.0.93, pytest 9.1.1 (이번 세션에서 신규 설치 - 세션
  시작 시 `python`/`python3`가 가리키는 인터프리터(3.11)와 시스템 `pip`이 가리키는
  인터프리터(3.13)가 달라 처음 설치한 패키지가 보이지 않는 문제를 발견, `python3 -m pip
  install`로 재설치해 해결함).
- `yolo11n.pt`/`yolo11n-cls.pt`는 ultralytics 공식 GitHub Release에서 자동 다운로드됨.

## Configuration
- `src/attributes/color.py::gamma_background_then_white_balance()` (신규 추가) -
  `adaptive_gamma_correct_from_background()`(EXP-030/037 대안 B) 적용 후
  `white_balance_gray_world()`(EXP-026 대안 B)를 적용(EXP-030의 `b_wb_gamma`와 동일한
  순서: 감마 먼저, WB 나중).
- `scripts/run_exp038_reid_wb_gamma_background.py`: EXP-037과 동일한 배경 추출/Query
  구성 로직을 재사용하되, 보정 함수를 `adaptive_gamma_correct_from_background`(재현,
  EXP-037 대안 B)와 `gamma_background_then_white_balance`(신규) 두 Pipeline으로
  나란히 실행해 직접 비교.
- 임베딩: `color_histogram_embedding`(Baseline), `YoloBackboneEmbedder`(Candidate,
  yolo11n-cls.pt backbone), 50:50 Score Fusion(`rank_gallery_fused`).

## Baseline
EXP-037 대안 B(Gamma-Background 단독)를 이 실험에서 그대로 재실행해 재현성을 확인함
(Overall Top-1 Baseline 0.5111/Candidate 0.7429 - EXP-037 기록값과 완전히 일치).

## Result

| 조건 (Candidate=YOLO Backbone) | Gamma 단독 (EXP-037) | Gamma+WB (EXP-038) | 변화 |
|---|---|---|---|
| Overall Top-1 | 0.7429 | 0.7492 | +0.63pp |
| warm_cast Top-1 | 0.9778 | 1.0000 | +2.22pp |
| cool_cast Top-1 | 0.9556 | 1.0000 | +4.44pp |
| low_light Top-1 | 0.3333 | 0.3778 | +4.44pp |
| strong_light Top-1 | 0.6667 | 0.6444 | -2.22pp |
| occlusion Top-1 | 0.7778 | 0.7333 | -4.44pp |
| bbox_jitter Top-1 | 0.6222 | 0.6000 | -2.22pp |
| flip Top-1 | 0.8667 | 0.8889 | +2.22pp |

| 조건 (Baseline=Color Histogram) | Gamma 단독 (EXP-037) | Gamma+WB (EXP-038) | 변화 |
|---|---|---|---|
| Overall Top-1 | 0.5111 | 0.6857 | +17.46pp |
| warm_cast Top-1 | 0.1111 | 0.9111 | **+80.00pp** |
| cool_cast Top-1 | 0.0667 | 0.8889 | **+82.22pp** |
| low_light Top-1 | 0.3556 | 0.2222 | -13.33pp |
| strong_light Top-1 | 0.2889 | 0.2222 | -6.67pp |
| occlusion Top-1 | 0.8000 | 0.6222 | **-17.78pp** |
| bbox_jitter Top-1 | 0.9556 | 0.9333 | -2.22pp |
| flip Top-1 | 1.0000 | 1.0000 | 0 |

| 조건 (Fusion 0.5/0.5) | Gamma 단독 (EXP-037) | Gamma+WB (EXP-038) | 변화 |
|---|---|---|---|
| Overall Top-1 | 0.6349 | 0.7778 | +14.29pp |
| warm_cast Top-1 | 0.3778 | 0.9556 | **+57.78pp** |
| cool_cast Top-1 | 0.3111 | 0.8889 | **+57.78pp** |
| occlusion Top-1 | 0.8889 | 0.7556 | **-13.33pp** |

원본 수치: `results/EXP-038/summary.json`, 쿼리별 예측: `results/EXP-038/
per_query_predictions_{label}.csv`.

## Failure Cases
**FC-014(신규) - Gamma-Background+WB 결합이 Re-ID Occlusion 보존 성능을 저하시킴.**
- 현상: Occlusion Variant Top-1이 Gamma+WB 적용 후 모든 방법에서 하락(Baseline -17.78pp,
  Candidate -4.44pp, Fusion -13.33pp) - 유일하게 세 방법 모두에서 일관되게 악화된 Variant.
- 재현 조건: `apply_occlusion()`이 bbox 폭 40~60%/높이 30~50% 영역을 중립 회색
  `(120,120,120)`으로 덮은 Query crop에 White Balance(Gray World)를 적용할 때.
- 원인 가설(코드로 직접 확인): White Balance는 "crop 전체 채널 평균이 중립 회색에
  가까워야 한다"는 Gray World 가정으로 게인을 계산한다. 중립 회색 Occlusion 패치 자체가
  이미 "완벽한 무채색"이라 crop의 20~30%를 차지하면 채널 평균을 인위적으로 중립 쪽으로
  끌어당겨, 같은 사람의 가려지지 않은 crop과 가려진 crop이 서로 다른 게인을 받는다(실측:
  `000000000368_0` crop에서 비가림 게인 B/G/R=[1.32, 0.92, 0.87], 가림 후 게인=[1.22,
  0.91, 0.93] - R 게인이 0.87→0.93으로 유의하게 이동).
  확인 방법: 동일 crop에 Occlusion 적용 전/후 `white_balance_gray_world()` 직전 채널
  평균과 게인을 직접 계산해 비교(본 experiment.md 작성 과정에서 재현, 코드는 세션
  스크래치 - 재현 가능).
- 검토한 대안: 없음(이번 세션은 FC-013(b) 개선 여부 검증이 목적이었고, FC-014는 그
  과정에서 발견된 부산물 - 다음 세션 과제로 등록).
- 최종 처리: FC-014로 신규 등록, production 미연결 상태 유지(아래 Decision 참고).
  Occlusion 영역을 White Balance 평균 계산에서 제외하는 방안(향후 Segmentation Mask와
  결합 시 자연스럽게 해결될 가능성 - EXP-028/029의 Mask 영역이 이미 person 픽셀만
  분리하므로)을 다음 후보로 제안.

## Analysis
가설이 **부분적으로 확인**됐다. 목표였던 warm_cast/cool_cast는 세 방법(Baseline/
Candidate/Fusion) 모두에서 매우 크게 개선됐고(Baseline은 거의 바닥 수준이던 6.7~11.1%가
88.9~91.1%로, Candidate는 이미 높았던 95.6~97.8%가 100%로), 세 가지 Overall Top-1
(Baseline/Candidate/Fusion) 전부 Gamma 단독보다 역전 없이 높아졌다 - EXP-037이 채택
근거로 쓴 것과 같은 "헤드라인 집계 지표 전부 지배" 기준을 통과한다.

그러나 "평균 개선만 보고 끝내지 않는다"는 이 프로젝트의 핵심 원칙대로 Variant별로
슬라이싱하면, 의도하지 않은 대가가 분명히 드러난다: (1) Baseline에서 low_light/
strong_light가 추가로 악화됐다(Gamma 단독 적용만으로도 이미 발생했던 EXP-030/037
patterns과 유사하게, WB가 노출이 심하게 왜곡된 crop의 색상 신호를 더 교란시키는
것으로 추정 - crop 전체가 이미 거의 단색에 가까운 상태에서 Gray World 게인이 극단적
으로 작동할 수 있음, 단 이번 실험에서 직접 검증하지는 않음, 추측으로 단정하지 않음),
(2) 세 방법 모두 Occlusion이 일관되게 악화됐고 FC-014로 신규 등록했으며 원인을 코드
레벨에서 직접 확인했다(Gray World 가정이 중립 회색 Occlusion 패치에 오염됨).

EXP-036/037이 이미 보여준 패턴("한 조건에 최적화된 보정이 다른 조건에서 대가를
치른다")이 이번에도 반복됐다 - 다만 이번에는 그 대가(Occlusion)가 원인까지 코드
레벨에서 명확히 규명됐다는 점이 이전 사례(EXP-037의 warm_cast/cool_cast 악화는
원인 미규명 상태로 남겨짐)와 다르다.

## Decision
**Gamma-Background+WB(`gamma_background_then_white_balance`)를 Gamma-Background
단독(EXP-037)보다 선호하는 다음 후보로 채택**한다 - 세 가지 헤드라인 Overall Top-1
지표 전부에서 역전 없이 개선됐고, FC-013(b)가 겨냥한 cast 문제를 사실상 해결 수준
(88.9~100%)까지 끌어올렸다. 다만 **production(`src/reid/`, `run_full_pipeline.py`)
에는 여전히 연결하지 않는다** - 이유: (1) n=45 합성 Query의 근본 한계(FC-013 전체가
공유하는 제약)가 해소되지 않았고, (2) 이번 실험이 새로 발견한 FC-014(Occlusion 저하)가
카지노 CCTV 환경의 핵심 시나리오(사람 간 교차, 가림)와 정면으로 겹친다 - 조명 Cast를
고치려다 이 프로젝트가 가장 중요하게 다루는 Occlusion 강건성을 희생하는 것은 아직
받아들일 수 없는 Trade-off다.

## Next Action
1. FC-014(Occlusion+WB 상호작용)를 Segmentation Mask 영역(EXP-028/029의
   `segment_largest_instance_mask`)으로 White Balance 평균 계산을 person 픽셀에만
   한정하는 방식으로 완화할 수 있는지 검토 - 중립 회색 Occlusion 패치는 person Mask
   내부에 있어 Mask만으로는 자동 해결되지 않을 수 있으므로, Mask+Occlusion 영역 제외
   로직을 실제로 설계·검증해야 한다(가정 없이 실측).
2. Baseline(Color Histogram)의 low_light/strong_light 악화 원인을 Gray World 게인
   값을 직접 로깅해 규명할 것(현재는 "추정"으로만 기록, 확정하지 않음).
3. 원본 Wide-angle CCTV 영상 접근이 가능해지면 이번 합성 Query Protocol(EXP-036/037과
   동일)을 실제 같은 사람의 여러 Track으로 교체 검증(EXP-037 Next Action (b)와 동일,
   여전히 미해결).
4. n=45 한계와 FC-014가 모두 해소되기 전까지 MetadataStore/VMS 연결은 보류.
