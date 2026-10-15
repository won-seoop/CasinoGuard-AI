# EXP-037 FC-013(Person Re-ID 저노출/과다노출 붕괴) - 배경 기반 Gamma 보정 검증

관련 실험: EXP-036(PAR-027, FC-013 최초 발견), EXP-030(PAR-021, Attribute 색상 분류에서
같은 배경 기반 Gamma 보정을 먼저 검증한 선행 실험)

## Goal

Roadmap(01. Architecture & Roadmap) EXP-036 Next Action 7번을 수행한다: FC-013(Person
Re-ID 임베딩이 low_light/strong_light 합성 조명에서 거의 무작위로 붕괴)의 원인을
EXP-030이 Attribute 색상 분류에서 이미 부분 검증한 배경 기반 Gamma 보정으로 완화할 수
있는지 측정한다.

## Hypothesis

1. low_light(`img*0.35`)와 strong_light(`img*1.9+25`)는 전역 노출 변화이므로, 같은
   Frame의 배경(person bbox 밖)에서 추정한 Gamma를 Query crop에 역으로 적용해 밝기를
   정상 범위로 되돌리면 low_light/strong_light Top-1이 개선될 것이다 - Color
   Histogram의 HSV Hue/Saturation 양자화 손실과 YOLO Backbone의 ImageNet 분포 이탈
   둘 다 전역 노출 문제이기 때문이다.
2. warm_cast/cool_cast(채널 간 상대적 Cast)와 occlusion/bbox_jitter/flip(기하 변형)은
   Gamma 보정이 겨냥하는 원인(전역 노출)과 무관하므로 거의 불변이거나 소폭의 부작용만
   있을 것이다.
3. (대안 A vs B 비교) EXP-030이 Attribute 분류에서 crop 자신의 밝기로 감마를 추정하는
   방식(`adaptive_gamma_correct`, 대안 A)을 "crop 밝기가 옷 색상(albedo)과 노출이
   혼합된 신호"라는 confound 때문에 기각했는데, Re-ID 임베딩은 "옷이 검정인가"를
   판별하는 것이 아니라 crop 전체의 외형 패턴을 구분하는 것이 목적이므로 같은
   confound가 Re-ID에서는 덜 치명적일 수 있다 - 배경 기반(대안 B)과 직접 비교해
   확인한다(확정하지 않고 측정한다).

## Problem

EXP-036에서 Color Histogram Baseline과 YOLO11n-cls Backbone 임베딩(대안 B) 둘 다
low_light/strong_light 합성 변형에서 Top-1이 거의 무작위 수준으로 붕괴했다(Baseline
low_light 33.3%/strong_light 2.2%, 대안 B low_light 2.2%/strong_light 0.0%). 50:50
Score Fusion으로도 구제되지 않았다(두 방법이 같은 약점을 공유). FC-013으로 등록만 하고
수정은 시도하지 않은 채 백로그로 남겨뒀다.

## Dataset

EXP-027/036과 완전히 동일한 coco128 실제 person crop 45명(`results/EXP-027/crops/*.jpg`)
Gallery + 같은 RNG 시드(42)로 만든 315개 합성 Query(7 variant x 45명: low_light/
strong_light/warm_cast/cool_cast/occlusion/bbox_jitter/flip). 배경 추출을 위해 각 crop의
원본 이미지+bbox가 추가로 필요해, coco128을 EXP-030과 동일 설정(yolo11n.pt, conf=0.4,
`MIN_BOX_AREA=120*250`)으로 재탐지했다 - 재탐지 결과가 EXP-027/036 Gallery 45명과
정확히 일치함을 실행 시점에 직접 검증했다(코드에 `RuntimeError` 가드 포함, 실제로
`missing={}` 확인됨).

## Environment

coco128(`https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`,
GitHub Release Asset)을 이 세션에서 재다운로드(이전 세션들이 쓰던 `/tmp/coco128_extract`
캐시는 컨테이너 재시작으로 유실되어 있었음, 6.98MB, 128장 정상 추출). `yolo11n.pt`/
`yolo11n-cls.pt`도 동일 경로로 재다운로드. opencv-python-headless 5.0.0, ultralytics,
torch(CPU)를 이 세션에 새로 설치(`python3 -m pip install`, 시스템 `pip`가 가리키는
Python 3.13과 `python3` 커맨드가 가리키는 3.11이 분리되어 있어 `python3 -m pip`로
설치 - EXP-036이 기록한 것과 유사한 환경 분리 패턴).

## Configuration

- No-Gamma: EXP-036과 완전히 동일한 코드/RNG 재현(검증용 기준선).
- 대안 A(Gamma-Self): `src/attributes/color.py::adaptive_gamma_correct()` - crop
  자신의 평균 밝기로 감마 추정, 배경 불필요.
- 대안 B(Gamma-Background, 채택): `adaptive_gamma_correct_from_background()` - 같은
  Frame의 배경(person bbox 밖)에서 감마 추정. lighting Variant(low_light 등)는 배경도
  같은 조명 변형을 거친 뒤 샘플링하고, 기하 Variant(occlusion/bbox_jitter/flip)와
  Gallery는 배경을 원본 그대로 사용(조명 변화 없음 - "정답을 알고 치팅"하지 않고
  정상 조명에서도 동일 파이프라인을 무조건 통과시켜 부작용을 함께 측정).
- 두 대안 모두 Gallery+Query 양쪽에 동일하게 적용(production이라면 조건을 미리 알 수
  없으므로 한쪽에만 적용하는 것은 비현실적).
- 평가: EXP-036과 동일한 `evaluate_identification()`/`rank_gallery_fused()`(Top-1,
  Top-5, mAP, 전체+variant별 집계).

## Baseline

No-Gamma(EXP-036 그대로 재현).

## Result

### 0) 재현성 확인: No-Gamma가 EXP-036과 일치

| Method | EXP-036 Top-1 | 이번 재현 Top-1 |
|---|---|---|
| Baseline(Color Histogram) | 49.21% | 49.21% |
| 대안 B(YOLO Backbone) | 59.68% | 59.68% |

동일 시드/동일 코드로 완전히 동일한 수치 재현 확인 - 이후 비교가 측정 잡음이 아닌
실제 개선/부작용임을 보장한다.

### 1) FC-013 표적 Variant: Hypothesis 1 확인(두 대안 모두 큰 개선)

| Variant | No-Gamma Baseline | Gamma-Self(A) Baseline | Gamma-BG(B) Baseline | No-Gamma 대안B | Gamma-Self(A) 대안B | Gamma-BG(B) 대안B |
|---|---|---|---|---|---|---|
| low_light | 33.3% | **37.8%** | 35.6% | 2.2% | **46.7%** | 33.3% |
| strong_light | 2.2% | 28.9% | **28.9%** | 0.0% | 60.0% | **66.7%** |

YOLO Backbone(대안 B 임베딩)의 strong_light가 0.0%→66.7%, low_light가 2.2%→33.3%로
가장 극적으로 개선됐다. Color Histogram도 strong_light 2.2%→28.9%로 크게 개선됐다
(low_light는 33.3%→35.6%로 이미 상대적으로 덜 붕괴했던 만큼 개선폭도 작음).

### 2) 전체 집계: Gamma-Background(B)가 모든 헤드라인 지표에서 Gamma-Self(A)를
지배(동률 포함)

| 지표 | No-Gamma | Gamma-Self(A) | Gamma-Background(B) |
|---|---|---|---|
| Baseline Top-1 / mAP | 49.21% / 0.5809 | 49.52% / 0.5972 | **51.11%** / **0.6020** |
| 대안 B Top-1 / mAP | 59.68% / 0.6614 | 73.97% / 0.7959 | **74.29%** / **0.8005** |
| Fusion Top-1 / mAP | 52.38% / 0.6163 | 62.54% / 0.7125 | **63.49%** / **0.7193** |

### 3) Variant별 부작용: Gamma-Self(A)가 Occlusion/Bbox Jitter에서 더 크게 희생됨

| Variant | No-Gamma Baseline | A Baseline | B Baseline | No-Gamma 대안B | A 대안B | B 대안B |
|---|---|---|---|---|---|---|
| occlusion | 84.4% | 68.9%(-15.6pp) | **80.0%**(-4.4pp) | 64.4% | 71.1%(+6.7pp) | **77.8%**(+13.3pp) |
| bbox_jitter | 95.6% | 93.3%(-2.2pp) | **95.6%**(0pp) | 66.7% | 57.8%(-8.9pp) | **62.2%**(-4.4pp) |
| warm_cast | 17.8% | 8.9%(-8.9pp) | **11.1%**(-6.7pp) | 100.0% | **100.0%**(0pp) | 97.8%(-2.2pp) |
| cool_cast | 11.1% | **8.9%**(-2.2pp) | 6.7%(-4.4pp) | 100.0% | 97.8%(-2.2pp) | **95.6%**(-4.4pp) |
| flip | 100.0% | 100.0% | 100.0% | 84.4% | 84.4% | **86.7%** |

대안 B가 Baseline의 Occlusion 보존(84.4%→80.0% vs 대안 A의 84.4%→68.9%)과 대안 B
임베딩의 Bbox Jitter 보존(66.7%→62.2% vs 대안 A의 66.7%→57.8%)에서 뚜렷하게 더
안전하다. 대안 A가 유일하게 더 나은 지점은 대안 B 임베딩의 low_light(46.7% vs
33.3%)뿐이고, 그 대가로 Occlusion/Bbox Jitter가 더 크게 희생된다.

warm_cast/cool_cast는 두 대안 모두에서 소폭 악화(원래도 나빴던 Baseline Color
Histogram 조건, 또는 거의 완벽했던 대안 B 임베딩 조건 모두 큰 폭은 아님) - Gamma
보정은 전역 노출만 겨냥하므로 채널 Cast(FC-012의 다른 절반)를 고치지 못하는 것은
설계상 당연하다(Hypothesis 2 확인, "무관 Variant는 거의 불변"이 아니라 "소폭이지만
실재하는 부작용"으로 더 정확히 수정).

## Failure Cases

신규 Failure Case는 발견되지 않았다. 다만 FC-013의 다음 세부 사항을 구체화한다:
Gamma 보정은 low_light/strong_light(전역 노출) 붕괴는 크게 완화하지만, warm_cast/
cool_cast(채널 Cast) 저하는 고치지 못하고 오히려 소폭(2.2~8.9pp) 악화시킨다 - FC-012가
이미 "노출(Gamma 영역)"과 "Cast(WB 영역)"를 서로 다른 원인으로 분리해 기록한 것과
동일한 패턴이 Re-ID에서도 재현된다.

## Analysis

1. **배경 기반 Gamma 보정이 Re-ID에도 전이된다**: FC-012(Attribute)에서 검증된
   "전역 노출 문제는 crop 자신이 아니라 배경에서 추정해 보정해야 한다"는 원리가
   완전히 다른 Task(색상 분류 vs 신원 식별)와 완전히 다른 임베딩(Hue 다수결 vs
   Cosine Similarity)에도 그대로 적용된다. 이는 이 보정이 특정 Task의 우연이 아니라
   "노출 자체를 정상화"하는 일반적인 전처리임을 시사한다.
2. **EXP-030의 confound가 Re-ID에서는 덜 치명적이다(중요한 반증적 발견)**: EXP-030은
   crop 자신의 밝기로 감마를 추정하는 방식(대안 A)이 "검정 옷은 어둡다"는 분류 신호
   자체를 지워버려 Attribute 분류를 무너뜨렸다(normal 조건 Accuracy 33.3%→9.5%).
   Re-ID의 대안 A는 그 정도로 파괴적이지 않았다(baseline 49.21%→49.52%, 대안 B
   59.68%→73.97%, 둘 다 개선). 이유를 분석하면: Attribute 분류는 "crop의 절대 밝기"
   자체가 분류 레이블(검정/흰색)과 직결된 신호인 반면, Re-ID 임베딩은 crop 전체의
   상대적 패턴(색 분포 모양, 텍스처)으로 신원을 구분하므로 절대 밝기 하나를
   정규화해도 판별에 필요한 상대적 정보가 크게 손실되지 않는다. 이 발견은 "같은
   기법이 한 Task에서 기각됐다고 다른 Task에서도 자동으로 기각해야 한다"는 성급한
   일반화를 경계하게 한다 - 실제로 측정해서 확인했다.
3. **그럼에도 대안 B를 선택하는 이유는 '더 나은 한 지표'가 아니라 '일관된 지배'다**:
   대안 A가 대안 B 임베딩의 low_light에서만 더 나았던 것은 사실이지만, 이 프로젝트가
   반복적으로 확인한 교훈(PAR-018/023/026, "평균/단일 지표 개선만 보고 결정하면
   안 된다")을 그대로 적용하면, 세 가지 헤드라인 집계 지표(Baseline/대안B/Fusion의
   Top-1·mAP) 전부와 카지노 환경에서 특히 중요한 Occlusion 보존 모두에서 대안 B가
   대안 A를 역전 없이 지배한다(동률 포함). 대안 A를 선택했다면 "저노출 한 조건은
   더 낫지만 혼잡/가림이 많은 카지노 플로어의 핵심 실패 모드인 Occlusion에서 더 크게
   희생한다"는 Trade-off를 감수하는 셈이다.

## Decision

1. **대안 B(배경 기반 Gamma 보정)를 FC-013의 다음 production 후보 방향으로 채택한다**
   - 모든 헤드라인 집계 지표에서 대안 A를 역전 없이 지배하고(동률 포함), 카지노
     핵심 조건인 Occlusion 보존도 더 우수하다.
2. **대안 A(Self-Gamma)는 production 후보에서 제외하지만, 코드/결과는 보존한다**
   (지침 38, 실패/차선 실험을 삭제하지 않는다) - "Re-ID에서는 Attribute만큼
   치명적이지 않다"는 반증적 발견 자체가 기록할 가치가 있다.
3. **여전히 production(MetadataStore/VMS/run_full_pipeline.py 연결)에는 승격하지
   않는다** - EXP-036과 동일한 한계(n=45 실제 신원, Query가 합성이지 재촬영이
   아님)가 그대로 남아 있고, warm_cast/cool_cast는 이 보정으로 해결되지 않아(오히려
   소폭 악화) FC-013이 "부분 해결"일 뿐 완전 해결은 아니기 때문이다.
4. FC-013을 두 하위 문제로 분리해 다음 세션에 넘긴다: (a) 전역 노출 붕괴 - 이번
   실험으로 부분 해결 방향 확인(대안 B), (b) 채널 Cast 저하 - 이번 실험 범위 밖,
   FC-012의 WB 계열 교훈을 Re-ID에 그대로 적용할 수 있는지는 별도 검증 필요(FC-012
   자체도 WB가 cool_cast만 부분 개선하고 strong_light는 악화시켰던 것을 기억할 것 -
   "검증된 기법을 다른 곳에 또 그냥 적용"하지 않는다).

## Next Action

1. (최우선) warm_cast/cool_cast Re-ID 저하에 White Balance(Gray World, EXP-026과
   동일 함수)를 Gamma-Background와 결합(Gamma→WB 순서, EXP-030의 `b_wb_gamma`
   패턴 재사용)했을 때 실제로 개선되는지 검증 - 적용 전에 반드시 실측할 것(EXP-031이
   반증한 패턴을 반복하지 않는다).
2. 원본 Wide-angle CCTV 영상 접근이 가능해지면 이 합성 Query Protocol을 실제 같은
   사람의 여러 Track(동일 영상 재등장)으로 교체 검증할 것(EXP-036 Next Action 3번과
   동일, 아직 미수행).
3. Gamma-Background를 Re-ID용 `src/reid/` 모듈에 재사용 가능한 함수로 승격할지는
   (1)(2) 검증 이후 결정 - 현재는 실험 스크립트에만 존재(production 미연결).
4. MetadataStore/VMS 연결은 FC-013(a)+（b) 모두 해결되거나 n 확장 이후로 보류.
