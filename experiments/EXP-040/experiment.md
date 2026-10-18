# EXP-040 Person Re-ID FC-013(a) - Gray World 게인 직접 로깅으로 Baseline low_light/strong_light 추가 악화 원인 규명

## Goal
EXP-038(Gamma-Background+White Balance)이 FC-013(b)(warm_cast/cool_cast)를 크게
개선하는 과정에서, Baseline(Color Histogram)의 low_light Top-1이 0.3556(EXP-037,
Gamma 단독)→0.2222(-13.33pp), strong_light Top-1이 0.2889→0.2222(-6.67pp)로 추가
악화됐다. 이 원인은 EXP-038/039 Next Action에 "Gray World 게인 값을 직접 로깅해
규명할 것(현재는 추정으로만 기록)"으로 두 세션째 이월돼 있었다. Roadmap(01.
Architecture & Roadmap) EXP-037 Next Action 7-(a'')를 수행해 이 추정을 실측으로
확인/반증한다.

## Hypothesis
EXP-038 Analysis가 기록한 추정: "crop 전체가 이미 거의 단색에 가까운 상태에서
Gray World 게인이 극단적으로 작동할 수 있다"(low_light/strong_light는 전역 노출
변형이라 채널 평균이 극단에 가까워지기 쉽고, white_balance_gray_world의
gain_min/gain_max(0.3~3.0) clip이 자주 발동해 과도한 보정이 들어갈 것이다).

## Problem
추정만으로는 production 연결 여부를 결정할 수 없다 - 원인이 "극단적 게인"이라면
Clip 범위를 좁히거나 극단 영역에서 WB를 건너뛰는 방식으로 고칠 수 있지만, 원인이
다르다면 그 수정은 아무 효과가 없을 것이다(EXP-031의 교훈: 검증 없이 가정하면
반증 사례를 반복한다).

## Dataset
- Gallery: coco128 실제 person crop 45명(`results/EXP-027/crops`, EXP-027/036~039와
  완전히 동일한 집합 - 실행 시 재탐지 결과로 검증됨).
- Query: EXP-036~039와 완전히 동일한 Protocol(RNG seed 42, 동일 호출 순서) - 조명
  4종(low_light/strong_light/warm_cast/cool_cast) + 기하 변형 3종(occlusion/
  bbox_jitter/flip) = 45명 x 7 variant = 315개 Query. Part 1(게인 진단)은 조명
  4종 + 변형 없음(gallery) = 5개 그룹만 사용(기하 변형은 채널 Cast와 무관해 제외).
- coco128 원본 이미지는 이번 세션에서 egress 허용 경로
  (`https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip`)로
  재다운로드함(EXP-027/035/038/039가 확인한 동일 경로).

## Environment
- 원격 자동화 세션(claude-sonnet-5). 이번 세션도 컨테이너가 완전히 새로 시작돼
  Python 패키지가 전혀 설치돼 있지 않은 상태였다 - `python3.11 -m pip install
  ultralytics opencv-python-headless fastapi uvicorn pytest httpx`로 재설치
  (ultralytics 최신, torch 2.14.1, opencv-python-headless, pytest - EXP-039와 동일
  패턴). `yolo11n.pt`/`yolo11n-cls.pt`는 ultralytics 공식 GitHub Release에서 자동
  재다운로드.

## Configuration
- `src/attributes/color.py` 신규 추가: `white_balance_gray_world_diagnostic(crop_bgr,
  gain_min=0.3, gain_max=3.0)` - `white_balance_gray_world()`와 완전히 동일한 Gray
  World 게인 계산을 수행하지만 보정된 이미지 대신 중간값(채널 평균, overall_mean,
  clip 전/후 게인, clip 발동 채널 수)을 반환한다(`region_dominant_color_diagnostic`
  과 동일한 thin-diagnostic 패턴, EXP-032). `white_balance_gray_world()`는 이 함수를
  호출하는 thin wrapper로 리팩터링했다(동작 비변경 - Regression Test로 고정,
  `TestWhiteBalanceGrayWorldDiagnostic`).
- `scripts/run_exp040_reid_wb_gain_diagnostics.py`:
  - **Part 1(게인 진단)**: EXP-038과 동일한 배경 추출 + `adaptive_gamma_correct_
    from_background` 적용 후, WB 직전에 `white_balance_gray_world_diagnostic()`을
    호출해 variant별(gallery/low_light/strong_light/warm_cast/cool_cast) 채널 게인/
    overall_mean/clip 발동 여부를 로깅.
  - **Part 2(대안 비교)**: Part 1 진단과 무관하게 EXP-038/039 Next Action이 제안한
    "극단적 게인" 프레이밍을 그대로 따르는 두 대안을 실제로 구현해 Top-1을 재측정:
    - **Alt A(Tight Clip)**: `white_balance_gray_world`의 `gain_min/gain_max`를
      (0.3,3.0)→(0.5,2.0)으로 좁혀 극단적 게인 자체를 억제.
    - **Alt B(Confidence Gate)**: 감마 보정 직후 `overall_mean`이 [40,215] 범위를
      벗어나면(거의 흑/백에 가까운 crop) WB를 건너뛰고 Gamma만 적용.

## Baseline
EXP-038의 `gamma_background_plus_wb`(Masking 없음) 수치를 그대로 기준값으로 사용
(Overall Top-1 Baseline 0.6857, low_light 0.2222, strong_light 0.2222, warm_cast
0.9111, cool_cast 0.8889 - 이 실험의 Part 2 두 대안 모두 "변화 없음"이면 이 값과
완전히 일치해야 한다).

## Result

### Part 1: Gray World 게인 진단 (WB 직전, 감마 보정 후)

| Variant | n | 감마 후 overall_mean | mean｜gain-1｜(채널 평균) | 채널별 min~max gain | Clip 발동 비율 |
|---|---|---|---|---|---|
| gallery(무변형) | 45 | 113.2 | 0.085 | 0.708~1.575 | **0.0%** |
| low_light | 45 | 98.1 | **0.057** | 0.790~1.284 | **0.0%** |
| strong_light | 45 | 143.7 | 0.104 | 0.685~1.982 | **0.0%** |
| warm_cast | 45 | 113.0 | **0.244** | 0.609~2.197 | **0.0%** |
| cool_cast | 45 | 113.5 | 0.119 | 0.654~1.812 | **0.0%** |

원본 수치: `results/EXP-040/gain_diagnostics.csv`(crop별), `results/EXP-040/
gain_diagnostics_summary.json`(variant별 집계).

### Part 2: 대안 비교 (Top-1, Baseline=Color Histogram)

| 조건 | EXP-038 원본(기준) | Alt A(Tight Clip) | Alt B(Confidence Gate) |
|---|---|---|---|
| Overall Top-1 | 0.6857 | 0.6857 (±0) | 0.6857 (±0) |
| low_light Top-1 | 0.2222 | 0.2222 (±0) | 0.2222 (±0) |
| strong_light Top-1 | 0.2222 | 0.2222 (±0) | 0.2222 (±0) |
| warm_cast Top-1 | 0.9111 | 0.9111 (±0) | 0.9111 (±0) |
| cool_cast Top-1 | 0.8889 | 0.8889 (±0) | 0.8889 (±0) |
| occlusion Top-1 | 0.6222 | 0.6222 (±0) | 0.6222 (±0) |

Candidate(YOLO Backbone)/Fusion도 모든 Variant에서 EXP-038 원본과 완전히 동일한
수치(low_light 0.3778/strong_light 0.6444/Fusion Overall 0.7778, 변화 없음). 원본
수치: `results/EXP-040/alternative_comparison.json`.

## Failure Cases
새 Failure Case는 아니다 - 이번 실험은 FC-013(a)에 이미 등록된 현상의 원인 분석이며,
그 결과 **원래의 원인 가설이 반증**됐다(아래 Analysis).

## Analysis
**Hypothesis가 명확하게 반증됐다.** 세 가지 독립적인 측정이 모두 같은 결론을
가리킨다:

1. **Clip이 단 한 번도 발동하지 않았다** - 5개 Variant x 45 crop = 225건 전부
   `n_channels_clipped=0`이다. 채널별 게인의 전체 범위(0.609~2.197)도
   `gain_min/gain_max`(0.3~3.0) 경계와 거리가 멀다. "극단적 게인"은 애초에
   일어나지 않았다.
2. **low_light가 가장 작은 게인 편차(0.057)를 보인다** - 가설대로라면 전역 노출
   변형(low_light/strong_light)이 채널 Cast 변형(warm_cast/cool_cast)보다 게인이
   더 커야 하는데, 실측은 **정반대**다: warm_cast(0.244) > cool_cast(0.119) >
   strong_light(0.104) > gallery(0.085) > low_light(0.057). 오히려 "진짜 채널
   Cast가 있는" warm_cast/cool_cast가 더 큰 게인을 받고(의도된 동작 - 그래서
   Top-1이 크게 개선됐다), "진짜 Cast가 없는" low_light/strong_light(합성 변형
   자체가 R/G/B를 동일 비율로 scale - `img*0.35`, `img*1.9+25`)는 작은 게인만
   받는다.
3. **두 대안(Alt A/B) 모두 완전한 No-op이었다** - Alt A의 더 좁은 Clip
   범위(0.5~2.0)도 실제 게인 범위(0.609~2.197, 최솟값 0.609만 0.5~2.0 밖인 1개
   채널 제외하면 대부분 범위 안)를 거의 건드리지 못했고, Alt B의 Gate
   범위([40,215])는 low_light의 overall_mean(98.1)과 strong_light의
   overall_mean(143.7) 모두를 넉넉히 포함해 Gate가 단 한 번도 발동하지 않았다 -
   그 결과 Part 2의 모든 지표가 EXP-038 원본과 소수점까지 완전히 동일했다(위 표).

**실제 메커니즘(이번 실험으로 새로 특정)**: WB가 Color Histogram Baseline을
악화시키는 것은 "게인이 극단적이기 때문"이 아니라, **low_light/strong_light에는
보정할 진짜 채널 Cast가 없는데도 WB가 crop마다 독립적으로 작은(그러나 0은 아닌)
보정을 가하기 때문**이다. 이 보정은 warm_cast/cool_cast처럼 "진짜 문제를 고치는"
것이 아니라, 사람마다 다른 옷 색상 분포에 Gray World 가정을 적용해 생기는 개인별
잡음(crop마다 다른 방향·크기의 게인)을 추가하는 것에 가깝다 - Color Histogram은
세밀한 HSV bin 경계에 의존하므로 이런 작은 crop별 잡음에도 민감하게 반응해 일부
Query(45개 중 약 6개 = 13.3pp)의 1-NN 매칭이 바뀐다. 반대로 Candidate(YOLO
Backbone)의 low_light/strong_light Top-1은 EXP-037/038/039/040 전부에서 동일한
값(0.3778/0.6444)을 유지해 이 작은 잡음에 영향받지 않는다는 점도 이 설명과
일치한다 - Deep Feature는 Low-level 색상 통계 잡음에 Color Histogram보다 훨씩
강건하다.

## Decision
**FC-013(a)의 "Baseline low_light/strong_light 추가 악화" 원인을 "극단적 게인/
Clip"에서 "불필요한(진짜 Cast가 없는 crop에 대한) 개별 보정 잡음"으로 교정한다.**
두 대안(Tight Clip, Confidence Gate)은 모두 틀린 원인 가정(극단성) 위에 설계돼
실측에서 아무 효과가 없었으므로 채택하지 않는다 - **production(`src/reid/`,
`run_full_pipeline.py`)에는 여전히 연결하지 않는다**(FC-013(a)는 계속 미해결로
유지, n=45 합성 Query 한계도 그대로 남음). 다만 `white_balance_gray_world_
diagnostic()`은 향후 "진짜 Cast 존재 여부"를 겨냥한 새 Gate를 설계할 때 재사용할
수 있도록 `src/attributes/color.py`에 유지한다.

이번 실험은 "검증 없이 가정하지 않는다"는 원칙을 그대로 보여준다 - Roadmap에
두 세션째 "추정"으로 남아 있던 설명을 실제로 측정하자 완전히 틀렸음이 드러났고,
그 틀린 설명 위에 설계된 두 "직관적인" 수정안(Clip 좁히기, 저신뢰 구간 Gate)이
가정이 깨지는 순간 자동으로 무의미해진다는 것도 함께 확인했다(둘 다 범위를 전혀
건드리지 못해 No-op).

## Next Action
1. 이번 실험이 특정한 새 메커니즘("진짜 Cast가 없는 crop에 WB를 적용하면 Color
   Histogram에 잡음만 추가된다")을 직접 겨냥하는 Gate를 설계하려면, "이 crop에
   진짜 채널 Cast가 있는가"를 판단하는 신호가 필요하다 - 이번 실험의 게인 편차
   (mean|gain-1|)가 그 신호의 첫 후보다(warm_cast 0.244 > cool_cast 0.119 >
   strong_light 0.104 > low_light 0.057, gallery 0.085). 다만 strong_light(0.104)와
   cool_cast(0.119)의 차이가 작아(0.015) 단순 Threshold로 깨끗하게 분리되지
   않을 수 있다 - 다음 세션에서 Threshold Sweep으로 분리 가능 여부를 실측 확인할
   것(가정하지 않는다).
2. 원본 Wide-angle CCTV 영상 접근이 가능해지면 이번 합성 Query Protocol(EXP-036~039
   와 동일)을 실제 같은 사람의 여러 Track으로 교체 검증(EXP-037 Next Action (b)와
   동일, 여전히 미해결).
3. FC-013(a)+(b) 모두 완전히 해결되거나 n 확장 전까지 MetadataStore/VMS 연결은
   계속 보류.
4. `white_balance_gray_world_diagnostic()`은 Attribute 분류(FC-012)의 filter 폴백
   경로 재설계(EXP-033/034)에도 같은 패턴(진단 함수로 실측 후 설계)으로 재사용할
   여지가 있는지 검토(아직 미검토, 추측 아님을 명시).
