# EXP-022 Circular Buffer Tier별 JPEG Quality 차등 적용

관련 실험: EXP-021 (Circular Buffer 압축 청크 버퍼링, PAR-012), EXP-020 (Circular Buffer Baseline, PAR-011), EXP-019 (Event Clip Hybrid, FC-008, PAR-010)

## Goal
EXP-021 Next Action #1("Tier별 quality 차등 적용")을 실제로 구현하고, 그 절감이 공짜인지
검증한다. 구체적으로: (1) IDLE/NORMAL/EVENT Tier에 다른 JPEG quality를 적용하면 EXP-021의
Uniform quality=85 Baseline 대비 메모리를 추가로 절감할 수 있는가, (2) 그 절감이 실제
Pre-Roll Event Clip의 화질에 어떤 대가를 요구하는가.

## Hypothesis
1. 카메라가 하루 대부분(이 시나리오 기준 1,600프레임 중 1,201프레임=75.1%) IDLE Tier이므로,
   IDLE quality를 낮추면 Uniform 85 대비 버퍼 메모리를 추가로 절감할 수 있을 것이다.
2. 그러나 이 프로젝트가 이미 FC-008(EXP-019)에서 확인한 "조용하다가 갑자기" 패턴 때문에,
   실제 Pre-Roll 시나리오(trigger_frame=1339, window=[1089,1339])는 창문의 상당 부분이
   IDLE Tier와 겹칠 것이다 — 이 경우 낮아진 IDLE quality가 그대로 Event Clip에 남아 화질을
   떨어뜨릴 것이다.
3. EVENT quality만 높게 유지하면(예: 95), 실제 사고가 찍힌 프레임 자체의 화질은 Uniform 85
   Baseline보다 오히려 더 좋아지면서도 전체 메모리는 절감될 것이다 — 반면 quality를
   전체에 균일하게 낮추는 방식(Uniform 50)은 그 중요한 EVENT 프레임까지 함께 깎아먹을
   것이다.

## Problem
EXP-021은 quality를 전체 프레임에 고정값(85)으로만 적용했고, Next Action에 "IDLE/NORMAL/
EVENT Tier에 따라 quality를 다르게 적용해 메모리와 화질을 Tier별로 다르게 관리"를 다음
과제로 남겼다. 이 실험은 그 과제를 실제로 구현하고, 구현 과정에서 자연스럽게 드러난 진짜
문제 — Tier 경계를 넘나드는 Pre-Roll 창(FC-008과 동일 구조)에서 "IDLE quality를 얼마나
낮춰도 되는가"라는 질문 — 를 실측으로 답한다.

## Dataset
EXP-019/020/021과 완전히 동일한 egress 제약(외부 도메인 차단, pypi/GitHub Release Assets만
허용) 아래, `pip download --no-deps ultralytics`로 wheel(1.4MB)만 받아 `ultralytics/
assets/bus.jpg`를 추출해 `data/raw/bus.jpg`에 저장했다(공개 오픈소스 샘플 이미지, 지침 8
원칙 부합). 실제 YOLO 추론은 다시 하지 않고 EXP-019가 캐시해 둔
`results/EXP-019/detection_cache.json`(person_present/event_active, 1,600프레임)을
재사용했다.

## Environment
- 원격 자동화 세션, `.venv311`(Python 3.11.15) 신규 생성, opencv-python-headless,
  numpy, pytest.
- 프레임 810×1080 BGR uint8.

## Configuration
```yaml
capacity: 301  # EXP-021 "production_with_margin"과 동일
trigger_frame: 1339
window: [1089, 1339]  # 251 frames, EXP-019/020/021과 동일 시나리오
scenarios:
  baseline_uniform85:  {idle: 85, normal: 85, event: 85}   # EXP-021 Baseline
  alt_a_uniform50:     {idle: 50, normal: 50, event: 50}   # 균일 저quality (대안 A)
  tier_diff_idle30:    {idle: 30, normal: 75, event: 95}
  tier_diff_idle50:    {idle: 50, normal: 75, event: 95}
  tier_diff_idle60:    {idle: 60, normal: 75, event: 95}
  tier_diff_idle70:    {idle: 70, normal: 75, event: 95}
```

## Baseline
EXP-021의 Uniform quality=85 Circular Buffer(capacity=301에서 72.47MB, 전체 픽셀 MAE
2.61)를 그대로 Baseline으로 삼는다.

## Result

### 0) 사전 확인: Pre-Roll 창문의 실제 Tier 구성
`compute_tier_sequence(person_present, event_active, post_roll_frames=250)`로 계산한
결과, `window=[1089,1339]`(251프레임)의 구성은 **IDLE 112프레임(44.6%) / NORMAL 138프레임
(55.0%) / EVENT 1프레임(0.4%, trigger 프레임 자신)** 이었다 — 가설 2가 예측한 대로, 이
프로젝트의 실제 Pre-Roll 시나리오는 창문의 거의 절반이 IDLE Tier다.

### 1) Tier × Quality별 버퍼 메모리 (`results/EXP-022/tier_quality_sweep.csv`)

| Scenario | IDLE q | NORMAL q | EVENT q | 버퍼(IDLE 정상상태) | 버퍼(trigger 시점) | 메모리 절감(trigger 기준) |
|---|---|---|---|---|---|---|
| baseline_uniform85 | 85 | 85 | 85 | 65.04 MB | 72.47 MB | 0% (기준) |
| alt_a_uniform50 | 50 | 50 | 50 | 33.72 MB | 38.07 MB | 47.5% |
| tier_diff_idle30 | 30 | 75 | 95 | 24.68 MB | 42.45 MB | 41.4% |
| tier_diff_idle50 | 50 | 75 | 95 | 33.72 MB | 47.31 MB | 34.7% |
| tier_diff_idle60 | 60 | 75 | 95 | 38.31 MB | 49.77 MB | 31.3% |
| tier_diff_idle70 | 70 | 75 | 95 | 45.52 MB | 53.65 MB | 26.0% |

(baseline_uniform85의 72.47MB/alt_a_uniform50의 38.07MB는 EXP-021의 capacity=301, q85/q50
결과와 정확히 일치 — 구현이 EXP-021과 동일한 값을 재현함을 교차 검증했다.)

카메라가 가장 오래 머무는 상태인 "IDLE 정상상태"에서는 tier_diff_idle30이 Baseline 대비
**62.1%**(65.04→24.68MB) 절감된다 — 이 시점의 프레임은 나중에 어떤 Event의 Pre-Roll에도
포함되지 않으면 그대로 evict되어 영원히 읽히지 않는다.

### 2) 실제 Pre-Roll Event Clip의 Segment별 화질(픽셀 MAE, 0~255)

| Scenario | IDLE Segment(112f) | NORMAL Segment(138f) | **EVENT Segment(1f, 사고 프레임)** | Clip 전체 |
|---|---|---|---|---|
| baseline_uniform85 | 2.29 | 2.87 | **2.59** | 2.61 |
| alt_a_uniform50 | 4.32 | 5.50 | **5.40** | 4.97 |
| tier_diff_idle30 | 5.77 | 3.83 | **1.34** | 4.69 |
| tier_diff_idle50 | 4.32 | 3.83 | **1.34** | 4.04 |
| tier_diff_idle60 | 3.96 | 3.83 | **1.34** | 3.88 |
| tier_diff_idle70 | 3.39 | 3.83 | **1.34** | 3.63 |

가장 중요한 관찰: **alt_a_uniform50(균일 저quality)은 실제 사고가 찍힌 EVENT 프레임 자체의
화질(MAE 5.40)을 Baseline(2.59)보다 2배 이상 나쁘게 만든다.** Tier를 구분하지 않고 전체
quality를 낮추면 "가장 중요한 한 프레임"과 "평생 안 볼 수도 있는 배경 프레임"이 똑같이
희생된다. 반면 Tier 차등 방식은 IDLE quality를 얼마로 낮추든(30~70) EVENT Segment MAE가
항상 1.34로 고정되어 Baseline보다도 오히려 좋다(EVENT quality=95 고정이므로 당연하지만,
실측으로 확인).

## Failure Cases
새로 발견한 문제는 없음(EXP-021의 버퍼/코덱 계층을 그대로 재사용). 다만 최초 스크립트
버전에서 Pre-Roll 창문을 "전체 1,600프레임을 다 push한 뒤" 꺼내려다 `KeyError`(누락 1089개
프레임)로 실패했다 — capacity=301인 버퍼는 마지막 301프레임([1299,1599])만 들고 있으므로
`window=[1089,1339]`는 evict되고 없었다. `trigger_frame(1339)`에 프레임을 push한 "바로 그
순간"에 buffer에서 꺼내도록 수정해 해결했다(실험 방법론 버그, 별도 FC 등록 없음 —
`coverage()`가 사전에 바로 이 실수를 잡아준 사례이기도 하다, FC-008/EXP-020 설계 의도대로
동작함).

## Analysis
1. 가설 1(IDLE quality를 낮추면 추가 절감) 확인: IDLE 정상상태 기준 Baseline 대비 최대
   62.1%(idle=30) 추가 절감.
2. 가설 2(Pre-Roll 창이 IDLE과 겹쳐 화질 손실이 Clip에 남음) 확인: window의 44.6%가 IDLE
   Tier였고, IDLE quality를 낮출수록 IDLE Segment MAE가 그대로 악화됨(2.29→5.77 @idle=30).
3. 가설 3(EVENT quality를 별도로 지키면 사고 프레임 화질은 오히려 개선) 확인: 모든 Tier
   차등 설정에서 EVENT Segment MAE=1.34로 Baseline(2.59)보다 좋음. **alt_a_uniform50은
   반대로 EVENT Segment MAE를 5.40으로 악화시켜, "메모리 절감 폭만 보면 이득이지만 정작
   가장 보고 싶은 장면의 화질을 가장 많이 희생하는" 구조적으로 나쁜 선택임을 실측으로
   확인했다.**
4. "Overall Clip MAE" 단일 지표는 오해를 부를 수 있다는 것도 발견했다 — tier_diff_idle60의
   Clip 전체 MAE(3.88)는 Baseline(2.61)보다 나빠 보이지만, 이는 251프레임 중 44.6%를 차지하는
   IDLE Segment(배경, 사건과 무관)의 오차가 평균을 끌어올린 것일 뿐, 실제 조사에 결정적인
   EVENT Segment는 오히려 더 좋다. Segment별로 나눠 보지 않으면 "Tier 차등이 전체적으로
   더 나쁘다"는 잘못된 결론에 이를 수 있었다.

## Decision
**Tier 차등 방식(대안 B, `TierQualityConfig`)을 채택**하고, `src/recording/adaptive.py`에
`TierQualityConfig(idle_quality, normal_quality, event_quality)`를 추가했다. 기존
`FrameCircularBuffer`/`frame_codec`은 전혀 수정하지 않았다 — quality 선택은 호출자가
`quality_for_tier(tier)`로 프레임마다 고르고 그 값을 그대로 `encode_frame(frame, quality=…)`
에 넘기면 된다(EXP-021이 이미 만든 "버퍼는 quality를 모른다" 경계를 그대로 유지).

**대안 A(Uniform 저quality, 예: 50)는 기각** — 메모리 절감 폭(47.5%)은 가장 크지만, 그
절감이 EVENT 프레임(실제 사고 장면)에도 균등하게 적용되어 가장 보고 싶은 장면의 화질을
가장 많이 희생시킨다(MAE 5.40, Baseline 대비 2배 이상 악화). 카지노 사고 조사에서 "배경
CCTV 화질은 나빠도 되지만 사고 순간 화질은 절대 나빠지면 안 된다"는 우선순위와 정반대다.

Default 값은 **IDLE=60, NORMAL=75, EVENT=95**로 정한다. 근거: (1) IDLE Segment MAE(3.96)가
EXP-021이 이미 "저조명 등 다른 화질 저하 요인과 누적되면 위험할 수 있다"고 경계선으로
지목했던 Uniform quality=70의 MAE(3.86)와 거의 같은 수준으로, 그보다 더 낮추지는 않는다.
(2) IDLE 정상상태 메모리는 Baseline 대비 41.1%(65.04→38.31MB) 절감되어 실질적인 이득이
있다. (3) EVENT Segment는 항상 1.34로 Baseline보다 좋다. 이 기본값은 배포 환경(카지노
게이밍 플로어처럼 "조용한 시간에도 사전 정황이 중요한" 구역)에 따라 더 보수적으로(IDLE=70)
조정 가능하도록 `TierQualityConfig`를 그대로 노출해 둔다 — 하나의 "정답 quality"를 코드에
박아넣지 않는다.

## Next Action
- `run_full_pipeline.py`에는 Adaptive Recording/Circular Buffer/Tier Quality 모두 아직
  통합되지 않았다(EXP-017/019/020/021/022 전부 독립 실험) — Core MVP 파이프라인 통합은
  Stretch Goal 우선순위 재검토 후 진행.
- 이번 실험은 "IDLE Segment MAE"를 근거로 IDLE quality 기본값을 정했지만, 실제 사람이
  IDLE quality=60 압축 화면을 보고 "조사에 쓸 만한가"를 판단한 정성 평가는 하지 않았다
  (정량 MAE만 근거로 삼음, 한계로 기록).
- 실제 카지노와 유사한 복잡한 텍스처(카펫, 다양한 조명)에서도 이 Segment별 MAE 패턴이
  동일한지는 검증하지 못했다(FC-002 해소 후 재검증 필요, EXP-021과 동일한 한계).
- 코어당 동시 채널 수 제약(EXP-021 Analysis 2번)은 quality가 Tier마다 달라지면 인코딩
  시간도 프레임마다 달라진다 — Tier 혼합 비율에 따른 평균 동시 채널 수 재계산은 다음 과제.
