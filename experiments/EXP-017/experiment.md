# EXP-017 Adaptive Recording Baseline (지침 23번)

## Goal
CLAUDE.md 지침 23(Adaptive Recording)의 Baseline(항상 30fps High Quality)과 개선안
(No Person→Low FPS / Person Detected→Normal FPS / Security Event→High FPS·High Quality
+ Circular Buffer 기반 Event Clip -10s~+10s)을 실제로 구현하고, 저장 용량/CPU 사용량/
Event Clip 완전성을 실측 비교한다.

## Hypothesis
1. 항상 모든 프레임을 고화질로 저장하는 Baseline 대비, 사람이 없는 구간(IDLE)의 Frame
   Skip+저해상도만으로도 연속 녹화(Idle+Active Stream) 저장 용량이 유의미하게 줄어들 것이다.
2. Event(Intrusion) 간격이 Post-Roll 시간보다 짧으면, Event Clip을 Event마다 독립적으로
   만드는 방식(대안 A)은 겹치는 구간을 중복 저장해 병합 방식(대안 B)보다 총 저장 프레임
   수가 더 많을 것이다.

## Problem
지침 23은 아직 구현되지 않은 Stretch Goal이었다. Notion 01. Architecture & Roadmap의
다음 Action(Person Re-ID 또는 Adaptive Recording 중 선택)에서, 이 원격 자동화 세션은
egress 정책상 Wikimedia 등 외부 도메인이 차단되어 있고 Person Re-ID의 Pretrained
Weight(torchreid 계열은 보통 Google Drive/외부 서버에서 받음) 접근성이 불확실하므로,
외부 데이터/가중치가 전혀 필요 없는 Adaptive Recording을 이번 세션의 작업으로 선택했다.

## Dataset
EXP-010/014/015/016과 동일한 Blocker(재확인 완료: `commons.wikimedia.org`,
`raw.githubusercontent.com` 모두 CONNECT 403, `pypi.org`/`github.com/*/releases`는 정상)로
기존 Wikimedia 영상을 사용할 수 없어, ultralytics 패키지에 내장된 실제 사진 `bus.jpg`
(1080x810, 실제 사람 4명 이상)에 Pan/Jitter를 적용한 3-Phase 합성 시나리오(900프레임,
가정 25fps=36초)를 사용했다. Detection은 매 프레임 실제 YOLO11n 추론 결과다.

- Phase 1 [0,300) IDLE: 사람이 없는 배경(이미지 상단 15% 하늘/지붕 영역을 타일링)만 촬영
- Phase 2 [300,600) NORMAL: bus.jpg 전체 + Pan(±12px)/밝기 Jitter(±3%), ROI 밖
- Phase 3 [600,900) EVENT: 동일 + 추가 수평 이동으로 사람 무리를 ROI 안팎으로 두 번 이동

## Environment
- Remote automated session, egress 제한 (위 Dataset 항목 참고)
- Python 3.11.15 (신규 `.venv`), opencv-python-headless, ultralytics 8.4.x, numpy
- YOLO11n pretrained (conf=0.4, imgsz=640), ByteTrack (bytetrack.yaml, persist=True)
- CPU 전용 (원격 컨테이너, GPU 없음)

## Configuration
```yaml
n_frames: 900
assumed_fps: 25.0
pre_roll_sec: 10.0
post_roll_sec: 10.0        # -> pre/post_roll_frames = 250
idle_frame_stride: 5       # Low FPS: 5프레임 중 1개만 저장 (5fps 상당)
idle_scale: 0.5            # Low FPS 구간 해상도 절반
roi_polygon_x: [430, 620]  # 중앙 대역 (아래 Failure Cases 참고 — 처음엔 오른쪽 절반이었음)
event_phase_extra_dx_amplitude: 300
event_phase_extra_dx_period: 150   # 프레임, Phase 3(300프레임)에 약 2주기
merge_gap_sec: 0.0          # Event Clip 병합 기준 (겹치거나 맞닿으면 병합)
```

## Baseline
`baseline_always_high_quality.mp4`: 900프레임 전부를 원본 해상도(810x1080)·25fps로
빠짐없이 저장. Event Clip 개념 없음.

## Result

### 1) Tier 분류 (compute_tier_sequence, 실제 Detection+ROI 결과 기반)

| Tier | Frame 수 |
|---|---|
| IDLE (사람 없음) | 301 |
| NORMAL (사람 있음, Event 없음) | 338 |
| EVENT (Intrusion 활성 또는 Post-Roll 이내) | 261 |

Intrusion Event는 실제로 2회 발생했다: `[639,714]`(76프레임), `[789,863]`(75프레임).
두 Event 사이 간격은 74프레임(≈3초)으로 Post-Roll(250프레임=10초)보다 짧다.

### 2) 저장 용량 (Baseline vs Adaptive 연속 녹화)

| 항목 | Baseline | Adaptive |
|---|---|---|
| 파일 | `baseline_always_high_quality.mp4` | `adaptive_idle_low_fps.mp4` + `adaptive_active_stream.mp4` |
| 총 크기 | 22,836,795 bytes (21.8MB) | 1,098,527 + 17,462,442 = 18,561,969 bytes (17.7MB) |
| 기록 Frame 수 | 900 | 61 (idle) + 599 (active) = 660 |
| **저장 절감률** | - | **18.7%** |

### 3) Event Clip: 대안 A(병합 없음) vs 대안 B(병합, 채택)

| 방식 | Clip 개수 | 총 저장 Frame 수 |
|---|---|---|
| 대안 A: Event마다 독립 Clip (병합 없음) | 2 | 872 |
| 대안 B: 겹치거나 인접한 Window 병합 (채택) | 1 | 511 |

대안 A로 계획했다면 두 Clip의 Window가 `[389,899]`와 `[539,899]`로 완전히 겹쳐(두
번째 Clip이 첫 번째 Clip에 포함), 361프레임(41.4%)이 두 파일에 중복 저장되었을 것이다.
대안 B(채택)는 이 두 Window를 하나(`[389,899]`, 511프레임)로 병합해 중복을 0으로 만든다.

### 4) Event Clip 완전성 (실제로 저장된 Clip 파일 검증)

| clip_id | 계획된 구간 | 기대 Frame 수 | 실제 저장 Frame 수 | 누락 |
|---|---|---|---|---|
| 0 | [389, 899] | 511 | 511 | 0 |

`event_clip_0.mp4` 파일 크기 15,173,104 bytes. 하나도 누락되지 않았다.

### 5) CPU 사용량 (getrusage, encode+write 구간만)

| | Baseline | Adaptive |
|---|---|---|
| CPU Time (user+sys) | 10.16s | 11.20s |

Adaptive가 오히려 더 높다 — 아래 Analysis 3번 참고.

### 6) "Event Clip 포함" 총 저장량 (반직관적 결과, 있는 그대로 기록)

| | Baseline | Adaptive (연속 녹화만) | Adaptive (Event Clip 포함) |
|---|---|---|---|
| 총 저장량 | 22,836,795 | 18,560,969 (-18.7%) | 33,734,073 (**+47.7%**) |

## Failure Cases

**실험 설계 버그 (ROI 위치)**: 최초 시도에서는 ROI를 "오른쪽 절반(x>405)"으로 설계했다.
실행 결과 Phase 2(사람 있음, Event 없음으로 의도)부터 이미 event_active=True가 되어
NORMAL Tier가 **0프레임**만 나왔다(재현: `person_present=599, event_active_frame_count=584,
tier_counts={idle:301, normal:0, event:599}`). 원인을 bbox 좌표를 직접 찍어 확인한 결과,
bus.jpg에는 원래 x≈734 지점(프레임 폭 810 기준 오른쪽)에 사람이 서 있어 Pan/Jitter만으로도
`x>405` ROI 안에 처음부터 들어가 있었다(코드 진단 스크립트로 4개 인물의 bottom-center
x좌표가 각각 ≈30, ≈140, ≈280, ≈734임을 직접 확인). 이는 시스템 로직 버그가 아니라
실험 시나리오(ROI 위치) 설계 실수였다 — 실제 CCTV 운영에서도 "ROI를 그릴 때 화면 안에
이미 있는 정지 물체/사람 위치를 확인하지 않으면 최초 프레임부터 오탐 Event가 발생한다"는
현실적인 교훈과 같은 종류의 문제다. ROI를 아무도 서 있지 않은 중앙 대역(x=[430,620])으로
재설계하고, 그 대역으로 사람 무리를 밀어 넣도록 진폭(300px, 주기 150프레임)을 다시 설계해
해결했다(재현 스크립트로 사전 검증 후 반영, 위 Result 1번 수치가 최종 버전).

**ByteTrack ID 재사용/짧은 Track 파편화**: 위 실험 설계 버그를 조사하는 과정에서 사람이
ROI 경계 근처를 오갈 때 ByteTrack이 짧은 시간 안에 새 ID를 여러 번 부여하는 현상을
관찰했다(예: 오른쪽 절반 ROI 버전에서 `id 8,10,11,18,19`가 30초 안에 번갈아 등장). 이는
EXP-016의 Failure Case("feeder_confirmed_track_count > total_unique_tracks")와 같은
계열의 문제로 보이나, 이번 실험의 목적(Adaptive Recording 저장 정책 검증)과는 직접
관련이 없어 별도로 깊게 파지 않고 기록만 해둔다.

## Analysis

1. **연속 녹화 저장 절감(18.7%)은 실측으로 확인되었다.** IDLE 구간(301프레임, 전체의
   33.4%)을 Frame Skip(1/5)+절반 해상도로 줄인 것만으로도 이 정도 절감이 나왔다. 이
   데모 영상은 IDLE:NORMAL:EVENT 비율이 대략 1:1.1:0.9로 비교적 균형 잡혀 있어, 실제
   카지노처럼 "사람이 거의 항상 있는" 환경에서는 절감폭이 이보다 작고, 반대로 "야간에
   대부분 비어 있는" 카메라에서는 절감폭이 훨씬 클 것으로 예상된다(이번 실험에서 직접
   검증하지는 못함 — Next Action).

2. **Event Clip 병합(대안 B)의 효과는 가설대로 뚜렷하게 확인되었다.** 두 Intrusion Event
   간격(74프레임)이 Post-Roll(250프레임)보다 짧다는 조건에서, 병합하지 않으면(대안 A)
   두 Clip Window가 완전히 겹쳐 361프레임(41.4%)이 중복 저장되었을 것을 `plan_event_clips`
   로 직접 계산해 확인했다. 병합(대안 B)은 이 중복을 정확히 0으로 만든다.

3. **CPU 시간이 Adaptive가 더 높게 나온 것은 예상과 반대였다 — 있는 그대로 기록한다
   (지침 38).** 원인으로 추정되는 것은 (1) Adaptive 경로가 IDLE Tier에서 매번
   `cv2.resize()` 호출을 추가로 수행하고, (2) Event Clip Writer를 별도로 열어 511개
   프레임을 한 번 더 인코딩하기 때문에 "쓰기 호출 자체의 총 횟수"는 Baseline(900회)보다
   Adaptive가 오히려 많다(idle 61 + active 599 + clip 511 = 1,171회)는 것이다. 즉 이번
   구현은 저장 "용량"은 줄이지만 저장 "횟수"(따라서 인코딩 CPU 비용)는 늘어날 수 있다 —
   Event Clip을 매번 새로 인코딩하지 않고 이미 인코딩된 Active Stream의 프레임을 그대로
   복사(또는 참조)하는 방식이 실제 시스템에서는 더 합리적일 수 있음을 시사한다(Next Action).

4. **"Event Clip 포함 총 저장량"이 오히려 Baseline보다 47.7% 많아진 것은 이 데모의
   시간 축이 극단적으로 압축되어 있기 때문이다 — 이 결과를 숨기지 않고 그대로 기록한다.**
   Pre/Post-Roll(각 10초)은 36초짜리 데모 영상 전체 길이의 절반 이상을 차지한다. 실제
   카지노 CCTV처럼 한 카메라가 몇 시간~며칠 단위로 녹화되는 환경에서는 10초 Pre/Post-Roll이
   전체 녹화 시간의 극히 일부에 불과해 이런 역전이 일어나지 않을 것으로 예상되지만, 이번
   실험에서 그 조건까지 직접 검증하지는 못했다(Next Action). 이 결과는 "Adaptive Recording이
   항상 저장 공간을 줄여준다"는 성급한 결론을 그대로 받아들이면 안 되고, Pre/Post-Roll
   설정값을 실제 운영 환경의 시간 축에 맞춰 튜닝해야 한다는 실용적인 교훈을 준다.

## Decision

- **연속 녹화 정책**: Baseline(항상 고화질)을 유지하는 대신, IDLE Tier에서 Frame
  Skip(1/5)+저해상도(0.5x)를 적용하는 방식(`compute_tier_sequence` + `should_keep_idle_frame`)을
  채택한다. 대안(예: IDLE Tier에서 아예 녹화를 끄는 방식)도 검토했으나, 완전히 끄면
  "사람이 없다고 판단했지만 실제로는 Detection이 놓친 경우"를 사후에 전혀 검증할 수
  없어(Robustness 관점에서 위험) Frame Skip 방식을 선택했다.
- **Event Clip 계획**: `plan_event_clips(..., merge_gap_frames=0)`으로 겹치거나 맞닿은
  Event Window를 병합하는 대안 B를 채택한다. `merge_gap_frames`를 매우 작은 음수로 주면
  대안 A(병합 없음)를 그대로 재현할 수 있어, 두 방식을 같은 코드로 비교 가능하게 했다.
  → **PAR-008로 정리** (실제 문제 재현 + 2가지 대안 비교 + 단위 테스트 + 통합 실험
  실측 검증 + 개선 수치 확보).
- **Pre/Post-Roll 값**: 지침 23의 권장값(±10초)을 그대로 사용했다. 이번 실험에서 "Event
  Clip 포함 총 저장량"이 역전되는 것을 실측으로 확인했으므로, 실제 운영 환경(장시간
  녹화)에서 재검증하기 전까지는 이 값을 프로덕션 기본값으로 확정하지 않고 Config로 남긴다.

## Next Action
- 더 긴 합성 영상(예: 10분 이상, EXP-010 Long Running Test 데이터 재활용)으로 IDLE
  비율이 더 크거나 작은 경우, Event 발생 빈도가 다른 경우에 저장 절감률이 어떻게
  달라지는지 재검증할 것.
- Event Clip을 별도로 재인코딩하지 않고 이미 인코딩된 연속 스트림에서 프레임 범위를
  복사(컨테이너 레벨 remux 등)하는 방식으로 바꿔 CPU 비용을 줄일 수 있는지 검토할 것.
- 실제 다양한 군중 영상(Wikimedia 접근 복구 후) 및 FC-002(배경 밀집 군중 Detection 실패)
  해소 후, 사람이 실제로 훨씬 많은 환경에서 IDLE:NORMAL:EVENT 비율이 어떻게 달라지는지
  교차 검증.
- 로컬 세션의 EXP-011~013 Git 병합은 이번 세션에서도 수행하지 않음(원격 세션은
  origin/master 위에서만 작업 가능 — Notion 01. Architecture & Roadmap 다음 Action 1번 참고).
