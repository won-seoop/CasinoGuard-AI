# EXP-019 Adaptive Recording Event Clip 저장 방식 재검증 (지침 23, 27)

관련 실험: EXP-017 (Adaptive Recording Baseline, PAR-008)

## Goal
Notion 01. Architecture & Roadmap의 다음 Action "Adaptive Recording의 Next Action(더 긴
영상으로 재검증, Event Clip 재인코딩 대신 복사 방식 검토)"을 실제로 진행한다.
EXP-017에서 발견한 반직관적 결과(Event Clip을 포함하면 총 저장량이 Baseline보다
오히려 47.7% 증가)의 원인을 재인코딩 구조에서 찾고, FFmpeg Stream Copy로 대체했을 때
실제로 얼마나 개선되는지, 그리고 그 과정에서 새로 드러나는 한계는 무엇인지 측정한다.

## Hypothesis
1. Event Clip을 Active Stream에서 별도로 재인코딩하는 대신 FFmpeg Stream Copy(-c copy)로
   오려내면, 이미 인코딩된 바이트를 재사용하므로 CPU 시간이 크게 줄어들 것이다.
2. 더 긴 영상(IDLE 비중이 훨씬 큰, 더 현실적인 시나리오)에서는 Event의 Pre-Roll이
   IDLE Tier 구간까지 파고드는 경우가 생길 수 있고, 이 경우 Active Stream에는 애초에
   IDLE 프레임이 기록되지 않으므로 Stream Copy만으로는 Clip을 완전하게 만들 수 없을
   것이다.
3. Stream Copy는 Keyframe(GOP) 경계에서만 정확히 잘리므로, Active Stream의 GOP 크기가
   작을수록(Keyframe이 촘촘할수록) 잘림 정확도는 높아지지만 연속 녹화 파일 크기는
   커질 것이다(Trade-off).

## Problem
EXP-017(PAR-008)에서 Event Clip을 포함한 총 저장량이 Baseline 대비 -18.7%(연속 녹화만)에서
+47.7%(Event Clip 포함)로 역전되는 것을 확인했지만, 원인을 "재인코딩 구조 자체"로
지목만 하고 실제로 대안을 구현/검증하지는 않았다(Next Action으로 이월). 이번 실험은
그 이월된 문제를 실제로 구현하고 측정한다.

## Dataset
EXP-010/014~018과 동일한 Blocker(재확인 완료: `commons.wikimedia.org`,
`raw.githubusercontent.com`, `archive.org` 모두 CONNECT/HTTP 000, `pypi.org`는 200,
`github.com/*/releases`를 통한 `yolo11n.pt` 다운로드는 정상)로 Wikimedia 원본 영상을 쓸 수
없어, 기존과 동일하게 ultralytics 내장 `bus.jpg`(1080x810, 실제 사람 4명 이상)에
Pan/Jitter를 적용한 합성 시나리오를 사용했다. Detection은 매 프레임 실제 YOLO11n
추론 결과다.

이번에는 EXP-017(900프레임=36초)보다 78% 긴 **1,600프레임(64초 @ 25fps)** 시나리오를
새로 설계했다:

- Phase 1 `[0,1200)` **IDLE** (75%): 사람이 없는 배경만 촬영 — "대부분 비어 있는 통로
  카메라"라는, EXP-017(IDLE 33%)보다 훨씬 더 현실적인 야간/한산한 시간대 비율.
- Phase 2 `[1200,1300)` **NORMAL** (100프레임=4초): 사람이 잠깐 등장하지만 ROI 밖.
  **의도적으로 Pre-Roll(250프레임=10초)보다 짧게** 설계해, Event 발생 시 Pre-Roll이
  Phase 1(IDLE)까지 파고들도록 만들었다.
- Phase 3 `[1300,1600)` **EVENT 유발** (300프레임): EXP-017 Phase 3과 동일한 진폭(300px)·
  주기(150프레임) 수평 이동으로 사람 무리가 ROI(x=[430,620])를 두 번 침범.

## Environment
- Remote automated session, egress 제한 (위 Dataset 항목 참고)
- 신규 `.venv`, Python 3.11.15, opencv-python-headless 5.0.0, ultralytics 8.4.163, numpy 2.4.6
- **imageio-ffmpeg 0.7.x (PyPI)로 받은 정적 FFmpeg 7.0.2 바이너리(libx264 포함)** — 지침 27
  (OpenCV VideoCapture만 쓰지 않고 FFmpeg로 Codec/Encode/Event Clip을 실제로 다뤄보는 것)을
  이번 실험에서 처음 충족했다. ffprobe 바이너리는 imageio-ffmpeg에 포함되지 않아, 프레임 수
  검증은 FFmpeg로 rawvideo(bgr24) 디코딩한 출력 바이트 수를 프레임 크기로 나눠 직접 계산했다
  (`-f null` 텍스트 로그 파싱은 concat 출력 등에서 progress 라인이 리셋/중복돼 부정확함을
  실측으로 확인하고 버렸다).
- YOLO11n pretrained (conf=0.4, imgsz=640), ByteTrack (bytetrack.yaml, persist=True), CPU 전용

## Configuration
```yaml
n_frames: 1600
assumed_fps: 25.0
pre_roll_sec: 10.0
post_roll_sec: 10.0        # pre/post_roll_frames = 250
roi_polygon_x: [430, 620]  # EXP-017과 동일 (bus.jpg 실측: 아무도 서 있지 않은 중앙 대역)
gop_values_tested: [1, 25] # Active Stream 인코딩 Keyframe 간격 (1=전부 Keyframe, 25=1초당 1개)
codec: libx264 (FFmpeg)    # EXP-017의 cv2.VideoWriter(mp4v)에서 교체 — 재인코딩 vs Stream Copy를
                            # 같은 코덱으로 공정 비교하기 위함, 실제 CCTV/NVR도 대부분 H.264/H.265 사용
```

## Baseline
EXP-017에서 이미 "항상 고화질 녹화 vs Adaptive 연속 녹화"의 저장 절감(-18.7%)을
검증했으므로 이번 실험에서 반복하지 않는다. 이번 실험의 Baseline은 **EXP-017과 동일한
방식(Event Clip 전체 구간을 원본 프레임에서 처음부터 다시 인코딩)** 이며, 아래에서는
"방식 A(전체 재인코딩)"로 표기한다.

## Result

### 1) Tier 분류 및 Event Clip 계획 (실제 Detection+ROI 결과 기반)

| Tier | Frame 수 | 비율 |
|---|---|---|
| IDLE | 1,201 | 75.1% |
| NORMAL | 138 | 8.6% |
| EVENT | 261 | 16.3% |

Intrusion Event는 실제로 2회 발생했다: `[1339,1414]`(76프레임), `[1490,1564]`(75프레임).
간격 75프레임(3초) < Post-Roll(250프레임=10초) → EXP-017과 동일하게 **1개의 Clip으로
병합**됨(`plan_event_clips`, 가설과 다른 새 코드 경로가 아니라 기존 로직 그대로 재확인).

**Event Clip 계획**: `start_frame=1089, end_frame=1599` (511프레임).

### 2) 새로 발견한 경계 사례: Pre-Roll이 IDLE Tier까지 파고듦

`clip_window_all_active(tiers, 1089, 1599)` → **False**.
Clip 앞쪽 **112프레임(511프레임 중 21.9%)** 이 IDLE Tier(`[1089,1200]`)에 속한다 — 이
구간은 Active Stream에 애초에 기록되지 않는다(IDLE 프레임은 Active Stream 밖, `idle_low_fps`
스트림에 저해상도로만 존재). EXP-017의 900프레임 시나리오(NORMAL 구간이 Pre-Roll보다
길어 이 경계에 걸리지 않음)에서는 나타나지 않았던, **"더 긴/더 현실적인(IDLE 비중이 큰)
영상"에서만 드러나는 새로운 실패 유형**이다.

### 3) Event Clip 생성 3가지 방식 비교 (GOP=25, 1초당 Keyframe 1개 — 실무에 가까운 설정)

| 방식 | CPU 시간 | Wall 시간 | 파일 크기 | 완전성(511프레임 기대) |
|---|---|---|---|---|
| A. 전체 재인코딩 (EXP-017 방식) | 0.467s | 2.838s | 5,713,451 bytes | 511/511 (완전) |
| B. Naive Stream Copy (Active Stream 그대로 오려냄) | **0.0003s** | 0.037s | 4,722,960 bytes | **399/511 (112프레임=21.9% 누락, 불완전)** |
| C. Hybrid (IDLE 구간만 재인코딩 + 나머지 Stream Copy, concat) | **0.085s (-81.7%)** | 0.568s (-80.0%) | 5,763,825 bytes (+0.9%) | **511/511 (완전)** |

GOP=1(전부 Keyframe)에서도 같은 경향: A=0.433s/37,012,602B, C=0.062s(**-85.6%**)/37,163,497B(+0.4%),
둘 다 511/511 완전. B는 GOP과 무관하게 399/511로 동일하게 불완전하다(Active Stream 자체에
그 프레임이 없으므로 GOP을 조정해도 해결되지 않음).

### 4) Active Stream 연속 녹화 저장량 — GOP Trade-off (같은 399프레임 내용)

| GOP | 파일 크기 | 비고 |
|---|---|---|
| 1 (전부 Keyframe) | 29,929,152 bytes | Stream Copy 절단 시 Byte 단위로 정확 |
| 25 (1초당 1개) | 4,722,960 bytes (**-84.2%**) | 절단 시 최대 GOP-1 프레임(~1초) 오차 가능 |

### 5) Stream Copy 절단 정확도 — 파일 시작이 아닌 중간 지점에서 자르는 경우 (보조 측정)

실제 Event Clip 절단은 우연히 Active Stream의 맨 처음(위치 0)부터 시작해(§2의 IDLE 겹침 때문에
Active 구간 전체가 Clip에 포함됨) GOP에 따른 절단 오차가 드러나지 않았다. 이를 별도로
검증하기 위해, 이미 생성된 `active_stream.mp4`(GOP=1, GOP=25)에서 파일 중간(위치 130에서
200프레임)을 오려내는 보조 실험을 추가로 수행했다:

| GOP | 요청 프레임 수 | 실제 추출 프레임 수 | 오차 |
|---|---|---|---|
| 1 | 200 | 200 | 0 |
| 25 | 200 | 204 | **+4 (앞쪽으로 최대 ~0.16s 더 포함, Keyframe 스냅 때문)** |

GOP=25에서 요청 시작 지점(프레임 130) 이전의 가장 가까운 Keyframe(프레임 125, 125=25×5)으로
스냅되어 잘렸다 — 예상한 Keyframe 경계 오차 메커니즘이 실측으로 확인됐다. 이 오차는
"요청한 것보다 조금 더 이른 시점부터" 포함되는 것이라 보안 Event Clip 맥락에서는 오히려
여유(추가 Pre-Roll)로 작용해 무해하지만, §2의 "IDLE 겹침으로 인한 프레임 누락"과는 종류가
다른 문제임을 구분해서 기록한다.

## Failure Cases

**Naive Stream Copy는 "빠르지만 조용히 틀린" 결과를 만든다**: B 방식은 CPU 시간이 A 대비
1,000배 이상 빠르지만, 사고 발생 직전 4.5초(112프레임)가 통째로 빠진 Clip을 "정상
완료"로 반환한다 — 파일이 정상적으로 열리고 재생도 되기 때문에 자동화된 파이프라인에서
누락을 알아채기 어렵다. 이번 실험에서 `clip_window_all_active()`로 사전에 계획 단계에서
감지했기 때문에 실제 파일을 열어보지 않고도 문제를 잡을 수 있었다 — Metadata/코드
검증이 육안 확인보다 먼저 이런 종류의 조용한 실패를 잡아낼 수 있다는 근거가 됐다.

## Analysis

1. **가설 1(재인코딩 제거로 CPU 절감)은 확인됐다.** Hybrid 방식은 두 GOP 설정 모두에서
   전체 재인코딩 대비 CPU 시간을 74.6~85.6% 줄이면서 파일 크기는 0.4~0.9%만 늘었다(정확히
   같은 픽셀 데이터를 두 번 인코딩하지 않고 재사용하기 때문 — 원인이 코드로 명확히 설명됨).
2. **가설 2(IDLE 겹침으로 Stream Copy가 불완전해짐)도 확인됐다** — 다만 EXP-017이 아니라
   이번의 "더 긴/IDLE 비중이 큰" 시나리오에서만 드러났다. 이는 Roadmap Next Action의
   "더 긴 영상으로 재검증"이 실제로 새로운 정보를 준 사례다: 같은 로직이라도 IDLE:NORMAL
   비율이 달라지면 없던 경계 사례가 나타난다.
3. **가설 3(GOP Trade-off)도 확인됐다.** GOP=1은 절단이 정확하지만 연속 녹화 저장량이
   6.3배(+84.2%p 상당) 커진다 — Adaptive Recording의 원래 목적(저장량 절감)과 정면으로
   충돌한다. GOP=25는 저장량을 크게 아끼면서 절단 오차는 최대 ~1초(무해한 방향, 더 포함됨)
   수준이라 실무적으로 합리적인 절충점으로 보인다.
4. **B(Naive Copy)만으로는 채택할 수 없고, A(전체 재인코딩)는 CPU/시간을 낭비한다.**
   C(Hybrid)가 두 방식의 장점(A의 완전성 + B에 가까운 저비용)을 모두 만족해 최종 채택
   대상이다. 다만 C도 IDLE 구간 원본 프레임을 다시 만들어낼 수 있어야 하는데, 이번
   실험은 결정론적 합성 프레임 생성 함수(`make_frame`)로 그 프레임을 "재생성"했다 —
   실제 카메라 환경에서는 원본 프레임이 결정론적으로 재생성되지 않으므로, 지침 23이
   언급한 **Circular Buffer(최근 N초의 고화질 원본 프레임을 Tier와 무관하게 항상 보관)**
   가 실제로 필요하다는 근거가 이번 실험으로 명확해졌다(추측이 아니라, "재생성 불가능한
   실제 카메라에서는 이 방법 자체가 성립하지 않는다"는 구조적 결론).

## Decision
Event Clip 생성 방식으로 **C(Hybrid: IDLE 겹침 구간만 재인코딩 + 나머지는 Active Stream에서
FFmpeg Stream Copy, concat demuxer로 병합)** 를 채택한다. Active Stream 인코딩은 GOP=25
(1초당 1 Keyframe) 를 기본값으로 권장한다 — GOP=1은 절단 정확도를 얻는 대신 연속 녹화
저장량이 6.3배 커져 Adaptive Recording의 핵심 목적과 상충하고, GOP=25의 절단 오차(~1초,
더 포함되는 방향)는 보안 Event Clip 용도에서 실질적 위험이 없다고 판단했다.

`clip_window_all_active()`를 Clip 생성 전 필수 점검 단계로 두어, IDLE 겹침이 없으면(대부분의
경우, 특히 사람이 자주 있는 카지노 플로어 환경) 순수 Stream Copy(방식 B, 최저 비용)만으로
충분하고, 겹침이 있을 때만 Hybrid 경로로 전환하는 정책을 권장한다.

## Next Action
- 이번 실험은 IDLE 겹침 구간의 원본 프레임을 결정론적 함수로 "재생성"했다. 실제 카메라
  파이프라인에 적용하려면 지침 23의 Circular Buffer(예: 최근 15~20초 고화질 원본 프레임을
  Tier와 무관하게 Ring Buffer로 보관)를 실제로 구현해, Hybrid 방식의 재인코딩 조각을
  "재생성"이 아니라 "버퍼에서 꺼내기"로 바꿔야 한다 — 다음 세션 후보.
  자세한 근거는 위 Notion 01. Architecture & Roadmap과 이번 experiment.md Analysis 4번 참고.
- `run_full_pipeline.py`에는 아직 Adaptive Recording이 통합되지 않았다(EXP-017/019 모두
  독립 실험) — Core MVP 파이프라인 통합은 Stretch Goal 우선순위 재검토 후 진행.
