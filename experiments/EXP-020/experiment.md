# EXP-020 Adaptive Recording Circular Buffer 실제 구현 (지침 23)

관련 실험: EXP-017 (Adaptive Recording Baseline, PAR-008), EXP-019 (Event Clip Stream Copy Hybrid, PAR-010)

## Goal
EXP-019/PAR-010의 Next Action과 Notion 01. Architecture & Roadmap의 최우선 후보를 실제로
구현한다: Hybrid Event Clip 생성 방식이 IDLE Tier와 겹치는 구간의 원본 프레임을
`make_frame()`으로 "결정론적으로 재생성"했는데, 실제 카메라는 지나간 원본 프레임을
재생성할 수 없다. 지침 23의 Circular Buffer(최근 N초 고화질 원본 프레임을 Tier와 무관하게
항상 보관)를 실제로 구현해 "재생성"을 "버퍼에서 꺼내기"로 바꾸고, 그 과정에서 버퍼
용량(capacity) 설정 자체가 새로운 실패 지점이 될 수 있는지 측정한다.

## Hypothesis
1. Circular Buffer가 Pre-Roll 전체를 커버할 만큼 충분히 크면(capacity >= pre_roll_frames+1),
   버퍼에서 꺼낸 프레임으로 만든 Hybrid Clip은 EXP-019의 "재생성" Hybrid Clip과 완전성·
   내용이 동일할 것이다.
2. 버퍼 용량을 "Pre-Roll 프레임 수와 똑같이"(pre_roll_frames, 흔히 할 법한 단순 설정) 잡으면,
   트리거 프레임 자신도 버퍼에 남아있어야 한다는 사실을 놓쳐 정확히 1프레임이 모자라는
   Off-by-One이 발생할 것이다.
3. Edge Camera처럼 메모리 예산이 Pre-Roll 전체를 담기에 부족하면, 버퍼는 이를 침묵하지 않고
   명시적으로 탐지해야 한다(EXP-019의 Naive Stream Copy가 만든 "조용한 불완전성"과 같은
   실수를 이 계층에서 반복하지 않아야 한다).

## Problem
EXP-019 Analysis 4번: "C(Hybrid)도 IDLE 구간 원본 프레임을 다시 만들어낼 수 있어야 하는데,
이번 실험은 결정론적 합성 프레임 생성 함수로 그 프레임을 '재생성'했다 — 실제 카메라
환경에서는 원본 프레임이 결정론적으로 재생성되지 않는다." 이 실험은 그 이월된 한계를
실제 코드(`src/recording/circular_buffer.py`)로 해소하고 검증한다.

## Dataset
EXP-010/014~019와 동일한 egress 제약(Wikimedia 등 외부 도메인 차단, pypi/GitHub Releases는
정상)으로, 이번에도 ultralytics 내장 `bus.jpg`+Pan 합성 시나리오를 사용한다. **다만 이번
실험은 EXP-019가 이미 실제 YOLO11n+ByteTrack+ROI State Machine으로 추론해 캐시해 둔
`results/EXP-019/detection_cache.json`(1,600프레임 person_present/event_active/roi_events)을
그대로 재사용한다** — Detection/Tracking 로직은 이 실험의 대상이 아니라 Recording/Circular
Buffer 계층만 바뀌므로, 동일 입력을 다시 추론하는 것은 낭비이며 캐시 재사용은 EXP-019
스크립트 자체가 쓰는 패턴과 동일하다.

## Environment
- Remote automated session, `.venv310`(Python 3.11.15) 신규 생성, opencv-python-headless
  5.0.0, ultralytics 8.4.163, numpy, imageio-ffmpeg(정적 FFmpeg 7.0.2/libx264), pytest 신규 설치.
- 프레임 810x1080 BGR uint8 (frame_nbytes = 2,624,400 bytes ≈ 2.5MB/frame).

## Configuration
```yaml
n_frames: 1600            # EXP-019와 동일 시나리오, detection_cache.json 재사용
pre_roll_frames: 250      # 10초 @ 25fps (EXP-017/019와 동일)
gop: 25                   # EXP-019 Decision에서 채택된 Active Stream 기본값
capacity_scenarios:
  undersized_edge_budget: 64      # Edge Camera 메모리 예산이 극단적으로 빠듯한 경우
  off_by_one_naive: 250           # capacity == pre_roll_frames (흔한 실수)
  exact_minimum: 251              # required_capacity_for_pre_roll(250) = pre_roll_frames+1
  production_with_margin: 301     # exact_minimum + 2초(fps*2) 안전마진
```

## Baseline
EXP-019의 방식 C(Hybrid, IDLE 구간을 `make_frame()`으로 재생성)를 Baseline으로 삼는다.
이번 실험은 "재생성"을 "버퍼에서 꺼내기"로 바꿨을 때 (a) 결과가 동일한지, (b) 버퍼 용량
설정에 새로운 실패 지점이 생기는지를 측정한다.

## Result

### 1) 시나리오 재확인 (EXP-019와 동일)
- Clip Plan: `start_frame=1089, end_frame=1599` (511프레임)
- IDLE 겹침(Pre-Roll이 파고든 구간): `idle_prefix_frames_needed = 112` (`[1089,1200]`)
- 첫 Event 트리거 프레임(버퍼 추출이 일어나는 시점): `trigger_frame = 1339`
- `required_capacity_for_pre_roll(250) = 251` (정확한 최소 용량), `+2초 마진 = 301`

### 2) Circular Buffer 용량별 결과

| Scenario | Capacity | 메모리 사용량 | Coverage | 결과 |
|---|---|---|---|---|
| undersized_edge_budget | 64 | 160.18 MB | **불완전 (112프레임 누락)** | `get_range()` KeyError로 명시적 중단 |
| off_by_one_naive | 250 (=pre_roll_frames) | 625.71 MB | **불완전 (정확히 1프레임 누락: frame 1089)** | `get_range()` KeyError로 명시적 중단 |
| exact_minimum | 251 (=pre_roll_frames+1) | 628.21 MB | **완전** | Hybrid Clip 511/511 프레임, EXP-019와 파일 크기 동일(5,763,825 bytes) |
| production_with_margin | 301 | 753.35 MB | **완전** | Hybrid Clip 511/511 프레임, 동일 결과 |

가설 1, 2, 3 모두 실측으로 확인됐다:
- capacity=250(off-by-one)은 정확히 `frame 1089` 1개만 누락 — 이론적으로 계산한 Off-by-One이
  코드 실행 결과로도 정확히 재현됐다(추측이 아니라 `missing_frames=(1089,)`로 직접 확인).
- capacity=251(exact_minimum)부터는 완전한 Clip이 만들어지고, 파일 크기(5,763,825 bytes)가
  EXP-019의 "재생성" Hybrid와 정확히 일치해 버퍼 기반 방식이 내용상 동일함을 확인했다.
- capacity=64(undersized)는 112프레임 전부 누락되며, `coverage()`/`get_range()`가 이를
  조용히 삼키지 않고 즉시 `KeyError`로 알린다(EXP-019 Naive Stream Copy의 "조용한 불완전성"
  실수를 이 계층에서 반복하지 않음).

### 3) 새로 발견한 문제: 원본 프레임 그대로 보관하는 비용

버퍼가 완전성을 보장하는 최소 용량(251프레임)만 유지해도 **메모리 628MB**, 실무에서 쓸
법한 여유(301프레임)는 **753MB**가 필요하다 — 이는 810×1080 해상도, 카메라 1채널
기준이다. 카지노 CCTV는 채널이 수백 개 규모(지침 8/CLAUDE.md 참고)이므로, 이 방식을
그대로 채널마다 적용하면 Pre-Roll 10초만으로 채널당 0.6~0.75GB, 100채널이면 수십~수백GB
RAM이 필요해 Edge Camera/NVR에 전혀 현실적이지 않다. 이는 이번 실험이 실제로 원본 BGR
배열을 그대로 버퍼에 쌓는 구현이기 때문이며(추측이 아니라 `memory_bytes()` 실측), 실제
제품이라면 원본 디코드 프레임이 아니라 **이미 인코딩된 압축 청크(H.264 GOP 조각)** 를
버퍼에 저장해야 한다는 근거가 된다.

## Failure Cases
새로 발견한 실패 사례를 `05. Failure Cases`에 **FC-009**로 등록한다(아래 Analysis/Decision
참고). "완전성만 확보하면 끝"이 아니라 "무엇을 버퍼에 담을지"도 별도의 설계 문제라는 것을
이번 실험으로 확인했다.

## Analysis
1. Circular Buffer 도입으로 EXP-019가 편법으로 남겨뒀던 "재생성"을 실제 구현으로 대체하는
   데는 성공했다 — 용량만 올바르게 잡으면 결과가 EXP-019와 동일하다.
2. 그러나 "올바른 용량"이 자명하지 않다: `pre_roll_frames`와 똑같이 잡는 것은 직관적으로
   그럴듯해 보이지만 정확히 1프레임 부족하다. 이는 코드 리뷰만으로는 놓치기 쉬운 Off-by-One
   클래스의 버그이며, 실제 값으로 재현해 보지 않았다면 "거의 맞는" 것처럼 보였을 것이다.
3. 버퍼가 원본 프레임을 무압축으로 들고 있는 방식은 정확성은 보장하지만 메모리 비용이
   카지노처럼 다채널인 환경에서는 비현실적이다 — "정확하게 동작하는가"와 "실제로 배포할 수
   있는가"는 서로 다른 질문이라는 것을 이번 실험이 정직하게 드러냈다.

## Decision
**Circular Buffer 자체는 채택**하되(EXP-019 Next Action 해소), 용량은 대안 A(무제한/과대
용량)가 아니라 대안 B(`required_capacity_for_pre_roll()`로 계산한 최소값 + 안전마진)를
채택한다 — 상세 근거는 PAR-011 5/6번 참고. 원본 프레임 무압축 저장의 메모리 비용 문제는
이번 실험 범위 밖의 별도 문제(FC-009)로 남기고, Next Action으로 이월한다.

## Next Action
- FC-009(원본 프레임 무압축 Circular Buffer의 메모리 비용)를 해결하려면, 버퍼에 raw
  ndarray 대신 각 프레임을 즉시 JPEG/H.264로 압축해 저장하거나, Active Stream 인코딩
  파이프라인 자체를 프레임 단위로 청크 인코딩해 그 압축 청크를 버퍼링하는 방식을 다음
  세션에서 설계·검증해야 한다.
- `run_full_pipeline.py`에는 아직 Adaptive Recording/Circular Buffer가 통합되지 않았다
  (EXP-017/019/020 모두 독립 실험) — Core MVP 파이프라인 통합은 Stretch Goal 우선순위
  재검토 후 진행.
- Circular Buffer는 이번에도 IDLE 겹침이 "첫 Event 하나"에서만 발생하는 시나리오로
  검증했다 — 짧은 간격으로 여러 Event가 연속 발생해 두 번째 Event의 Pre-Roll이 첫 번째
  Event 처리 중에 아직 버퍼에 남아있는 오래된 프레임을 요구하는 경우(동시 다발 Event)는
  아직 검증하지 않았다.
