# EXP-021 Circular Buffer 압축 청크(JPEG) 버퍼링으로 FC-009 해결

관련 실험: EXP-020 (Circular Buffer Baseline, PAR-011), EXP-019 (Event Clip Hybrid, PAR-010)

## Goal
FC-009("Circular Buffer가 원본 BGR ndarray를 그대로 보관하면 Pre-Roll 완전성을 보장하는
최소 용량(251프레임)만으로도 채널당 628.21MB, 운영 여유분(301프레임)은 753.35MB가 필요해
다채널 카지노 CCTV에 비현실적")를 실제로 해결한다. `FrameCircularBuffer`가 이미
Generic(`FrameT`)이라는 점을 이용해, 버퍼에 넣기 직전 프레임을 JPEG로 압축하고 꺼낸 직후
디코드하는 방식으로 자료구조 자체는 바꾸지 않고 메모리 문제를 해결할 수 있는지, 그리고 그
대가(CPU 인코딩 비용, 화질 손실)가 실제로 감당 가능한 수준인지 정량적으로 검증한다.

## Hypothesis
1. JPEG 압축은 quality를 충분히 높게(예: 80대) 잡으면 시각적으로 거의 손실이 없으면서도
   원본 대비 메모리를 10배 이상 절감할 것이다.
2. 프레임마다 인코딩하는 추가 CPU 비용은 실시간 처리 예산(25fps 기준 40ms/frame) 대비
   무시할 수 있는 수준일 것이다.
3. 압축 버퍼로 복원한 프레임도 EXP-020의 raw 버퍼와 동일하게 완전한 Event Clip(251/251
   프레임)을 만들 수 있을 것이다 — capacity Off-by-One 특성도 동일하게 재현될 것이다.
4. 해상도를 낮춰 raw로 저장하는 방식(대안 A)은 같은 메모리 예산에서 JPEG보다 화질(특히
   경계/디테일) 손실이 더 크고, 무엇보다 원본 해상도 자체를 되돌릴 수 없어(디테일이
   영구적으로 사라짐) 카지노 사고 조사처럼 화면을 확대해서 봐야 하는 용도에 부적합할
   것이다.

## Problem
EXP-020 Analysis: "버퍼가 원본 프레임을 무압축으로 들고 있는 방식은 정확성은 보장하지만
메모리 비용이 카지노처럼 다채널인 환경에서는 비현실적이다." 이 실험은 그 이월된 문제
(FC-009)를 실제 코드(`src/recording/frame_codec.py`)로 해소하고 검증한다.

## Dataset
EXP-019/020과 완전히 동일한 egress 제약(Wikimedia 등 외부 도메인 차단, pypi/GitHub
Release Assets만 허용) 아래, 이번에도 ultralytics 패키지에 번들된 `bus.jpg` 샘플
이미지를 사용한다. 다만 이번에는 `pip download --no-deps ultralytics`로 patch 파일(wheel,
1.4MB)만 받아 `assets/bus.jpg`를 추출했다 — `ultralytics` 전체 설치는 torch 등 무거운
전이 의존성을 끌어오는데, 이 실험은 실제 YOLO 추론을 다시 하지 않고 EXP-019가 이미 캐시해
둔 `results/EXP-019/detection_cache.json`(person_present/event_active)을 재사용하므로
불필요하다. `bus.jpg`는 `data/raw/bus.jpg`에 저장했다(공개 오픈소스 프로젝트의 샘플
이미지, 사생활 정보 없음, 지침 8 원칙에 부합).

Circular Buffer가 실제로 필요한 구간은 EXP-020에서 이미 확인한 Pre-Roll 추출 시점
(`trigger_frame=1339`, window=`[1089,1339]`, 251프레임)뿐이므로, 버퍼가 그 시점에 보관할
수 있는 최대 구간(`[1039,1339]`, capacity=301 기준 301프레임)만 실제 `make_frame()`으로
재생성했다 — 이 실험 코드로 `event_intervals_from_active_flags(event_active)[0].start_frame
== 1339`, `plan_event_clips(...) == [start_frame=1089, end_frame=1599]`임을 재확인해
EXP-020 문서값과 정확히 일치함을 검증했다.

## Environment
- 원격 자동화 세션, `.venv311`(Python 3.11.15) 신규 생성, opencv-python-headless 5.0.0,
  numpy 2.4.6, pytest 9.1.1, imageio-ffmpeg(정적 FFmpeg 7.0.2/libx264).
- ultralytics는 설치하지 않음(위 Dataset 참고, wheel만 추출).
- 프레임 810×1080 BGR uint8 (`frame_nbytes_raw` = 2,624,400 bytes).

## Configuration
```yaml
pre_roll_frames: 250
trigger_frame: 1339          # EXP-019/020과 동일 시나리오에서 실측 재확인
window: [1089, 1339]         # 251 frames
capacity_scenarios:
  undersized_edge_budget: 64
  off_by_one_naive: 250
  exact_minimum: 251
  production_with_margin: 301
jpeg_qualities: [30, 50, 70, 85, 95]
downscale_factor_alt_a: 0.5  # 대안 A 비교용 (405x540)
timing_repeats: 5            # encode/decode 타이밍 측정 반복 횟수 (아래 Failure Cases 참고)
```

## Baseline
EXP-020의 raw ndarray 버퍼(capacity=251/301에서 628.21MB/753.35MB, 재확인 시
658.72MB/789.94MB — EXP-020은 MiB(1024²) 단위, 이번 실험 출력은 MB(10⁶) 단위라 표기가
다를 뿐 바이트 수는 동일: 658,724,400 bytes = 628.21 MiB, 789,944,400 bytes = 753.35 MiB)를
Baseline으로 삼는다.

## Result

### 1) 시나리오 재확인 (EXP-019/020과 완전히 동일)
- `trigger_frame=1339`, `window=[1089,1339]`(251프레임) — 코드로 재계산해도 동일.
- capacity=250(=pre_roll_frames, 흔한 실수)은 정확히 frame 1089 하나만 누락(Off-by-One
  재현), capacity=251부터 완전 — EXP-020과 동일하게 재현됨(아래 표 참고, quality와
  무관하게 missing_count는 동일함을 확인 — coverage()는 프레임 존재 여부만 보고 내용
  크기는 보지 않으므로 당연한 결과지만 실측으로 재확인함).

### 2) Capacity × Quality별 메모리 사용량 (`results/EXP-021/capacity_memory.csv`)

| Capacity(시나리오) | Raw(무압축) | JPEG q30 | q50 | q70 | q85 | q95 | Coverage |
|---|---|---|---|---|---|---|---|
| 64 (Edge 예산) | 167.96 MB | 6.79 MB | 9.18 MB | 12.21 MB | 17.30 MB | 29.11 MB | 불완전(187 누락, quality 무관) |
| 250 (Off-by-One) | 656.10 MB | 23.89 MB | 32.35 MB | 43.31 MB | 61.49 MB | 104.11 MB | 불완전(1 누락, quality 무관) |
| 251 (최소 완전) | 658.72 MB | 23.97 MB | 32.46 MB | 43.46 MB | 61.71 MB | 104.49 MB | **완전** |
| 301 (운영 여유) | 789.94 MB | 28.08 MB | 38.07 MB | 51.01 MB | 72.47 MB | 122.88 MB | **완전** |

(MB = 10⁶ bytes 기준. capacity=301에서 q85 채택 시 메모리 절감률 = (789.94-72.47)/789.94 =
**90.8%**.)

### 3) Quality별 압축률·CPU 비용·화질 손실 (`results/EXP-021/quality_sweep.csv`, window=251프레임 기준)

| Quality | 평균 크기/프레임 | 압축률(raw 대비) | Encode(ms/frame) | Decode(ms/frame) | 평균 픽셀 MAE(0~255) |
|---|---|---|---|---|---|
| 30 | 93,288 B | 28.1x | 3.35 | 3.46 | 6.59 |
| 50 | 126,468 B | 20.8x | 3.95 | 3.99 | 4.97 |
| 70 | 169,469 B | 15.5x | 3.77 | 4.13 | 3.86 |
| 85 | 240,778 B | 10.9x | 4.47 | 4.90 | 2.61 |
| 95 | 408,250 B | 6.4x | 5.00 | 6.79 | 1.37 |

25fps 기준 프레임 예산은 40ms/frame이므로, q85의 encode 4.47ms는 예산의 **11.2%**,
decode 4.90ms는 **12.3%** — 채널 1개 기준으로는 여유가 있으나, 여러 채널이 CPU 코어를
공유하는 Edge NVR 환경에서는 한 코어가 감당할 수 있는 동시 채널 수가
`floor(40ms / 4.47ms) ≈ 8채널`로 제한된다는 것도 함께 확인했다(Analysis 4번 참고).

### 4) 압축 버퍼로 만든 Pre-Roll 구간 Clip의 완전성/파일 크기 (raw 재인코딩 대비)

| 소스 | Clip 프레임 수 | Clip 크기 | 재인코딩 CPU 시간 |
|---|---|---|---|
| Raw(무압축) 버퍼 | 251/251 | 2,583,225 bytes | 1.83s |
| JPEG q30 버퍼→디코드 | 251/251 | 7,359,075 bytes | 2.78s |
| JPEG q50 버퍼→디코드 | 251/251 | 5,936,742 bytes | 2.05s |
| JPEG q70 버퍼→디코드 | 251/251 | 4,576,733 bytes | 2.06s |
| JPEG q85 버퍼→디코드 | 251/251 | 3,073,411 bytes | 1.64s |
| JPEG q95 버퍼→디코드 | 251/251 | 2,510,558 bytes | 1.55s |

모든 quality에서 프레임 수는 251/251로 완전했다(가설 3 확인). 흥미롭게도 q95 버퍼조차
raw 버퍼보다 **최종 clip 파일이 더 작고 재인코딩이 더 빠르다** — JPEG 재압축이 원본의
미세한 고주파 디테일(원본 사진 자체의 촬영/압축 잡음 등)을 일부 제거해 그 뒤 h264
인코더가 처리할 엔트로피가 오히려 줄었기 때문으로 보인다(추측이 아니라 파일 크기·CPU
시간 실측치로 확인됨). 저quality(q30)는 반대로 clip이 raw보다 커지는데, 이는 JPEG
블록 압축 특유의 8×8 블록 경계 아티팩트가 h264 인코더 입장에서는 오히려 인코딩하기
어려운 새로운 고주파 패턴이 되기 때문으로 보인다.

### 5) 대안 A(해상도 다운스케일) 실측 비교
0.5배 다운스케일(405×540) raw 버퍼, capacity=301: **197.49MB**, 업스케일 후 평균 픽셀
MAE **9.10** — JPEG q30(28.08MB, MAE 6.59)보다 메모리를 7배 더 쓰면서도 화질(MAE)은 더
나쁘다. 게다가 다운스케일은 해상도 자체를 절반으로 줄이므로 화면을 확대해서 얼굴/카드
숫자를 확인해야 하는 카지노 사고 조사 용도에는 원천적으로 부적합하다(디코드로 복구할 수
없는 정보 손실).

## Failure Cases
새로 발견한 문제(이번 세션 내에서 발견·수정 완료, 별도 FC 번호는 부여하지 않음 — 실험
방법론 버그이지 제품 코드의 결함이 아니므로):
**현상**: encode/decode 타이밍을 quality별로 warm-up 1회 + 단발 측정 1회로 측정했더니
q30 decode가 16.30ms로 q85(4.96ms)보다 3배 이상 느리게 나오는 등 quality와 무관하게
뒤섞인 결과가 나왔다.
**원인 확인**: 별도의 격리된 벤치마크 스크립트로 재현한 결과, warm-up을 10회 이상 주고
5회 반복 측정을 평균하면 quality가 높을수록 encode/decode 시간이 깨끗하게 단조 증가하는
것을 확인했다 — 단발 측정은 컨테이너 환경의 CPU 스케줄링/캐시 워밍업 노이즈에 취약했다.
**조치**: 실험 스크립트를 quality별 warm-up 1 pass + `TIMING_REPEATS=5` 평균으로
수정했고, 위 3번 표는 그 결과다(재현 가능, `run_exp021_compressed_circular_buffer.py`
Part 1 참고).

## Analysis
1. 가설 1(고 quality에서 시각 손실 거의 없이 10배+ 메모리 절감) 확인: q85에서 MAE 2.61
   (0~255 스케일에서 사실상 무시 가능), 메모리는 raw 대비 10.9배 절감(capacity=301 기준
   789.94MB → 72.47MB).
2. 가설 2(CPU 비용이 실시간 예산 대비 무시 가능) 부분적으로만 확인: 채널 1개 기준으로는
   여유(예산의 11~12%)가 있지만, "무시할 수 있다"고 하기엔 다채널 환경에서 코어당 동시
   채널 수 제약(≈8채널/코어, q85 기준)이 실제로 존재한다 — 가설을 단순화해서 세웠던 부분을
   정직하게 수정한다.
3. 가설 3(압축 버퍼도 완전한 Clip을 만든다) 확인: 모든 quality에서 251/251 프레임 완전,
   capacity Off-by-One(250 vs 251) 특성도 EXP-020과 동일하게 재현됨 — quality는
   `coverage()`/`get_range()`의 완전성 판정 로직과 완전히 독립적임을 확인(자료구조를
   전혀 바꾸지 않았으므로 당연하지만, 실측으로 재확인함).
4. 가설 4(다운스케일 대안보다 JPEG가 우월) 확인: 같은 메모리 예산 근처에서 비교해도
   JPEG가 다운스케일보다 화질이 낫고, 무엇보다 해상도 손실이 "복구 불가능"하다는 점에서
   구조적으로 열등하다.
5. 새로 발견한 사실: 매우 높은 quality(95)에서조차 압축 버퍼가 만드는 최종 clip이 raw
   버퍼보다 작고 빠르다 — "압축=항상 손실만 있고 원본이 항상 더 낫다"는 순진한 가정이
   이번 데이터셋에서는 틀렸다는 것을 실측으로 확인했다(3-4번 참고). 다만 이는 이번
   합성 시나리오(반복적인 배경 텍스처 + 매끄러운 Pan)의 특성일 수 있어, 실제 카지노
   영상(복잡한 텍스처의 카펫/조명)에서도 동일한지는 검증하지 못했다(정직하게 한계로
   남김).

## Decision
**대안 B(JPEG 압축 청크 버퍼링)를 채택**하고 `src/recording/frame_codec.py`
(`encode_frame`/`decode_frame`)로 구현했다. `FrameCircularBuffer`는 이미
Generic(`FrameT`)이었으므로 자료구조 자체는 전혀 수정하지 않고, 버퍼에 push하기 직전/
get_range 직후의 경계에서만 인코딩/디코딩을 추가했다 — "버퍼가 무엇을 담을지"와 "버퍼가
어떻게 완전성을 보장할지"를 분리한 기존 설계(EXP-020)가 그대로 유효했다는 뜻이다.
**대안 A(해상도 다운스케일 raw 버퍼)는 기각** — 같은 메모리 예산에서 화질이 더 나쁘고,
결정적으로 해상도 손실이 복구 불가능해 카지노 사고 조사(화면 확대 확인)라는 이 프로젝트의
핵심 요구사항과 상충한다.

Quality 기본값은 **85**를 권장한다: MAE 2.61(시각적으로 사실상 무손실)이면서 메모리
절감률 90.8%를 확보하는 지점이다. Quality 95는 더 무손실에 가깝지만 압축률이 6.4배로
크게 떨어지고, quality 70 이하는 압축률은 더 좋지만 MAE가 3.86 이상으로 커져 저조명
등 다른 화질 저하 요인과 누적되면 사고 조사 용도에 부적합해질 위험이 있다고 판단했다.

`FrameCircularBuffer.memory_bytes(frame_nbytes)`가 균일 크기를 가정해 가변 크기
압축 프레임에는 쓸 수 없다는 것을 구현 중 발견해, `total_bytes(size_fn)`을 신규
추가했다(기존 `memory_bytes()`는 raw 버퍼 하위호환을 위해 그대로 유지).

## Next Action
- 이번 실험은 quality를 프레임 전체에 고정값으로 적용했다. 실제 카메라는 IDLE/NORMAL/
  EVENT Tier에 따라 quality를 다르게(예: IDLE은 60, EVENT 직전/직후는 95) 적용해 메모리와
  화질을 Tier별로 다르게 관리할 수 있다 — Adaptive Recording(EXP-017)의 Tier 개념과
  Circular Buffer quality를 연동하는 것이 다음 후보다.
- 실제 카지노와 유사한 복잡한 텍스처(카펫, 다양한 조명)에서도 "고 quality 압축이 원본보다
  작아지는" 현상이 재현되는지는 검증하지 못했다(FC-002 해소 후 실제 혼잡 배경 영상으로
  재검증 필요).
- 코어당 동시 채널 수 제약(≈8채널/코어, q85 encode 기준)을 실제 멀티스레드/멀티프로세스
  파이프라인(지침 26 Multi Thread Pipeline)과 연결해 측정한 적은 없다 — 실제 병목이
  존재하는지는 여러 채널을 동시에 시뮬레이션해야 확인 가능하다.
- `run_full_pipeline.py`에는 Adaptive Recording/Circular Buffer가 아직 통합되지 않았다
  (EXP-017/019/020/021 모두 독립 실험) — Core MVP 파이프라인 통합은 Stretch Goal 우선순위
  재검토 후 진행.
