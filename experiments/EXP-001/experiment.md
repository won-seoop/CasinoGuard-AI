# EXP-001 Video Input Pipeline Baseline (계획, 미실행)

## Goal
공개 mp4 영상을 읽어 Frame 단위로 처리하는 Video Input Pipeline의 처리 성능 Baseline을 측정한다.

## Hypothesis
OpenCV VideoCapture 기반의 단순 read-loop만으로는 1080p 이상 원본 해상도에서 원본 FPS를 못 따라갈 가능성이 있다 (Resize 없이 처리할 경우).

## Problem
아직 없음 — Baseline 구축 전 단계.

## Dataset
자체 확보 공개 mp4:
- 720p 영상 1개 이상
- 1080p 영상 1개 이상
- 서로 다른 FPS(24/30/60) 영상
- 긴 영상 / 짧은 영상
- 손상된 영상 (읽기 실패 케이스)
- 사람이 없는 영상

## Environment
MacBook Air M3, Python 3.x, OpenCV

## Configuration
- config A: Resize 없음 (원본 해상도)
- config B: Resize 640

## Baseline
Frame read + 단순 timestamp 측정만 수행 (AI 추론 없음)

## Result

실행 환경: MacBook Air M3, Python 3.11, OpenCV 5.0.0 (cv2), AI 추론 없이 Frame Read만 측정.

| Case | 해상도 | 원본 FPS | Frame 수 | 처리 FPS | P50 Latency | P95 Latency | 비고 |
|---|---|---|---|---|---|---|---|
| normal_480p_11s | 640x480 | 30.0 | 345 | 3011.84 | 0.277ms | 0.573ms | 정상 |
| crowded_1080p_56s | 1920x1080 | 23.98 | 1356 | 474.55 | 1.702ms | 3.477ms | 정상, 1080p라 프레임당 지연 ↑ |
| short_3s_480p | 640x480 | 30.0 | 90 | 1107.91 | 0.352ms | 4.123ms | 정상 |
| no_person_5s (synthetic) | 640x480 | 30.0 | 150 | 2389.76 | 0.176ms | 0.844ms | 정상, 사람 없어도 파이프라인 동일 동작 |
| corrupted_truncated | - | - | 0 | 0 | - | - | **OPEN 자체가 실패** (아래 분석) |
| normal_480p_resized_640 | 640x480(리사이즈) | 30.0 | 345 | 2075.23 | 0.337ms | 0.711ms | Resize 오버헤드로 처리FPS 소폭 감소 |

전체 케이스에서 처리 FPS(474~3000)가 원본 FPS(24~30)를 훨씬 웃돌아, **AI 추론이 없는 순수 I/O 구간에서는 실시간 처리가 전혀 문제되지 않음**을 확인했다. 즉 이후 Phase의 실시간성 병목은 Video I/O가 아니라 Person Detection 이후 단계에서 발생할 것으로 예상된다 (가설 반증).

## Failure Cases

**corrupted_truncated.mp4** (정상 mp4의 앞부분 200KB만 남기고 자른 파일):
- 현상: `cv2.VideoCapture.isOpened()`가 `False`. ffmpeg 로그: `moov atom not found`
- 원인: mp4(ISO BMFF) 컨테이너는 파일 재생에 필요한 인덱스 정보(moov atom)가 파일 **끝부분**에 위치하는 경우가 많다. 앞부분만 남기고 자르면 파일을 여는 단계에서부터 실패한다.
- 확인 방법: 동일 파일을 `ffprobe`로도 열어봐서 OpenCV만의 문제가 아님을 확인
- 결과: 예외 없이 `opened_ok=False` + `error` 메시지로 안전하게 처리됨 (프로그램 전체가 죽지 않음) → 05. Failure Cases에 등록

## Analysis
- Video Input 단계는 병목이 아니다 → Phase 2 이후 Person Detection/Tracking 단계에서 실시간성 Trade-off를 본격적으로 다뤄야 한다.
- 실패 처리는 "프레임 중간 손상"보다 "파일 자체가 안 열리는" 형태로 먼저 나타날 수 있다는 것을 실제로 확인했다 → Phase 2부터는 `reader.open()`이 `None`을 반환하는 경우를 항상 먼저 체크하는 패턴을 유지한다.
- Resize를 적용해도(1920x1080→640x480 케이스는 아니지만 640x480→640x480 no-op이라 오버헤드가 정확히 측정되진 않음) 성능 영향은 크지 않아 보임 — 추후 1080p 영상으로 Resize 유무 비교가 필요하다면 별도 실험으로 분리.

## Decision
Phase 1 Video Input Pipeline은 이 Baseline 구조(VideoReader + PipelineStats)를 그대로 Phase 2의 입력단으로 재사용한다. 별도의 최적화(멀티스레딩 등)는 지침 26번 원칙에 따라 실제 병목이 확인되기 전까지 도입하지 않는다.

## Next Action
Phase 2 Person Detection Baseline(EXP-002) 진행 — 이번에는 실시간성 병목이 실제로 나타나는지 측정한다.
