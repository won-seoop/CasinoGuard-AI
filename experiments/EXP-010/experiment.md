# EXP-010 Long Running Stability Test — BestShot 후보 누적 Memory Leak 발견/수정 ✅ 실행 완료

## Goal
CCTV 시스템은 장시간 연속 실행되므로(지침 28), Core MVP 파이프라인(Detection+Tracking+Event+BestShot+Metadata)을
장시간 실행했을 때 Memory/FPS/Latency가 안정적으로 유지되는지 실측한다.

## Hypothesis
Track 상태(ROI State Machine, Line Crossing, Loitering)는 이미 FC-004(PAR-002)에서 `forget_track()`으로
정리하도록 되어 있으므로, 장시간 실행해도 Memory는 안정적일 것이다.

## Problem (Blocker / Decision)
이번 세션은 원격 자동화 실행 환경으로, egress 네트워크 정책상 pypi/npm/crates 등 패키지 레지스트리
외의 일반 도메인(commons.wikimedia.org, raw.githubusercontent.com, archive.org 등)에 대한 접근이
모두 403으로 차단되어 있었다(`curl`로 직접 확인, `/root/.ccr/README.md` 정책 문서 참고). 이로 인해
기존 EXP-001~009에서 쓰던 `crowded_intersection_1080p.webm`(Wikimedia Commons)을 이 세션에서
다시 받을 수 없었다(`data/raw/`는 지침 33에 따라 Git에 커밋하지 않는 대용량 디렉터리라 완전히 비어 있음).

**Decision**: ultralytics 패키지에 기본 포함된 실제 사진 `bus.jpg`(실제 사람 4명이 버스를 기다리는
YOLO 공식 샘플, 별도 다운로드 불필요)를 기반으로, 프레임마다 작은 Pan(±12px)과 밝기 Jitter(±3%)를
주어 "카메라가 고정된 채 동일 장면을 장시간 촬영"하는 상황을 합성했다. 이는 실제 카지노 CCTV에서
흔한 "딜러/캐셔가 한 자리에 오래 머무는" 시나리오와 구조적으로 유사하며, 이번 실험의 목적이
Detection 정확도가 아니라 파이프라인의 장시간 Memory/FPS 안정성이므로 정당한 대체로 판단했다.
(지침 8의 "실제 가족/일반인 사생활 영상 금지" 원칙에도 위배되지 않음 — 공개 오픈소스 CV 라이브러리에
내장된, 이미 공개적으로 배포되는 표준 벤치마크 이미지.)

## Dataset
`ultralytics/assets/bus.jpg`(1080x810, 실제 사람 4~5명) + 합성 Pan/Jitter (재현 가능, 외부 다운로드 불필요)

## Environment
원격 실행 컨테이너: Linux, CPU 4 core, RAM 15GB, PyTorch 2.14(CPU, MPS/CUDA 없음) — 지침에 명시된
MacBook Air M3와 다른 하드웨어이므로 절대 FPS 수치는 이전 실험(EXP-002/003/008/009, M3)과 직접 비교하지
않는다. 이번 실험의 핵심은 **동일 환경 내에서** Before/After Memory 추세를 비교하는 것이다.

## Configuration
- YOLO11n, ByteTrack, conf=0.4, imgsz=640 (기존 실험과 동일)
- 측정 간격: 200 프레임마다 RSS(`/proc/self/status` VmRSS), FPS(rolling), P95 Latency(rolling), 누적 Track 수,
  BestShot 후보 버퍼에 실제로 들고 있는 관측치 개수, 예외 횟수 기록
- 안전장치: 시작 대비 RSS가 3072MB 이상 증가하면 즉시 중단(컨테이너 OOM 방지)

## Baseline
`run_full_pipeline.py`(EXP-008)의 기존 로직 그대로: Track마다 관측된 모든 프레임의 crop 이미지를
`bestshot_candidates: dict[track_id, list[dict]]`에 계속 append하고, Track이 사라질 때(지금까지는 실제로는
사라져도 이 dict에서 제거되지 않음) 한꺼번에 정렬해서 최고 점수를 고르는 방식.

## Result

### Before (Baseline — 기존 코드)
| elapsed_sec | frame_idx | rss_mb | rss_growth_mb | fps_rolling | p95_ms | buffered_observations |
|---|---|---|---|---|---|---|
| 14.2 | 200 | 939.8 | 371.9 | 25.1 | 45.3 | 797 |
| 98.7 | 1400 | 1772.0 | 1204.1 | 23.3 | 48.2 | 5637 |
| 196.8 | 2600 | 2612.7 | 2044.8 | 22.2 | 49.1 | 10470 |
| 349.0 | 4200 | 3725.4 | 3157.5 | 22.0 | 49.2 | 16887 |

**349초(4201 프레임) 만에 안전 임계값(3072MB 증가)에 도달해 자동 중단됨.** 성장률 ≈ 9.0 MB/초
(≈ 0.75 MB/frame), `buffered_observations`(메모리에 실제로 들고 있는 crop 개수)가 프레임 수에 정비례해
계속 증가 — 전형적인 무제한 누적 Memory Leak 패턴. FPS도 25.1→22.0(−12.4%)으로 지속 하락, P95 Latency도
45.3ms→49.2ms로 증가.

### After (Fix — BestShotTracker, O(1) Incremental Best)
| elapsed_sec | frame_idx | rss_mb | rss_growth_mb | fps_rolling | p95_ms | buffered_observations |
|---|---|---|---|---|---|---|
| 14.2 | 200 | 820.5 | 252.8 | 25.1 | 48.6 | 4 |
| 690.3 | 7400 | 824.1 | 256.3 | 24.0 | 45.8 | 4 |
| 1546.0 | 13000 | 829.9 | 262.2 | 23.8 | 45.8 | 4 |
| 1888.2 | 14800 | 825.8 | 258.1 | 23.4 | 46.5 | 4 |

**15,000 프레임 / 1928.8초(≈32.1분) 끝까지 정상 완료(안전 중단 없음).** RSS는 초기 워밍업(모델/버퍼 초기화)
이후 완전히 평탄(820~830MB 사이 ±10MB 변동, 명확한 증가 추세 없음). `buffered_observations`는 실제
동시 활성 Track 수(4~5)로 고정되어, 지금까지 이 장면을 거쳐간 고유 Track ID가 92개까지 늘어난 것과
무관하게 메모리 사용량이 늘지 않았다. FPS도 큰 추세 없이 23~26 범위에서 유지됨(baseline처럼 지속적으로
하락하지 않음).

## Failure Cases
Baseline 실행에서 **FC-005: BestShot 후보 무제한 누적 Memory Leak**을 발견함 (05. Failure Cases에 별도 기록).

## Analysis
1. **원인**: `run_full_pipeline.py`가 Track마다 "관측된 모든 프레임의 crop 이미지"를 리스트에 계속
   append하고, `forget_track()` 호출 시에도 이 리스트(딕셔너리 항목)를 전혀 정리하지 않았다. ROI/Line/
   Loitering 상태는 FC-004(PAR-002) 개선 이후 `forget_track()`으로 정리되지만, BestShot 후보 버퍼는
   그 개선에서 다뤄지지 않아 별도로 남아 있던 것으로 확인된다(코드 직접 확인, `bestshot_candidates`는
   video 전체 처리가 끝난 뒤 한 번에만 순회됨 — EXP-008 원본 코드 기준).
2. **왜 이번에 처음 드러났는가**: EXP-008에서는 800프레임(≈33초)짜리 짧은 벤치마크만 돌렸기 때문에
   누적량이 작아(≈797개 관측치, +373MB 수준) 눈에 띄지 않았다. 이번처럼 장시간(수 분~수십 분) 실행해야만
   선형 증가 추세가 명확히 드러난다 — Long Running Test가 필요한 이유를 그대로 보여주는 사례.
3. **연쇄 영향**: 카지노 CCTV에서 한 자리에 오래 머무는 대상(딜러, 캐셔, 장시간 게임하는 고객)이 많을수록
   Memory 사용량이 더 빨리 증가하며, 다중 채널(수백 대 카메라)을 한 프로세스에서 처리하는 실제 환경이라면
   훨씬 빨리 문제가 발생한다. 장시간 미조치 시 결국 OOM으로 전체 영상분석 프로세스가 죽어 Forensic 검색·
   실시간 Event 탐지가 모두 중단되는 심각한 장애로 이어질 수 있다.

## Decision
**대안 A (Capped Window)**: Track별 관측 리스트를 최근 N개(예: 30개) 또는 상위 K개 점수만 유지하도록
자르는 방식을 검토했다. 메모리를 줄일 수는 있지만 (1) N을 얼마로 잡을지 임의적이고, (2) 여전히
Track 수 × N 만큼 메모리를 쓰며, (3) 매 프레임 정렬/자르기 로직이 추가로 필요해 코드 복잡도가 늘고,
(4) 이미 매 프레임 `bestshot_score()`를 계산할 수 있는데 굳이 원본 crop을 계속 들고 있다가 나중에
다시 점수를 매길 이유가 없다고 판단해 채택하지 않았다.

**대안 B (Incremental Best-Tracking, 채택)**: 매 프레임 즉시 `bestshot_score()`를 계산해 "지금까지의
최고 점수 후보 1개"만 유지하고 나머지는 즉시 버리는 방식(`src/bestshot/tracker.py::BestShotTracker`).
Track 하나당 메모리 사용량이 관측 프레임 수와 무관하게 O(1)로 고정되고, `forget_track()` 시 그 항목
자체를 dict에서 제거해 "총 몇 명이 스쳐 지나갔는지"와도 무관하게 메모리가 유지된다. 실시간 스트림을
무한정 버퍼링할 수 없는 실제 Edge Camera/NVR 환경의 제약과도 더 맞는 방식이라 채택했다.

## Regression Test
- `run_full_pipeline.py`에 동일한 `BestShotTracker`를 통합해 EXP-008 로직과 결과 스키마(트랙/이벤트/
  BestShot 저장)를 그대로 유지하도록 수정 (관측치 5개 미만인 짧은 False Positive성 Track은 저장하지
  않는 기존 필터도 `observation_count`로 동일하게 유지).
- 기존 pytest 48건 + 신규 `tests/test_bestshot_tracker.py` 5건 = 총 53건 통과 확인 (회귀 없음).
- 원본 영상(`crowded_intersection_1080p.webm`) 재확보 시 EXP-008과 동일 조건으로 재실행해 Track/Event/
  BestShot 개수가 이전과 동일하게 나오는지 확인하는 것을 다음 세션 Next Action으로 남긴다(이번 세션은
  영상 접근 불가로 직접 재검증하지 못함 — 정직하게 명시).

## Next Action
- (인프라) 원본 Wikimedia 영상이 로컬/캐시에 없는 세션에서도 재현 가능하도록, 오늘처럼 "내장 실제 이미지
  + 합성 Pan/Jitter" 방식의 `run_exp010_long_running_test.py`를 Long Running Test 표준 스크립트로 유지한다.
- 원본 영상 확보 가능한 환경에서 `run_full_pipeline.py`(수정판)를 동일하게 장시간(10분+) 재실행해
  실제 다양한 군중 장면에서도 Memory가 안정적인지 교차 검증.
- FC-002(배경 밀집 군중 Detection) 재검토는 Wikimedia 영상 접근이 복구되는 다음 세션으로 이월.
