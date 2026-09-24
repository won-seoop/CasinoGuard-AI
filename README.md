# CasinoGuard AI

<img width="2560" height="1656" alt="image" src="https://github.com/user-attachments/assets/f60a6888-0d0c-4c4d-a802-4c91e79fbcce" />


카지노 환경을 가정한 Edge AI 기반 지능형 CCTV 영상관제 시스템 — Computer Vision / AI 영상분석 / Edge AI 직무(특히 한화비전) 취업 준비를 위한 포트폴리오 프로젝트.

전체 프로젝트 지침은 [CLAUDE.md](CLAUDE.md), 실험/문제해결 기록은 Notion([CasinoGuard AI](https://app.notion.com/p/3e05323e448380efa562c79e81f3f6a4))에 있다. 이 README는 완성된 Core MVP를 요약한다.

## 1. 프로젝트 소개

이 프로젝트의 목적은 "YOLO를 실행해본 사람"이 되는 것이 아니라, CCTV 영상을 **Raw Video → Frame → Object → Track → Attribute → Behavior → Event → Searchable Metadata**로 바꾸는 실제 영상보안 시스템을 설계·구현·검증하는 것이다. 모든 주요 결정은 Baseline 구축 → 문제 관찰 → 원인 분석 → 대안 비교 → 개선 → 재측정의 과정을 거쳤다.

## 2. 카지노 CCTV 환경 문제 정의

카지노는 수백~수천 채널의 CCTV를 운영하며, 사람이 많고 조명이 복잡한 환경(저조도/강한 인공조명/역광)에서 부정행위 탐지·출입통제·포렌식 검색·고객경험 개선을 요구한다(한화비전 We-Ko-Pa Casino Resort 케이스스터디 참고). 이런 환경에서는 단순 Detection을 넘어 Tracking, BestShot, Event Analytics, Metadata 검색까지 이어지는 파이프라인이 필요하다.

## 3. 한화비전 영상보안 기술과의 연결

한화비전이 공식적으로 공개한 기능(BestShot, WiseAI Object Detection, Attribute Metadata, Intrusion/Line Crossing/Loitering Video Analytics, People Counting/Heatmap, Wisenet 9 Edge NPU)의 **문제 정의**를 참고해, 공개 데이터와 오픈소스 기술로 동일한 문제를 재현했다. 한화비전이 공개하지 않은 내부 모델(YOLO/ByteTrack 사용 여부 등)은 절대 추정하지 않았다. 자세한 근거는 Notion [02. Hanwha Vision Research] 참고.

## 4. System Architecture

```
Public CCTV Video (Wikimedia Commons)
        ↓
Video Decode / Input (OpenCV)
        ↓
Person Detection (YOLO11n)
        ↓
Multi Object Tracking (ByteTrack)
        ↓
Object State Manager (ROI State Machine)
        ↓
Visual Perception Layer
        ├─ BestShot (Confidence+Sharpness+Size+Occlusion+Position)
        ├─ Intrusion Detection (Polygon ROI)
        ├─ Line Crossing (방향성 Crossing)
        └─ Loitering (Dwell Time)
        ↓
Metadata Pipeline (SQLite)
        ↓
VMS Search API (FastAPI)
        ↓
Performance Benchmark
```

## 5. Dataset

- **영상**: Wikimedia Commons 공개 보행자 영상 2건 (480p/11.6s, 1080p/56.6s — Toronto Yonge-Dundas Square 횡단보도)
- **Detection 평가**: coco128 (person class, GT 61/128장)
- **직접 생성**: 손상된 mp4, 무인 합성영상, 짧은 클립 (Robustness Test용)
- CrowdHuman/ExDark/MOT17 GT는 용량 제약(수GB)으로 이번 세션에서는 미다운로드 — 숫자를 지어내지 않고 정직하게 Stretch로 남김 (Notion [03. Dataset & Evaluation] 참고)

## 6. Person Detection (Phase 2, EXP-002)

YOLO11n Pretrained으로 Baseline 구축. coco128에서 Confidence Threshold(0.2~0.7) Sweep 결과:

| Confidence | Precision | Recall | F1 |
|---|---|---|---|
| 0.2 | 0.776 | 0.697 | **0.734** (최고) |
| 0.4 | 0.940 | 0.555 | 0.698 |
| 0.7 | 0.990 | 0.382 | 0.551 |

imgsz 640/960/1280 비교: FPS 23→11→6으로 감소 — Accuracy(Recall)와 Latency의 Trade-off를 실측으로 확인.

## 7. Multi Object Tracking (Phase 3, EXP-003)

ByteTrack Baseline. crowded_1080p에서 고유 Track ID 41개, 최대 동시 Track 8개, FPS 24.0. MOT17 공식 GT(5.8GB)는 미다운로드해 ID Switch/IDF1은 측정하지 못했음을 명시. 대신 스냅샷 육안 검토로 **배경 밀집 군중은 Detection 자체가 되지 않는 문제(FC-002)**를 발견했다.

## 8. BestShot (Phase 4, EXP-004, **PAR-001**)

Confidence-only로 대표 프레임을 고르면 클로즈업/저품질 프레임이 선택되는 문제를 발견하고, Sharpness+Size+Occlusion+Position을 더한 Composite Score로 개선했다. 32개 Track 비교에서 품질 편차가 큰 Track일수록 개선 효과가 뚜렷했다(정성적 검증, 이미지 그리드로 확인).

## 9. Event Analytics (Phase 5~7)

| Event | 방법 | 핵심 결과 |
|---|---|---|
| Intrusion (EXP-005) | Polygon ROI + Bottom Center + State Machine | Naive 방식 대비 이벤트 96.3% 감소(630→23건) |
| Line Crossing (EXP-006) | Virtual Line + 외적 부호 판정 | 방향별 Crossing 카운트, Flicker 방지 확인 |
| Loitering (EXP-007) | Dwell Time > Threshold | 5초는 과민, 20~30초는 장소 특성상 0건 — Threshold는 장소별로 다시 튜닝 필요 |

Intrusion에서 발견한 **Track 소실 시 EXIT 이벤트 유실 문제(FC-004)**는 `PAR-002`로 정리하고 State Machine을 수정했다.

Line Crossing은 선 근처에서 검출 박스가 몇 픽셀만 흔들려도 Crossing 이벤트가 반복 발생하는 문제(**FC-007**)가 있었다. 900프레임 재현 실험(EXP-014)에서 기존 방식은 57건 중 24쌍이 0.6초 이내 방향이 반전되는 왕복 중복이었고, 선까지의 거리가 `band_px` 이상일 때만 확정 side를 바꾸는 Hysteresis를 적용해 같은 조건에서 중복을 0건으로 제거했다(총 이벤트도 57→31건으로 정상화). Cooldown 기반 대안도 비교했으나 FPS 의존성과 기하학적 근거 부재로 채택하지 않았다 → `PAR-005`.

### Adaptive Recording (Stretch, EXP-017, **PAR-008**)

지침 23의 Baseline(항상 고화질 녹화)과 개선안(No Person→Low FPS / Person→Normal FPS /
Security Event→High FPS+Event Clip)을 구현했다. bus.jpg+Pan 900프레임(IDLE 301 /
NORMAL 338 / EVENT 261) 시나리오에서 연속 녹화 저장량이 18.7% 절감됨을 확인했다.
Event Clip(-10s~+10s)을 Event마다 독립적으로 만들면(대안 A) 간격이 짧은 두 Intrusion
Event의 Window가 완전히 겹쳐 361프레임(41.4%)이 중복 저장되는 것을 발견하고, 겹치거나
인접한 Window를 병합하는 방식(대안 B)으로 개선해 중복을 0으로 만들었다(`PAR-008`).
반대로 Event Clip까지 포함한 총 저장량은 이 36초짜리 데모에서는 Pre/Post-Roll(20초)이
전체 길이의 절반을 넘어 오히려 Baseline보다 47.7% 많아지는 역전 현상도 정직하게 기록했다
(장시간 실제 운영 환경에서 재검증 필요, EXP-017 참고). 최초 ROI 설계(오른쪽 절반)가
bus.jpg에 원래 있던 사람과 겹쳐 NORMAL 구간이 0프레임만 나온 실험 설계 버그도
Failure Case로 기록하고 중앙 대역 ROI로 재설계해 해결했다.

### Heatmap & Crowd Analysis (Stretch, EXP-016, **PAR-007**)

지침 21/22의 Baseline을 구현했다. Heatmap은 Detection Center 누적(Baseline)과 Track이 `min_track_len` 이상 관측된 뒤에만 누적하는 Track-Gated 누적(대안)을 함께 제공한다(이번 bus.jpg+Pan 12px 실측에서는 노이즈 제거 효과가 0.11%로 미미했음을 정직하게 기록). Crowd Analysis는 "그럴듯한" 고정 Threshold(한 셀에 4~5명=Crowded)를 실측 분포에 적용하자 900프레임 내내 Crowded 등급이 한 번도 발동하지 않는 문제를 발견했고, 실측 Percentile(p50/p90) 기반 Threshold로 바꿔 전체의 7.9%(859/10,800)를 정확히 Crowded로 분리했다(`PAR-007`).

## 10. Metadata Pipeline (EXP-008)

Track(trajectory, dwell_time, bestshot_path 등) + Event(Intrusion/LineCrossing/Loitering)를 SQLite 스키마로 통합 저장. 164 Track, 115 BestShot, 190 Event.

저장되는 `dwell_frames`가 항상 0이었던 문제(**FC-006**)를 EXP-015에서 재현·수정했다. 원인은 단순 호출 순서가 아니라 Loitering의 "연속 체류" 상태를 VMS의 "누적 총 체류 시간" 집계에 그대로 재사용한 설계였음을 확인하고, 책임을 분리한 `DwellCounter`로 고쳤다(`PAR-006`).

## 11. VMS Search (EXP-008)

FastAPI로 최소 기능만 구현(Microservice/K8s/복잡한 인증 없음):
- `GET /tracks?min_dwell_sec=` — 특정 시간 이상 체류한 Track 검색
- `GET /tracks/{id}` — Track의 이동 기록 + Event 이력
- `GET /events?event_type=` — 특정 Event 발생 시점 검색
- `GET /` — 최소 Dashboard (Dwell 상위 5개, 최근 Event 10개)

## 12. Performance Benchmark (EXP-008)

| 단계 | FPS | P95 Latency |
|---|---|---|
| Video I/O만 | 474~3011 | 0.6~3.5ms |
| Detection만 | 23.4~27.4 | 51~57ms |
| **전체 통합 파이프라인** | **27.7** | **41.8ms** |

MacBook Air M3에서 원본 영상 FPS(24~30)보다 빠르게 전체 파이프라인이 동작 — 실시간 처리 가능함을 확인했다.

## 13. Failure Cases (누적)

| ID | 내용 | 상태 |
|---|---|---|
| FC-001 | 손상된 mp4는 OpenCV가 아예 못 엶(moov atom) | 예외 없이 안전 처리 완료 |
| FC-002 | 배경 밀집 군중 Detection 실패 | Wikimedia 영상 접근 복구되는 다음 세션으로 이월 |
| FC-003 | BestShot Position Score 경계 페널티 부정확 | 개선 백로그 |
| FC-004 | Track 소실 시 EXIT 유실 | **PAR-002로 수정 완료** |
| FC-005 | BestShot 후보 무제한 누적 Memory Leak | **PAR-004로 수정 완료** |
| FC-006 | Track 체류시간(dwell_frames)이 전부 0으로 저장됨 | **PAR-006로 수정 완료** |
| FC-007 | 선 근처 박스 흔들림으로 Line Crossing 왕복 중복 이벤트 | **PAR-005로 수정 완료** |

## 14. 주요 기술 의사결정

- **ByteTrack** (Baseline, BoT-SORT는 아직 미비교 — Stretch)
- **Bottom Center** (ROI/Line 판정 기준점 — 사람이 서 있는 바닥 위치에 가장 가까움)
- **SQLite** vs PostgreSQL — 단일 프로세스 데모 규모에 적합해 SQLite 채택
- **OpenCV VideoCapture** (Baseline, FFmpeg 직접 제어는 Stretch)

## 15. 개선 전후 결과 (PAR 요약)

- **PAR-001**: Confidence-only BestShot → Composite Score로 개선 (정성적 검증)
- **PAR-002**: Intrusion EXIT 이벤트 유실 → forget_track() 수정 (Unit Test로 검증)
- **PAR-003**: "ONNX/INT8이 항상 빠르다"는 통념이 M3에서는 성립하지 않음을 실측으로 확인 (PyTorch FP32 25.1 FPS > ONNX INT8 22.0 FPS > ONNX FP32 19.2 FPS)
- **PAR-004**: Long Running Test로 BestShot 후보 무제한 누적 Memory Leak 발견 → Incremental Best-Tracking(O(1))으로 개선 (349초 만에 +3157MB/안전중단 → 32분간 +6MB 수준으로 평탄화)
- **PAR-005**: Line Crossing 왕복 중복 이벤트(FC-007) → 선까지 거리 기반 Hysteresis(band_px)로 개선, Cooldown 대안 대비 채택 이유 포함 (900프레임 재현에서 중복 24쌍 → 0쌍, 총 이벤트 57 → 31건)
- **PAR-006**: Track 체류시간(dwell_frames)이 항상 0으로 저장되는 문제(FC-006) → 호출 순서 교정만으로는 실제 케이스의 2/3가 여전히 실패함을 실측으로 확인하고, Loitering의 연속-스트릭 상태와 분리된 DwellCounter로 근본 수정 (통제된 시나리오 3종 모두 Ground Truth와 일치, 실제 YOLO+ByteTrack 실행에서 dwell>0 Track 비율 0%→83.3%)
- **PAR-007**: Crowd Analysis 고정 Threshold가 실측 분포와 어긋나 Crowded 등급이 전혀 발동하지 않는 문제(Degenerate Classification) → Percentile 기반 Threshold 도출로 개선 (Crowded 분류 비율 0.0%→7.9%, 859/10,800건)
- **PAR-008**: Adaptive Recording Event Clip을 Event마다 독립적으로 저장하면(대안 A) 간격이 짧은 두 Event의 Window가 겹쳐 중복 저장되는 문제 → 겹치거나 인접한 Window를 병합하는 방식(대안 B)으로 개선 (중복 361프레임(41.4%) → 0, 연속 녹화 저장량 18.7% 절감)

## 16. Long Running Test (EXP-010, PAR-004)

Full Pipeline(Detection+Tracking+ROI/Line/Loitering+BestShot+Metadata)을 실제 사람이 찍힌 정지 장면
(+합성 Pan/Jitter)으로 연속 실행해 Memory/FPS 안정성을 측정했다. 기존 BestShot 로직은 Track마다 관측된
모든 crop을 무제한으로 쌓아 349초 만에 Memory가 +3.1GB 증가해 안전 중단되었고(≈9MB/초, FPS도 −12.4%
하락), 매 프레임 즉시 채점해 최고 점수 1개만 유지하는 Incremental Best-Tracking으로 바꾼 뒤에는 32.1분
(15,000프레임) 연속 실행에서도 Memory 증가가 워밍업 이후 사실상 0(+6MB)에 수렴함을 확인했다.
(이번 세션은 네트워크 정책상 기존 Wikimedia 영상을 재확보할 수 없어 대체 입력을 사용함 — EXP-010
experiment.md Decision 참고. 실제 다양한 군중 영상으로의 교차 검증은 다음 세션 Next Action으로 남김.)

## 17. Edge 환경 고려

MacBook Air M3(CPU/MPS)에서 CUDA/TensorRT 없이 YOLO11n Pretrained로 실시간 처리를 달성했다(EXP-008 참고). Edge Optimization(EXP-009)에서 PyTorch FP32 → ONNX FP32 → ONNX INT8(Dynamic Quantization)을 실측 비교한 결과:

| Model | Size(MB) | F1 | FPS |
|---|---|---|---|
| PyTorch FP32 | 5.61 | 0.698 | **25.14** |
| ONNX FP32 | 10.74 | 0.700 | 19.20 |
| ONNX INT8 (Dynamic) | **3.05** | 0.686 | 21.95 |

"ONNX/INT8이 항상 더 빠르다"는 통념과 반대로, M3에서는 PyTorch 원본이 가장 빨랐다(PAR-003). ONNX Runtime의 CPUExecutionProvider가 Apple Silicon 네이티브 커널보다 느리고, Dynamic Quantization은 가중치만 압축할 뿐 Convolution 연산 자체를 가속하지 못하는 것이 원인으로 분석된다. 모델 크기가 중요한 실제 Edge Camera 배포 시에는 Static Quantization/CoreML 변환을 추가로 검토해야 한다(미검증, 백로그). C++ 포팅(Video Frame Reader, ROI 계산, Line Crossing, Event State Machine 등)은 아직 Stretch Goal로 남아 있다.

## 18. 프로젝트 한계

- 카지노와 가장 유사한 "실내 상업시설 CCTV" 공개 Dataset이 거의 없어 리테일/캠퍼스/거리 Dataset(Wikimedia, coco128)으로 대체했다.
- MOT17/CrowdHuman/ExDark 등 대용량 GT Dataset은 용량 제약으로 미다운로드 — 정량 지표(ID Switch, Occlusion Recall 등)를 일부 측정하지 못했다.
- 다중 카메라 Person Re-ID는 한화비전 공식 자료에 없어 실제 구현과 연결짓지 않고 Stretch로만 다룬다.
- Ground Truth 없이 정성적으로만 검증한 부분(BestShot 등)이 있다.

## 19. RTSP / ONVIF 확장 계획

현재는 공개 mp4/webm 파일을 입력으로 사용한다. 추후 실제 IP Camera를 확보하면 RTSP 스트림 입력과 ONVIF 기반 카메라 제어로 확장할 계획이다.

---

## 실행 방법

```bash
cd cctv
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python ultralytics numpy fastapi "uvicorn[standard]" pytest lap matplotlib

# 단위 테스트 (50개)
pytest tests/ -q

# 각 Phase 실험 재현
python scripts/run_exp001_video_input_baseline.py
python scripts/run_exp002_detection_baseline.py
python scripts/run_exp003_tracking_baseline.py
python scripts/run_exp004_bestshot.py
python scripts/run_exp005_intrusion.py
python scripts/run_exp006_line_crossing.py
python scripts/run_exp007_loitering.py
python scripts/run_exp010_long_running_test.py --mode baseline  # Memory Leak 재현 (PAR-004)
python scripts/run_exp010_long_running_test.py --mode fixed     # 수정 후 재측정
python scripts/run_exp014_line_crossing_hysteresis.py           # FC-007 재현 + band_px A/B (PAR-005)
python scripts/run_exp015_dwell_time_fix.py                     # FC-006 재현 + DwellCounter 수정 검증 (PAR-006)
python scripts/run_exp016_heatmap_crowd.py                      # Heatmap Detection Center vs Track-Gated, Crowd Threshold 비교 (PAR-007)
python scripts/run_exp017_adaptive_recording.py                 # Adaptive Recording Baseline vs 개선, Event Clip 병합 A/B (PAR-008)

# 통합 파이프라인 + VMS Search API
python scripts/run_full_pipeline.py
uvicorn api.main:app --app-dir src --reload
# http://127.0.0.1:8000 (Dashboard), http://127.0.0.1:8000/docs (Swagger)
```

## 프로젝트 구조

```
cctv/
  CLAUDE.md              # 프로젝트 지침 (전체 원칙)
  src/
    video_pipeline/      # Phase 1
    detection/           # Phase 2 평가 유틸
    bestshot/             # Phase 4 (+ tracker.py: Incremental Best, PAR-004)
    events/               # Phase 5~7 (ROI/Line/Loitering) + dwell.py (누적 체류, PAR-006)
    analytics/            # Heatmap + Crowd Analysis (EXP-016, PAR-007)
    recording/            # Adaptive Recording Tier/Event Clip 계획 (EXP-017, PAR-008)
    metadata/             # Metadata Store (SQLite)
    api/                  # VMS Search API (FastAPI)
  scripts/                # EXP-001~017 실행 스크립트 + 통합 파이프라인
  tests/                  # Unit Test 90개 이상
  experiments/            # EXP-001~010/014~017, PAR-001~008 기록
  results/                # 실행 결과(CSV, 스냅샷, BestShot 그리드)
  data/                   # raw/test 영상 (대용량은 git 제외)
```
