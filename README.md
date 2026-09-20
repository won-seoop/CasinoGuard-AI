# CasinoGuard AI

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

## 10. Metadata Pipeline (EXP-008)

Track(trajectory, dwell_time, bestshot_path 등) + Event(Intrusion/LineCrossing/Loitering)를 SQLite 스키마로 통합 저장. 164 Track, 115 BestShot, 190 Event.

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
pip install opencv-python ultralytics numpy fastapi "uvicorn[standard]" pytest lap

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
    events/               # Phase 5~7 (ROI/Line/Loitering)
    metadata/             # Metadata Store (SQLite)
    api/                  # VMS Search API (FastAPI)
  scripts/                # EXP-001~010 실행 스크립트 + 통합 파이프라인
  tests/                  # Unit Test 53개
  experiments/            # EXP-001~010, PAR-001~004 기록
  results/                # 실행 결과(CSV, 스냅샷, BestShot 그리드)
  data/                   # raw/test 영상 (대용량은 git 제외)
```
