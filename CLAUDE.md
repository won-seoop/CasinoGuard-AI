# CasinoGuard AI 프로젝트 지침

나는 CCTV 영상보안 업계의 Computer Vision Engineer, AI 영상분석 Engineer, Edge AI Engineer, Edge Device SW Engineer 취업을 준비하고 있다.

특히 한화비전처럼 카지노, 리테일, 공항, 대형 시설 등 다수의 CCTV를 운영하는 환경에서 사용하는 영상인지 기술과 AI 영상분석 기술에 관심이 있다.

이 프로젝트의 목적은 단순히 YOLO를 실행해 사람에게 Bounding Box를 그리는 것이 아니다.

한화비전이 실제 영상보안 제품과 카지노 솔루션에서 공개적으로 다루는 문제를 참고해, 공개 데이터와 오픈소스 기술을 이용하여 실제 CCTV AI 시스템을 설계하고 구현하고 검증한다.

그리고 프로젝트를 진행하면서 실제 엔지니어처럼 다음 과정을 경험하는 것이 가장 중요하다.

문제 발견
→ 왜 중요한지 분석
→ 원인 분석
→ Baseline 구축
→ 대안 A 검토
→ 대안 A를 선택하지 않은 이유
→ 대안 B 검토
→ 대안 B를 선택한 이유
→ 구현
→ 테스트
→ 정량 검증
→ 실패 사례 분석
→ 개선
→ Regression Test
→ 재발 방지
→ PAR 경험 정리

최종적으로 프로젝트에서 발생한 실제 문제 해결 경험을 자기소개서와 기술면접에서 사용할 수 있도록 정리한다.

---

## 1. 프로젝트 이름

CasinoGuard AI

카지노 환경을 가정한 Edge AI 기반 지능형 CCTV 영상관제 시스템

---

## 2. 내 배경

- 컴퓨터공학 학사
- C++ 기반 반도체 설비 SW 개발 인턴 경험 6개월
- Java, Spring Boot 기반 백엔드 개발 경험
- Python 사용 가능
- Computer Vision 실전 프로젝트 경험은 아직 부족함
- MacBook Air M3 사용
- 최종 목표는 CCTV 영상보안 기업의 Computer Vision, AI 영상분석, Edge AI, Edge Device SW 직무
- 특히 한화비전 취업에 관심이 있음
- 실제 IP Camera는 당장 구매하지 않음
- 공개 CCTV 영상과 공개 데이터셋을 이용해 프로젝트 진행
- 추후 필요하면 RTSP, ONVIF 기반 실제 IP Camera 입력으로 확장

---

## 3. 프로젝트에서 가장 중요한 원칙

이 프로젝트에서는 기능 개수보다 문제 해결 깊이를 중요하게 생각한다.

Claude는 많은 기능을 빠르게 구현하는 방향보다, 하나의 기능이라도 다음 과정이 실제로 발생하도록 진행한다.

Baseline
→ 실제 테스트
→ 문제 관찰
→ 문제 측정
→ 원인 분석
→ 해결 방법 비교
→ 선택
→ 개선
→ 동일 조건 재측정
→ 결과 기록

처음부터 가장 좋은 설정과 완성된 코드를 제공하여 문제를 없애버리지 않는다.

그렇다고 일부러 나쁜 코드나 비현실적인 장애를 만들지도 않는다.

실제 개발 과정에서 자연스럽게 발생하는 정확도 문제, 성능 문제, 상태 관리 문제, 영상 처리 문제, 장시간 안정성 문제를 기반으로 개선 경험을 만든다.

---

## 4. 한화비전 기준으로 생각하는 원칙

모든 주요 기능을 구현하기 전에 먼저 다음을 확인한다.

1. 한화비전 공식 자료에서 이 기능 또는 유사한 문제를 확인할 수 있는가?
2. 이 기능이 실제 CCTV 영상보안에서 왜 필요한가?
3. 카지노처럼 사람이 많고 조명이 복잡한 환경에서는 왜 더 중요한가?
4. 이 기능은 Edge Camera, NVR, VMS 중 어디에 위치하는가?
5. 학사 신입 개발자가 이 기능을 구현함으로써 어떤 역량을 보여줄 수 있는가?

한화비전이 공개하지 않은 내부 알고리즘이나 모델을 추측하지 않는다.

예를 들어:

올바른 표현:
> 한화비전이 공개한 BestShot 기능의 문제 정의를 참고하여 자체적인 BestShot scoring 방식을 구현했다.

잘못된 표현:
> 한화비전은 YOLO와 ByteTrack을 사용한다.

한화비전의 내부 모델이 공개되지 않은 경우 특정 모델을 사용한다고 절대 주장하지 않는다.

항상 다음을 구분한다.

- Hanwha Vision 공개 기능 → 실제 산업에서 해결하는 문제
- 내 프로젝트 구현 → 해당 문제를 재현하기 위해 선택한 Open Source 기술

---

## 5. Visual Perception 관점

이 프로젝트를 단순한 Object Detection 프로젝트로 만들지 않는다.

CCTV 영상을 다음과 같이 의미 있는 정보로 변환하는 것이 목표다.

Raw Video → Frame → Object → Track → Attribute → Behavior → Event → Searchable Metadata

즉 모델이 사람을 찾는 것에서 끝나지 않고, 누가 / 언제 / 어디에 있었고 / 얼마나 머물렀으며 / 어떤 Event를 발생시켰는지 검색할 수 있는 영상인지 시스템을 만든다.

---

## 6. 목표 Architecture

```
Public CCTV Video
        ↓
Video Decode / Input
        ↓
Person Detection
        ↓
Multi Object Tracking
        ↓
Object State Manager
        ↓
Visual Perception Layer
        ├─ BestShot
        ├─ Attribute Metadata
        ├─ Person Re-ID
        ├─ Intrusion
        ├─ Loitering
        ├─ Line Crossing
        ├─ People Counting
        ├─ Crowd Analysis
        └─ Heatmap
        ↓
Metadata Pipeline
        ↓
Event Engine
        ↓
Event / Video Storage
        ↓
Search API
        ↓
VMS Dashboard
        ↓
Performance Benchmark
        ↓
Edge Optimization
```

---

## 7. Core MVP와 Stretch Goal

### Core MVP (반드시 먼저 완성)

Video Input → Person Detection → Multi Object Tracking → Object State Management → BestShot → Intrusion → Line Crossing → Loitering → Metadata 저장 → Event Search → 간단한 VMS Dashboard → Performance Benchmark

### Stretch Goal (Core MVP 완성 전에는 구현하지 않음)

Person Re-ID, Attribute Classification, People Counting, Heatmap, Crowd Analysis, Adaptive Recording, ONNX Optimization, C++ 일부 포팅, Multi Thread Pipeline, RTSP, ONVIF

기능을 많이 만드는 것보다 Core 기능에서 깊이 있는 문제 해결 경험을 만드는 것이 우선이다.

---

## 8. 데이터 원칙

실제 가족이나 일반인의 사생활 영상은 사용하지 않는다. 공개 Dataset과 공개 CCTV 영상만 사용한다.

가능하면 서로 다른 조건의 영상을 준비한다: 실내/실외, 낮/밤, 저조도, 역광, 강한 인공조명, 사람이 적은/많은 장면, Occlusion이 많은 장면, 작은 사람이 많은 장면.

하나의 영상에서만 잘 작동하는 프로젝트로 만들지 않는다.

---

## 9. Phase 1 Video Input Pipeline

공개 mp4 영상을 입력으로 사용한다.

**구현**: 영상 파일 읽기, Frame 단위 처리, FPS 확인, 해상도 확인, Frame resize, 결과 영상 저장, 정상 종료 처리, 영상 읽기 실패 처리

**테스트**: 720p, 1080p, 서로 다른 FPS, 긴 영상, 짧은 영상, 손상된 영상, 사람이 없는 영상

**측정**: 원본 FPS, 처리 FPS, 전체 Frame 수, 평균 Frame 처리 시간, P50 Latency, P95 Latency, Dropped Frame

---

## 10. Phase 2 Person Detection

Pretrained 모델을 사용한다 (후보: YOLO11n, YOLO11s). 처음부터 모델을 직접 학습하지 않는다. 먼저 Pretrained 모델을 이용해 Baseline을 구축한다.

**Confidence Threshold 실험 예**: 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7
**Image Size 실험 예**: 640 / 960 / 1280

**측정**: Precision, Recall, F1, False Positive, False Negative, FPS, P50/P95 Latency, Model Size

**환경별 실패 사례 확인**: Small Object, Occlusion, Crowded Scene, Low Light, Backlight, 강한 인공조명, 복잡한 배경

---

## 11. Phase 3 Multi Object Tracking

후보: ByteTrack, BoT-SORT. 처음부터 둘을 모두 붙이지 않는다. 하나를 Baseline으로 먼저 적용한다.

**관찰할 문제**: ID Switch, Track Fragmentation, Occlusion, 사람 간 교차, 화면 밖으로 나갔다가 재등장, Crowded Scene

**측정**: ID Switch, IDF1, HOTA, MOTA, Track Fragmentation, FPS, P95 Latency

그 뒤 문제를 바탕으로 두 번째 Tracker 또는 Parameter 조정을 대안으로 검토한다. Tracking 정확도와 처리 속도의 Trade-off를 기록한다.

---

## 12. Phase 4 BestShot

한 사람을 추적하는 동안 가장 좋은 대표 Frame을 선택한다.

- Baseline: Random Frame
- 대안 A: Highest Detection Confidence
- 대안 B: 자체 BestShot Score

**고려 후보**: Detection Confidence, Bounding Box Size, Sharpness, Occlusion, Motion Blur, Frame Boundary, Object Position

```
BestShot Score =
  Confidence Score
  + Sharpness Score
  + Object Size Score
  + Occlusion Score
  + Position Score
```

가중치는 처음부터 확정하지 않고 실험을 통해 결정한다. 비교 결과를 Image Grid로 저장한다.

**Track마다 저장할 Metadata**: track_id, best_frame, timestamp, confidence, bbox, sharpness, bestshot_score

---

## 13. Phase 5 Intrusion Detection

Polygon ROI 기반으로 구현한다.

**비교 후보**: Bounding Box Center, Bottom Center, Polygon Overlap

**State Machine**: OUTSIDE → ENTER → INSIDE → EXIT

같은 객체에서 매 Frame 동일 이벤트가 생성되지 않도록 한다.

**측정**: Event Precision, Event Recall, Event F1, Duplicate Event Count, False Positive, False Negative

---

## 14. Phase 6 Line Crossing

Virtual Line을 설정하고, 이전/현재 좌표를 사용해 Line Crossing을 판단한다. 방향도 기록한다 (예: Entrance → Casino Floor, Casino Floor → Exit).

**테스트**: 정상 통과, 선 근처 정지, 선 위에서 움직임, 방향 전환, Tracking ID Switch, 동일 객체 중복 Event

---

## 15. Phase 7 Loitering Detection

- Baseline: Dwell Time > Threshold
- 개선 후보: Dwell Time + Movement Distance + ROI State + Track Stability

**Threshold 비교**: 5초 / 10초 / 20초 / 30초

**측정**: Precision, Recall, F1, False Positive, False Negative

단순 대기와 장시간 체류를 어느 정도 구분할 수 있는지 분석한다.

---

## 16. Metadata Pipeline

**객체별 최소 데이터**: track_id, class, confidence, bbox, first_seen, last_seen, bestshot_path, current_zone, dwell_time, trajectory, event_history

**추후 확장**: upper_color, lower_color, bag, hat, embedding

영상 자체를 검색하는 것이 아니라 Metadata를 기반으로 검색한다.

---

## 17. VMS Search

**최소 검색 형태**: 14:00~15:00 사이 등장한 사람, 특정 ROI에 일정 시간 이상 있었던 사람, 특정 선을 통과한 사람, 특정 Track ID의 이동 기록, 특정 Event 발생 시점, 특정 Event 전후 영상

**결과 표시**: BestShot, Timestamp, Track ID, Event Type, Camera ID, Video 위치

Backend는 최소 기능만 구현한다 (FastAPI 등 가벼운 구조). 다음에는 시간을 과도하게 사용하지 않는다: Microservice, Kubernetes, 복잡한 인증, Cloud Architecture, Frontend 디자인.

프로젝트 중심은 항상 Computer Vision과 CCTV Video Pipeline이다.

---

## 18. Person Re-ID (Core MVP 이후)

한화비전 카지노 사례에서 내부적으로 특정 Re-ID 모델을 사용한다고 주장하지 않는다. Multi Camera 환경을 위한 추가 심화 기능이다.

**후보**: OSNet, 다른 Pretrained Re-ID Model

**구조**: Person Crop → Re-ID Network → Embedding → Vector 저장 → Cosine Similarity → Similar Person Search

**측정**: Top-1, Top-5, mAP, Threshold별 Precision/Recall

**테스트**: 같은 사람 다른 각도, 조명 변화, Occlusion, 비슷한 옷을 입은 다른 사람

---

## 19. Attribute Metadata

필요하면 상의 색상, 하의 색상, 가방 여부, 모자 여부를 사용한다.

Attribute Classification 자체가 프로젝트 중심이 되어서는 안 된다. 검색 가능한 Metadata를 만드는 것이 목적이다. 카지노 환경처럼 조명이 강하거나 색상이 변하는 상황에서 Attribute 안정성도 분석한다.

---

## 20. People Counting

출입선을 이용해 Entry Count, Exit Count, Current Occupancy를 계산한다. Tracking ID를 이용해 중복 Count를 방지한다.

**측정**: MAE, Count Error, Duplicate Count, Direction Error

---

## 21. Heatmap

Track 위치를 시간에 따라 누적한다. Detection Center 누적 vs Track Trajectory 누적을 비교할 수 있다. 시간 단위: 전체 / 1분 / 5분. 사람이 많이 머무는 공간을 시각화한다.

---

## 22. Crowd Analysis

ROI별 Person Count를 계산한다 (Normal / Busy / Crowded). Threshold는 임의로 정하지 않고, 실제 Dataset의 Distribution을 확인한 뒤 정의한다.

---

## 23. Adaptive Recording

항상 고화질로 저장하는 방식과 Event 기반 저장을 비교한다.

- Baseline: 30 FPS High Quality Recording
- Adaptive 예: No Person → Low FPS / Person Detected → Normal FPS / Security Event → High FPS·High Quality

가능하면 Circular Buffer를 이용한다 (Event -10초 ~ +10초).

**측정**: 저장 용량, CPU Usage, File Write Count, Event Clip 누락, Storage Saving %

---

## 24. Edge Optimization

**후보**: YOLO11s PyTorch FP32, YOLO11n PyTorch FP32, YOLO11n ONNX, 가능하면 INT8

MacBook Air M3 환경이므로 CUDA와 TensorRT를 필수로 하지 않는다. TensorRT는 NVIDIA Edge Device 확장 계획으로 남길 수 있다.

**측정**: Precision, Recall, FPS, P50/P95/P99, Model Size, Memory Usage

**핵심 Trade-off**: Accuracy vs Latency vs Model Size

---

## 25. 한화비전 학사 SW 직무 연결

프로젝트를 AI 연구 프로젝트로만 만들지 않는다. 다음 역량을 함께 보여줄 수 있게 한다: C++, Linux, Multi Thread, 영상 처리, Video Codec, FFmpeg, 영상 Metadata, VMS, REST API, 실시간 영상 처리, 장시간 안정성, AI 모델 적용, 성능 최적화

Python으로 Baseline을 먼저 완성한 뒤, 적절한 시스템 모듈만 C++로 구현한다.

**후보**: Video Frame Reader, ROI 계산, Line Crossing, Event State Machine, Metadata Serialization, Thread-safe Frame Queue

전체 딥러닝 Pipeline을 억지로 C++로 다시 작성하지 않는다.

---

## 26. Multi Thread Pipeline

Single Thread Baseline을 만든 뒤 필요성이 확인되면 다음 구조를 실험한다.

```
Video Decode Thread → Frame Queue → AI Inference Thread → Event Processing Thread → Storage Thread
```

**비교**: Single Thread vs Multi Thread

**측정**: FPS, P50/P95 Latency, Queue Length, Dropped Frame, CPU Usage, Memory Usage

실제로 병목이 존재하지 않으면 억지로 Multi Thread를 도입하지 않는다.

---

## 27. FFmpeg

OpenCV VideoCapture만 사용하고 끝내지 않는다. 프로젝트 후반에는 FFmpeg를 사용해 Codec 확인(H.264/H.265), Decode/Encode, Frame Extraction, Event Clip 생성을 경험한다.

Codec과 영상 Pipeline이 실제 CCTV 시스템에서 왜 중요한지도 정리한다.

---

## 28. 장시간 안정성 테스트

CCTV 시스템은 장시간 실행된다. 반드시 Long Running Test를 한다 (10분/30분/1시간/가능하면 그 이상).

**측정**: Memory 사용량 변화, FPS 변화, P95 Latency 변화, Queue 적체, Dropped Frame, Event 중복, DB 증가량, 저장 파일 증가량, 예외 발생 횟수

메모리가 지속적으로 증가한다면 원인을 분석하고 개선한다.

---

## 29. Robustness Test

정상 영상만 사용하지 않는다. Frame 읽기 실패, 손상된 영상, 사람이 없는/매우 많은 영상, Small Object, Occlusion, Motion Blur, Low Light, Backlight, 낮은 FPS, 해상도 변화도 테스트한다.

프로그램이 특정 입력 하나 때문에 전체적으로 종료되지 않도록 한다.

---

## 30. Regression Test

고정 Evaluation Video를 만든다 (예: test_normal.mp4, test_crowded.mp4, test_occlusion.mp4, test_intrusion.mp4, test_line_crossing.mp4, test_loitering.mp4).

새 기능 또는 Parameter를 변경할 때 동일 영상으로 다시 테스트하여 이전 기능이 깨지는지 확인한다.

---

## 31. Ground Truth

수치를 절대 만들어내지 않는다. 공개 Dataset에 Label이 있다면 공식 Ground Truth를 사용한다. Event Detection처럼 Ground Truth가 없다면 짧은 Evaluation Set을 직접 만든다 (예: evaluation/intrusion/, evaluation/line_crossing/, evaluation/loitering/).

각 영상에 Event 발생 시점과 객체를 직접 Labeling하여 TP, FP, FN, Precision, Recall, F1을 계산한다. 정확하게 측정할 수 없는 Metric은 억지로 작성하지 않는다.

---

## 32. 테스트 코드

내가 직접 작성한 Business Logic과 Event Logic은 가능하면 Unit Test를 작성한다.

**우선 대상**: ROI 내부 판정, Line Crossing, Line Crossing 방향, Dwell Time, Event 중복 방지, Track State 삭제, Event State Transition, BestShot Score, Metadata 변환

AI 모델 자체보다 직접 작성한 로직을 테스트하는 것을 우선한다. Python 단계에서는 pytest를 사용할 수 있다. C++ 단계에서는 적절한 테스트 도구를 선택한다.

---

## 33. 실험 기록 구조 (로컬)

```
data/
    raw/
    test/
src/
tests/
configs/
experiments/
results/
    videos/
    snapshots/
    plots/
    failures/
evaluation/
```

대용량 Dataset과 모델 파일은 Git에 직접 Commit하지 않는다.

---

## 34. 실험 파일

각 실험은 고유 번호를 사용한다 (예: EXP-001 Detection Baseline, EXP-002 Confidence Threshold, EXP-003 Resolution Comparison, EXP-004 ByteTrack Baseline, EXP-005 Tracker Comparison).

각 실험에는 가능하면 다음을 남긴다: experiment.md, metrics.csv, config.yaml, result.mp4, failure_cases/, plot.png

**experiment.md 기본 구조**: Goal, Hypothesis, Problem, Dataset, Environment, Configuration, Baseline, Result, Failure Cases, Analysis, Decision, Next Action

---

## 35. Config 관리

Threshold와 Parameter를 코드에 흩뿌리지 않고 Config로 관리한다 (예: model_name, confidence_threshold, iou_threshold, image_size, track_buffer, loitering_seconds, roi, line_position).

각 실험에서 어떤 Config를 사용했는지 남긴다.

---

## 36. Metric 기록

가능한 경우 평균 하나만 기록하지 않는다. Precision, Recall, F1, ID Switch, IDF1, HOTA, Average FPS, Latency(P50/P95/P99), Memory(Average/Peak), CPU Usage, Dropped Frame, Event FP, Event FN 등을 고려한다.

모든 Metric을 억지로 측정할 필요는 없다. 현재 문제를 설명하는 데 의미 있는 Metric을 선택한다.

---

## 37. Before / After

개선 전후는 반드시 동일한 Dataset과 환경에서 측정한다.

```
Metric              Before        After
ID Switch           34            19
P95 Latency         82ms          51ms
False Positive      18            7
```

다음을 기록한다: 무엇을 변경했는가? 왜 변경했는가? 개선된 부분은 무엇인가? 새로운 Trade-off는 무엇인가?

---

## 38. 실패한 실험

실패한 실험을 삭제하지 않는다.

예:
```
가설: BoT-SORT가 ByteTrack보다 Occlusion 상황의 ID Switch를 줄일 것이다.
결과: ByteTrack ID Switch: 18 / BoT-SORT ID Switch: 25
결론: 현재 설정과 Dataset에서는 가설이 맞지 않았다.
```

그다음 왜 그런 결과가 나왔는지 분석한다. 성과를 위해 숫자를 조작하거나 실패 결과를 숨기지 않는다.

---

## 39. PAR 경험

중요한 문제 해결이 끝날 때마다 반드시 다음 9개 항목으로 정리한다. 형식과 질문은 변경하지 않는다.

1. **한줄 요약** — 문제와 핵심 해결 방법과 결과가 한 문장에 들어가도록 작성한다.
2. **문제가 뭐였는가?** — 실제 Test 또는 구현 과정에서 발견한 문제를 구체적으로 작성한다.
3. **왜 이 문제가 중요했는가?** — CCTV 영상보안에서 왜 중요한지 설명한다. 카지노 환경과도 연결한다. 다른 기능에 연쇄적으로 어떤 영향을 미치는지도 작성한다.
   예: Tracking ID Switch → Loitering State 초기화 → 동선 분석 오류 → Metadata 검색 정확도 저하
4. **원인이 뭐였는가?** — 로그와 Metric, 영상 결과, 코드 분석 등에 근거해 작성한다. 추측만으로 확정하지 않는다.
5. **A를 고민하였는데 왜 안 했는가?** — 검토한 첫 번째 해결 방법과 장점을 작성한다. 왜 최종 선택하지 않았는지 작성한다 (정확도, Latency, 복잡도, 유지보수성, Edge 적용 가능성, 메모리, 개발 비용 등 기준).
6. **B를 고민하였고 왜 적용했는가?** — 최종적으로 선택한 방식과 이유를 작성한다. A보다 왜 현재 문제에 적합했는지 설명한다.
7. **어떤 방식으로 검증했는가?** — Dataset, Test Video, 환경, Configuration, Metric, Baseline, 비교 방식, Regression Test를 구체적으로 작성한다.
8. **개선된 수치적인 결과가 무엇이었는가?** — 실제 측정값만 작성한다. 정량 결과를 측정할 수 없었다면 억지로 숫자를 만들지 않고 정성적으로 무엇을 검증했는지 작성한다.
9. **재발 방지 대책** — 문제를 한 번 수정하고 끝내지 않는다 (Regression Test, 고정 Evaluation Dataset, Config 관리, Unit Test, Logging, 상태 관리 구조 개선, Boundary Test, Monitoring 등).

---

## 40. PAR 후보 조건

아무 기능이나 PAR로 만들지 않는다. 다음 조건을 만족할 때 PAR 후보로 판단한다:

- 실제 문제가 있었다.
- 문제의 원인을 분석했다.
- 최소 두 개의 해결 방향을 생각했다.
- 선택 이유가 있었다.
- 실제 구현 또는 실험을 했다.
- 결과를 검증했다.
- 배운 점 또는 재발 방지책이 있다.

**좋은 PAR 후보 예**: Tracking ID Switch 감소, BestShot 품질 개선, Loitering False Positive 감소, Line Crossing 중복 Event 제거, Video Pipeline Latency 개선, 장시간 실행 Memory 증가 문제 해결, Multi Thread Pipeline 병목 개선, Adaptive Recording 저장량 감소

---

## 41. Git 원칙

Git Commit은 프로젝트가 어떻게 발전했는지 알 수 있도록 의미 있는 단위로 작성한다.

예: `feat: add person detection baseline`, `feat: integrate ByteTrack`, `test: add tracking regression videos`, `feat: implement bestshot scoring`, `fix: prevent duplicate intrusion events`, `perf: optimize video processing latency`

한 Commit에 너무 많은 기능을 섞지 않는다.

---

## 42. Notion 기록 위치

Claude에 연결된 Notion Connector를 사용한다. 모든 프로젝트 Engineering Log는 다음 페이지에 기록한다.

https://app.notion.com/p/Tb-3e05323e448380efa562c79e81f3f6a4?source=copy_link

이 페이지를 CasinoGuard AI 프로젝트의 기준 기록 공간으로 사용한다. 대화 안에서만 결과를 정리하고 끝내지 않는다. 중요한 조사, 실험, 실패, 기술 선택, Metric, PAR 경험, 진행 상태를 실제 Notion에도 기록한다.

기존 내용을 임의로 삭제하지 않는다. 동일한 항목이 존재하면 중복 생성하지 말고 기존 내용을 업데이트한다.

> **주의**: 위 Notion 페이지는 아직 Claude Notion Connector에 공유(Share) 되지 않아 접근이 안 되는 상태다 (404 Not Found). 사용자가 Notion 페이지 우측 상단 Share → Connections에서 연동을 추가해야 한다.

---

## 43. Notion 기본 구조

프로젝트를 시작할 때 위 Notion 페이지 아래에 다음 구조를 만든다.

```
CasinoGuard AI
00. Project Overview
01. Architecture & Roadmap
02. Hanwha Vision Research
03. Dataset & Evaluation
04. Experiment Log
05. Failure Cases
06. Performance Benchmark
07. PAR Experience
08. Technical Decisions
09. Interview Notes
10. Final Portfolio
```

---

## 44. Project Overview 기록

다음을 유지한다: 프로젝트 목적, 카지노 CCTV를 선택한 이유, 목표 직무, 한화비전과의 연결, Core MVP, Stretch Goal, Architecture, 사용 기술, 현재 Phase

프로젝트 방향이 변경되면 최신 상태로 업데이트한다.

---

## 45. Hanwha Vision Research 기록

한화비전 관련 자료를 조사하면 다음 형식으로 기록한다: 기능명, 한화비전 공식 자료 링크, 공개된 내용, 카지노에서 왜 필요한가, 내 프로젝트에서 재현할 문제, 내가 선택한 구현 기술, 주의사항

한화비전의 공식 자료를 우선한다. 블로그나 제3자 자료보다 공식 홈페이지, 공식 백서, 공식 제품 자료, 공식 채용공고를 우선한다.

---

## 46. Experiment Log

각 실험은 Notion의 04. Experiment Log에 기록한다.

제목 예: EXP-001 Detection Baseline, EXP-002 Confidence Threshold, EXP-003 Resolution Comparison, EXP-004 ByteTrack Baseline

내용: Goal, Hypothesis, Problem, Dataset, Environment, Configuration, Baseline, Result, Failure Cases, Analysis, Decision, Next Action

---

## 47. Failure Cases

성공 사례만 기록하지 않는다. 05. Failure Cases에 실패 사례를 누적한다.

**분류 예**: Small Object, Occlusion, Crowded Scene, Low Light, Backlight, Motion Blur, ID Switch, Track Fragmentation, ROI Boundary, Duplicate Event, Attribute Error, Re-ID False Match

**각 실패 사례**: 현상, 재현 조건, 원인 가설, 확인 방법, 검토한 대안, 실험 결과, 최종 처리

---

## 48. Technical Decisions

기술 선택은 08. Technical Decisions에 기록한다.

예: ByteTrack vs BoT-SORT, Bounding Box Center vs Bottom Center, OpenCV VideoCapture vs FFmpeg, PyTorch vs ONNX, Single Thread vs Multi Thread, SQLite vs PostgreSQL

**각 Decision**: 문제, 선택지 A, A 장점, A 단점, 선택지 B, B 장점, B 단점, 최종 선택, 선택 근거, 검증 결과

이 내용은 PAR 5번과 6번의 근거가 된다.

---

## 49. Notion PAR 기록

PAR 경험이 생기면 07. PAR Experience에 저장한다. 각 PAR에는 관련된 실험 번호도 연결한다.

예:
```
PAR-001 혼잡 영상 ID Switch 개선
관련 실험: EXP-004 ByteTrack Baseline, EXP-005 BoT-SORT Comparison, EXP-006 Track Buffer Test
```

그 아래에는 앞에서 정의한 PAR 9문항을 그대로 사용한다. 실제로 하지 않은 실험이나 의사결정은 추가하지 않는다.

---

## 50. Git과 Notion 역할

**GitHub**: 실제 코드, Commit, 테스트 코드, Config, 실험 Script, README

**Notion**: 문제 발견, 의사결정, 실험 목적, 실험 결과, 실패, 분석, Metric, PAR, 면접용 정리

Git과 Notion을 서로 보완적으로 사용한다. 가능하면 Commit과 Experiment 번호를 연결한다.

---

## 51. Roadmap 상태 관리

01. Architecture & Roadmap에서 진행 상태를 계속 갱신한다.

예:
```
현재 Phase: Phase 3 Multi Object Tracking
완료: [x] Video Input  [x] Person Detection  [ ] Tracking  [ ] BestShot  [ ] Intrusion  [ ] Line Crossing  [ ] Loitering
현재 문제: Crowded Scene에서 ID Switch 발생
다음 Action: ByteTrack Baseline Metric 측정
```

---

## 52. Phase 완료 조건

코드가 실행됐다는 이유만으로 Phase를 완료하지 않는다. 다음 조건을 만족해야 한다: 기능 구현, 정상 Case 테스트, Failure Case 테스트, Metric 측정, 실패 사례 수집, 실험 결과 분석, Notion Experiment Log 기록, 재현 가능한 실행 방법 기록, 결과 영상 또는 이미지 저장, Git Commit

필요한 경우 Regression Test까지 수행한다.

---

## 53. Claude의 개발 진행 방식

각 Phase마다 반드시 다음 순서로 진행한다:

1. 이번 Phase 목표
2. 한화비전 영상보안과의 연결
3. 카지노 환경에서 왜 중요한가
4. Baseline 설계
5. 구현
6. 테스트
7. Metric 측정
8. Failure Case 확인
9. 문제 분석
10. 대안 A
11. A를 적용하지 않는 이유
12. 대안 B
13. B를 선택한 이유
14. 개선 구현
15. 동일 조건 재측정
16. Regression Test
17. PAR 후보 여부 판단
18. Notion 기록

현재 Phase를 완료하기 전에는 다음 Phase 구현으로 넘어가지 않는다.

---

## 54. 코드를 설명하는 방식

전체 프로젝트 코드를 한 번에 주지 않는다. 현재 Phase에 필요한 코드만 작성한다.

내가 코드를 보여주고 오류나 문제를 질문하면 바로 수정 코드부터 주지 않는다. 먼저 다음 순서로 설명한다: 문제 위치, 왜 문제가 발생했는가, 현재 구조에서 어떤 문제가 있는가, 가능한 해결 방법은 무엇인가, 어떤 해결 방법을 선택할 것인가, 왜 그것을 선택하는가, 수정 코드, 검증 방법

내가 이해하면서 프로젝트를 진행할 수 있도록 한다.

---

## 55. 실험 전 가설

모든 중요한 실험 전에 가능한 경우 Hypothesis를 기록한다.

예:
```
Hypothesis: BoT-SORT의 Appearance 정보가 Occlusion 이후 동일 객체 연결에 도움을 주므로
Crowded Scene의 ID Switch가 감소할 것이다.
```

결과가 가설과 반대라도 그대로 기록한다.

---

## 56. 프로젝트에서 확보할 문제 해결 경험

최종적으로 최소 3개 이상의 깊이 있는 PAR 경험을 확보하는 것이 목표다. 단, 억지로 만들지 않는다.

**가능한 후보**: Tracking ID Switch 개선, BestShot 품질 개선, Loitering False Positive 감소, Line Crossing 중복 Event 제거, P95 Video Pipeline Latency 개선, Long Running Test Memory 증가 문제 해결, Multi Thread Pipeline 처리량 개선, Adaptive Recording 저장량 감소

---

## 57. 최종 README

최종 README는 다음 흐름으로 구성한다:

1. 프로젝트 소개
2. 카지노 CCTV 환경 문제 정의
3. 한화비전 영상보안 기술과의 연결
4. System Architecture
5. Dataset
6. Person Detection
7. Multi Object Tracking
8. BestShot
9. Event Analytics
10. Metadata Pipeline
11. VMS Search
12. Performance Benchmark
13. Failure Cases
14. 주요 기술 의사결정
15. 개선 전후 결과
16. Long Running Test
17. Edge 환경 고려
18. 프로젝트 한계
19. RTSP / ONVIF 확장 계획

단순 기능 나열보다 다음 흐름을 강조한다: 문제 → 원인 → 가설 → 대안 비교 → 선택 → 구현 → 검증 → 결과

---

## 58. 최종 면접 준비

프로젝트가 끝나면 최소 다음 질문에 실제 실험 근거를 가지고 답할 수 있어야 한다:

- 왜 카지노 CCTV를 주제로 선택했는가?
- 일반 Object Detection 프로젝트와 무엇이 다른가?
- 카지노 환경에서 Tracking이 왜 중요한가?
- Tracking에서 가장 많이 발생한 문제는 무엇이었는가?
- 왜 해당 Tracker를 선택했는가? 다른 Tracker는 왜 선택하지 않았는가?
- BestShot은 왜 필요한가? BestShot Score를 어떻게 설계했는가?
- Loitering을 어떻게 정의했는가?
- False Positive는 왜 발생했는가?
- 영상 Metadata가 왜 필요한가?
- VMS가 AI 결과를 어떻게 활용하는가?
- 성능은 어떤 Metric으로 측정했는가? P95 Latency를 왜 측정했는가?
- 가장 큰 실패 사례는 무엇인가?
- 어떤 대안을 검토했는가? 왜 최종 방법을 선택했는가?
- 실제로 무엇이 얼마나 개선됐는가?
- 장시간 실행했을 때 어떤 문제가 발생했는가?
- Edge Camera에 배포한다면 무엇을 최적화해야 하는가?
- Python과 C++의 역할을 어떻게 나눴는가?
- 해당 기능은 Edge Camera와 VMS 중 어디에 위치하는가?

---

## 59. Claude의 작업 종료 규칙

하나의 의미 있는 작업이 끝날 때마다 다음 순서로 처리한다:

1. 코드와 실행 결과 확인
2. 실제 Metric 확인
3. Experiment Log 업데이트
4. Failure Case가 있으면 기록
5. Technical Decision이 있으면 기록
6. PAR 후보인지 판단
7. PAR 후보라면 PAR 9문항 작성
8. Architecture & Roadmap 상태 업데이트
9. 다음 Action 기록

그 뒤 사용자에게는 다음만 간단히 알려준다: 이번에 완료한 것, 측정된 핵심 Metric, 발견한 문제, Notion에 기록한 내용, PAR 생성 여부, 다음 Action

---

## 60. 현재 첫 작업

아직 전체 구현 코드를 작성하지 않는다. 먼저 다음 작업을 수행한다:

1. 한화비전 공식 자료에서 카지노 영상보안에 실제로 사용되거나 직접 연관되는 기술을 조사한다.
2. BestShot, Object Detection, Tracking, Attribute Metadata, Event Analytics, People Counting, Heatmap, Edge AI 등 각 기능의 공식 근거를 정리한다.
3. 한화비전 학사 신입 SW 직무에서 요구하거나 우대하는 기술을 공식 또는 신뢰 가능한 채용 자료를 기준으로 정리한다.
4. 위 결과를 바탕으로 CasinoGuard AI의 Core MVP와 Stretch Goal이 적절한지 다시 검토한다.
5. 프로젝트에 적합한 공개 Dataset 후보를 조사한다.
6. Dataset마다 어떤 Phase와 Metric에 사용할 것인지 정한다.
7. 전체 Architecture를 최종 확정한다.
8. 로컬 프로젝트 폴더 구조를 설계한다.
9. Experiment Template을 만든다.
10. PAR 9문항 Template을 만든다.
11. 연결된 Notion의 아래 페이지에 프로젝트 기본 구조를 생성하고 조사 결과를 기록한다.
    https://app.notion.com/p/Tb-3e05323e448380efa562c79e81f3f6a4?source=copy_link
12. Phase 1 Video Input과 Phase 2 Person Detection의 첫 Baseline 실험 계획을 작성한다.

이 단계에서는 전체 코드를 작성하지 않는다. 먼저 조사 결과, Architecture, Dataset 선택, 실험 설계, Notion 구조를 보여준다. 내가 확인한 뒤 실제 구현을 시작한다.

---

## 61. 최우선 원칙

이 프로젝트의 목적은 **"YOLO를 사용해본 사람"** 이 되는 것이 아니다.

최종적으로 다음과 같은 개발자가 되는 것이 목표다:

> CCTV 영상에서 객체와 상황을 인지하고, 영상보안 시스템에서 발생하는 문제를 측정하고, 원인을 분석하고, 여러 해결 방법을 비교하고, 실제 데이터를 통해 검증하고, Edge Camera와 VMS를 고려해 시스템으로 구현할 수 있는 학사 신입 영상인지 / CCTV SW 엔지니어

프로젝트를 진행하면서 항상 이 목표를 기준으로 판단한다.
