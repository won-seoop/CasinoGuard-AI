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

### Adaptive Recording/Circular Buffer 실시간 파이프라인 통합 (Stretch, EXP-023, **PAR-014**)

EXP-017~022는 전부 Adaptive Recording/Circular Buffer를 독립된 스크립트로만 검증했다 — 실제
Detection+Tracking+BestShot+Metadata를 SQLite에 쌓는 파이프라인과는 한 번도 같은 루프에서
동작한 적이 없었다. 이번에 `OnlineTierClassifier`(신규, Two-Pass `compute_tier_sequence`를
프레임 단위 스트리밍으로 재현)로 실시간 단일 패스 통합을 구현해, Circular Buffer가 이미
실시간인 전체 파이프라인에 추가하는 오버헤드가 P50 기준 5.7%(3.8ms/66.5ms)에 불과함을
확인했다. 더 중요하게는, Pre-Roll을 "EVENT 구간이 끝난 뒤" 꺼내는 더 단순한 설계(대안 A)가
이 프로젝트의 실제 시나리오(Post-Roll 간격보다 가까운 반복 Intrusion으로 EVENT Tier가
261프레임 연속됨)에서 Pre-Roll의 47%(240/511프레임)를 잃어버린다는 것을 구현 전 실측으로
발견하고, "트리거 순간 즉시 추출"하는 방식(대안 B)을 채택해 511/511 완전한 Event Clip을
만들었다(`PAR-014`).

## 10. Metadata Pipeline (EXP-008)

Track(trajectory, dwell_time, bestshot_path 등) + Event(Intrusion/LineCrossing/Loitering)를 SQLite 스키마로 통합 저장. 164 Track, 115 BestShot, 190 Event.

저장되는 `dwell_frames`가 항상 0이었던 문제(**FC-006**)를 EXP-015에서 재현·수정했다. 원인은 단순 호출 순서가 아니라 Loitering의 "연속 체류" 상태를 VMS의 "누적 총 체류 시간" 집계에 그대로 재사용한 설계였음을 확인하고, 책임을 분리한 `DwellCounter`로 고쳤다(`PAR-006`).

## 11. VMS Search (EXP-008)

FastAPI로 최소 기능만 구현(Microservice/K8s/복잡한 인증 없음):
- `GET /tracks?min_dwell_sec=` — 특정 시간 이상 체류한 Track 검색
- `GET /tracks/{id}` — Track의 이동 기록 + Event 이력
- `GET /events?event_type=&has_clip=` — 특정 Event 발생 시점 / Event Clip 유무 검색
- `GET /events/{id}/clip` — Event 전후 영상(Event Clip) 재생/다운로드 (EXP-024, **PAR-015**)
- `GET /` — 최소 Dashboard (Dwell 상위 5개, 최근 Event 10개 + Event Clip 링크)

API가 raw SQL로 `events.detail`을 파싱하지 않은 채 그대로 반환해(**FC-011**) `event_clip_path`
(EXP-023에서 부착)가 사실상 쓸 수 없었고, 영상을 실제로 재생하는 엔드포인트도 없던 문제를
`MetadataStore.query_events()` 재사용 + `GET /events/{id}/clip`(FileResponse)으로 해결했다
(`PAR-015`).

## 11-1. Attribute Metadata - 상의/하의 색상 분류 (Stretch, EXP-025, **PAR-016**)

지침 19의 Attribute Metadata(상의/하의 색상)를 Baseline(bbox 전체 평균, 상/하 구분 불가)
→ 대안 A(고정 비율 상/하 분할 + 단순 평균) → 대안 B(상/하 분할 + 그림자/하이라이트/피부색
필터 + Hue 다수결)로 비교했다. 실제 YOLO11n Person Detection crop 5개(ultralytics 번들
`zidane.jpg`/`bus.jpg`)에 수동 Ground Truth를 라벨링하고, 조명 변형 4종(저조도/강한조명·
역광/난색·한색 Cast)으로 안정성을 측정했다.

대안 B 최초 구현이 오히려 Baseline보다 낮은 정확도(16% < 32%)를 내는 역설을 실측으로
발견 — 원인은 그림자/하이라이트 필터가 저채도인 진짜 검정 옷까지 항상 제외시켜, 클로즈업
crop에서 소수 피부색 픽셀만 남아 Hue 다수결을 오염시키는 버그였다. 피부색 필터와 "필터
통과 비율이 낮으면 신뢰하지 않고 원본으로 되돌아가는" 폴백으로 수정해 전체 Accuracy
32%(Baseline)→40%, 조명 안정성 35%→40%로 개선했다(`PAR-016`). n=5의 작은 평가
Dataset이라는 한계를 솔직히 명시하며, 아직 MetadataStore/VMS에는 연결하지 않았다
(검증 Dataset 확장 후 통합 예정, Next Action).

### FC-012 재검토 (EXP-026, **PAR-017**)

FC-012(검정 옷이 `strong_light`/`cool_cast`에서 `blue`로 오분류)를 고치려 Gray World
White Balance(대안 B)와 raw BGR Spread 기반 무채색 Shortcut(대안 A)을 비교했다. 전체
평균 Accuracy는 대안 B가 더 높았지만(0.44→0.48), FC-012가 실제로 겨냥하는 "GT=black"
35셀만 슬라이싱하면 Accuracy가 정확히 동일했다(15/35→15/35) — 전체 평균 개선은 전혀
다른(진짜 파란 옷) 셀이 좋아진 부수 효과였을 뿐, 겨냥한 문제는 조건별 재배치만
일어났다(`cool_cast`는 개선, `normal`/`strong_light`는 악화). 두 대안 모두 production
기본값으로 승격하지 않고 FC-012를 "미해결"로 유지하되, 원인을 ①채널 비율 Cast형과
②노출/대비 손실형으로 분리해 기록했다(`PAR-017`).

### coco128 기반 n 확장 재검증 + 무채색 신호 분리도 분석 (EXP-027, **PAR-018**)

n=5(zidane.jpg/bus.jpg) 평가 Dataset의 "작고 편중됐다"는 한계를 실제로 검증하기 위해,
coco128(실제 COCO 128장, `github.com/ultralytics/assets/releases/download/v0.0.0/
coco128.zip`로 egress 제약 없이 접근 가능함을 새로 확인)에서 person crop 21개를 수동
라벨링해 n을 늘려 재측정했다. 모든 method의 절대 Accuracy가 n=5 때보다 큰 폭으로
낮아졌다(Baseline 32%→15%, 대안 A 28%→16%, 방법 B 40%→26%[normal 조명만 33%], B_wb
48%→32%) — method 간 상대적 순위(Baseline≈A < B < B_wb)는 유지됐지만, 이전에 기록한
절대 수치는 "작은 Dataset에서의 상대 비교 전용"으로 재규정했다. 또한 Roadmap이 다음
후보로 제안했던 "무채색 확신도 점수 기반 unknown 표시"를 설계하기 전에 후보 신호(HSV
채도, raw BGR 채널 표준편차)가 achromatic/chromatic을 실제로 분리하는지 먼저 측정했더니
**둘 다 분리되지 않음**을 확인했다(achromatic 채도 최댓값 158.89 > chromatic 채도
최솟값 62.92) — 검증 없이 기능부터 설계하는 것을 막은 사례다(`PAR-018`).

### Segmentation Mask 기반 영역 분리 (EXP-028, **PAR-019**)

EXP-027이 다음 후보로 이월한 "더 근본적인 영역 분리 방법"을 검증했다. method b/b_wb는
고정 비율 사각형(side_margin_ratio)으로 배경을 근사적으로만 제외하는데, person Instance
Segmentation Mask(`yolo11n-seg.pt`)로 배경을 픽셀 단위로 정확히 제외하는 대안 C를 같은
n=21 Ground Truth로 비교했다. **White Balance 없이 영역만 정확히 잘랐을 뿐인데** Overall
Accuracy가 기존 최고(b_wb, 32.4%)를 넘어 35.2%, GT=black Accuracy는 52.7%로 올랐다 —
그러나 GT=black→blue 오분류율(FC-012의 표적 실패)은 cool_cast에서 오히려 50.0%(b_wb
22.7%보다 나쁨)로, Mask는 "배경 Bleed"는 해결해도 "조명 Cast로 인한 Hue 왜곡"은 전혀
해결하지 못함을 실측으로 확인했다. 두 원인이 독립적이라는 가설에 따라 Mask+White
Balance를 결합(c_wb)하니 Overall Accuracy 38.1%, GT=black Accuracy 55.5%로 모든 method
중 최고를 기록하면서 cool_cast 오분류율도 b_wb 수준(22.7%)을 유지했다 — 단 strong_light
(노출/대비 손실형 Cast)는 여전히 개선되지 않아 FC-012는 이번에도 미해결로 유지했다. 추가
비용(Segmenter 모델)은 추론 시간 1.29배, 모델 용량 +0.57MB로 측정했다(`PAR-019`).

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
| FC-003 | BestShot Position Score 경계 페널티 부정확 | **PAR-009로 수정 완료** |
| FC-004 | Track 소실 시 EXIT 유실 | **PAR-002로 수정 완료** |
| FC-005 | BestShot 후보 무제한 누적 Memory Leak | **PAR-004로 수정 완료** |
| FC-006 | Track 체류시간(dwell_frames)이 전부 0으로 저장됨 | **PAR-006로 수정 완료** |
| FC-007 | 선 근처 박스 흔들림으로 Line Crossing 왕복 중복 이벤트 | **PAR-005로 수정 완료** |
| FC-008 | Adaptive Recording Event Clip Pre-Roll이 IDLE Tier까지 파고들면 Naive Stream Copy가 조용히 불완전한 Clip을 만듦 | **PAR-010로 수정 완료** |
| FC-009 | Circular Buffer가 원본 BGR 프레임을 무압축 버퍼링해 채널당 628~753MB 필요 | **PAR-012로 수정 완료** |
| FC-010 | Adaptive Recording+Circular Buffer 통합 시 Post-Roll이 영상/세션 종료 전에 끝나지 않으면 Event Clip이 잘림 | `truncated_at_video_end` 플래그로 명시 처리(EXP-023), Edge Camera 재시작 시 재현 가능성은 백로그 |
| FC-011 | VMS Search API가 raw SQL로 `event_clip_path`가 담긴 `detail`을 파싱하지 않고 그대로 반환 + Event Clip 재생 엔드포인트 자체가 없었음 | **PAR-015로 수정 완료** |
| FC-012 | Attribute 색상 분류에서 검정 옷이 `strong_light`/`cool_cast` 조명 하에서 반복적으로 `blue`로 오분류됨 | **EXP-026/PAR-017로 재검토**: Gray World WB는 `cool_cast`(채널 비율 Cast)만 부분 개선하고 `strong_light`(노출/대비 손실)는 악화시킴 — 겨냥한 검정 GT 셀 전체 Accuracy는 변화 없음(15/35→15/35). **EXP-027/PAR-018**: 제안된 "무채색 확신도" 대안도 신호(채도/raw BGR std)가 achromatic/chromatic을 분리 못 해 보류. **EXP-028/PAR-019**: Segmentation Mask(대안 C)도 `cool_cast`를 단독으로는 개선 못 함(50.0%, b_wb보다 나쁨) — Mask+WB 결합(c_wb)으로 `cool_cast`는 b_wb 수준(22.7%) 유지하지만 `strong_light`는 그대로. 미해결 유지 |

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
- **PAR-009**: BestShot Position Score의 margin_ratio 근접 판정이 실제 경계 접촉과 달라 부정확한 문제(FC-003) → boundary_eps(2px) 실제 접촉 판정으로 교체 (근접-비잘림 구간 False Positive Rate 91.7%→25.0%)
- **PAR-010**: Adaptive Recording Event Clip 재인코딩 낭비 → FFmpeg Stream Copy Hybrid로 개선하는 과정에서 Naive Stream Copy가 IDLE Tier와 겹치는 Pre-Roll을 조용히 누락시키는 문제(FC-008)를 발견해 IDLE 겹침 구간만 재인코딩하도록 수정 (완전성 유지, Clip 생성 CPU 시간 74.6~85.6% 절감)
- **PAR-011**: 지침 23 Circular Buffer를 실제 구현하는 과정에서 버퍼 용량=Pre-Roll 길이로 설정하면 트리거 프레임 자신이 evict되어 정확히 1프레임이 모자라는 Off-by-One 발견 → `required_capacity_for_pre_roll()`(pre_roll+1+안전마진)로 수정
- **PAR-012**: Circular Buffer가 원본 BGR 프레임을 무압축 버퍼링해 채널당 628~753MB가 필요한 문제(FC-009) → push/pop 경계에서 JPEG 인코딩/디코딩을 추가해 메모리 90.8% 절감(789.94MB→72.47MB), Event Clip 완전성과 화질(MAE 2.61/255) 유지
- **PAR-013**: Circular Buffer quality를 Uniform하게 낮추면(대안 A) 실제 사고(EVENT) 프레임 화질을 가장 많이 희생시키는 문제 발견(EVENT MAE 2.59→5.40) → EVENT quality만 보존하는 Tier 차등 방식(대안 B)으로 EVENT MAE를 Baseline보다 개선(1.34)하면서 IDLE 메모리 41.1% 추가 절감
- **PAR-014**: Adaptive Recording/Circular Buffer를 실시간 Detection+Tracking+Metadata 파이프라인에 처음 통합하며, Pre-Roll을 EVENT 종료 후 꺼내는 방식(대안 A)이 실제 시나리오에서 47%(240/511프레임) 손실됨을 구현 전 실측 발견 → 트리거 순간 즉시 추출하는 방식(대안 B)으로 511/511 완전성 확보, Circular Buffer 통합 오버헤드는 전체 파이프라인의 P50 5.7%로 측정
- **PAR-015**: VMS Search API가 `MetadataStore`를 거치지 않고 raw SQL로 `event_clip_path`가 담긴 `detail` JSON을 파싱 없이 반환해(FC-011) 영상 재생이 불가능했던 문제 → API가 `MetadataStore.query_events()`를 재사용하도록 바꾸고 `GET /events/{id}/clip`(FileResponse)을 신설해 해결 (실제 EXP-023 산출물로 Event Clip 27,033,338 bytes를 byte-for-byte 서빙 확인, has_clip 필터로 ENTER 4건/EXIT 4건 정확히 분리)
- **PAR-016**: Attribute 색상 분류(대안 B)의 그림자/하이라이트 필터가 저채도인 진짜 검정 옷까지 항상 제외시켜, 클로즈업 crop에서 소수 피부색 픽셀만 남아 Hue 다수결을 오염시키는 버그 발견(대안 B 최초 구현이 Baseline보다 낮은 Accuracy를 내는 역설로 실측) → 피부색 필터 + 필터 통과 비율 기반 폴백으로 수정 (전체 Accuracy 16%→40%, Baseline(32%)/대안 A(28%) 모두 상회, 조명 안정성 15%→40%)
- **PAR-017**: FC-012(검정 옷→blue 오분류) 수정을 위해 Gray World White Balance를 추가했더니 전체 평균 Accuracy는 올랐지만(0.44→0.48), FC-012가 겨냥하는 "GT=black" 35셀만 슬라이싱하면 Accuracy가 정확히 동일했음(15/35→15/35) → 평균 개선은 무관한 셀(진짜 파란 옷)의 부수 효과였음을 발견하고 production에 반영하지 않음(가설 기각을 정확한 Metric 슬라이싱으로 검증)
- **PAR-018**: n=5 평가 Dataset의 Attribute Accuracy(32~48%)를 coco128 실제 crop 21개로 재검증하니 15~32%로 하락(method 순위는 유지) → 작은 Dataset 수치를 "운영 가능한 성능"으로 재인용하지 않기로 함. 동시에 "무채색 확신도" fallback 설계 전에 후보 신호(채도/raw BGR std)의 achromatic/chromatic 분리도를 먼저 측정해 둘 다 분리되지 않음을 확인하고 구현을 보류(검증 없는 기능 추가 방지)
- **PAR-019**: Attribute 색상 분류의 고정 비율 사각형 영역이 배경을 구조적으로 포함하는 문제 → person Segmentation Mask로 배경을 픽셀 단위 제외(대안 C)하니 White Balance 없이도 Overall Accuracy가 기존 최고(b_wb 32.4%)를 넘어 35.2%로 개선, 단 FC-012의 표적 실패(cool_cast 흑→blue)는 그대로임을 확인(50.0%, b_wb보다 나쁨) → Mask+WB 결합(c_wb)으로 Overall/GT=black Accuracy 최고치(38.1%/55.5%) 달성하면서 cool_cast 오분류율도 b_wb 수준 유지(두 기법이 독립적인 원인을 해결함을 실측으로 확인)

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

# 단위 테스트 (197개, 7 skipped)
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
python scripts/run_exp018_bestshot_position_score_fix.py        # FC-003 재현 + boundary_eps 수정 검증 (PAR-009)
python scripts/run_exp019_event_clip_copy.py                    # FC-008 재현 + FFmpeg Stream Copy Hybrid (PAR-010)
python scripts/run_exp020_circular_buffer.py                    # Circular Buffer Off-by-One 발견/수정 (PAR-011)
python scripts/run_exp021_compressed_circular_buffer.py         # FC-009 재현 + JPEG 압축 버퍼링 (PAR-012)
python scripts/run_exp022_tier_quality_circular_buffer.py       # Circular Buffer Tier별 quality 차등 (PAR-013)
python scripts/run_exp023_pipeline_integration.py               # Adaptive Recording+Circular Buffer 실시간 파이프라인 통합, Pre-Roll 추출 시점 A/B (PAR-014)
python scripts/run_exp025_attribute_color.py                    # Attribute 상/하의 색상 분류 Baseline/A/B + 조명 안정성 (PAR-016)
python scripts/run_exp026_fc012_white_balance.py                 # FC-012 재현 + White Balance/Achromatic Shortcut 대안 비교 (PAR-017)
python scripts/run_exp027_attribute_coco128.py                   # coco128 기반 n 확장 재검증 + 무채색 신호 분리도 분석 (PAR-018)
python scripts/run_exp028_attribute_segmentation_mask.py         # Segmentation Mask 기반 영역 분리 대안 C + Mask/WB 결합 (PAR-019)

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
