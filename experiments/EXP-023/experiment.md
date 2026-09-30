# EXP-023 Adaptive Recording + Circular Buffer를 실시간 단일 패스 파이프라인에 통합

관련 실험: EXP-017(Adaptive Recording Baseline, PAR-008), EXP-019(FFmpeg Hybrid, FC-008, PAR-010),
EXP-020(Circular Buffer, PAR-011), EXP-021(JPEG 압축, FC-009, PAR-012), EXP-022(Tier별 quality, PAR-013),
EXP-008(Metadata Pipeline/run_full_pipeline.py)

## Goal
지금까지 EXP-017~022는 전부 "Adaptive Recording/Circular Buffer 로직만" 별도 스크립트로
검증했다 — 실제 Detection+Tracking+ROI/BestShot/Metadata를 SQLite에 쌓는 `run_full_pipeline.py`
계열과는 한 번도 같은 루프 안에서 동작한 적이 없다(01. Architecture & Roadmap에 반복
기록된 미해결 항목). 이 실험은 그 통합을 실제로 구현하고 두 가지를 실측으로 답한다.

1. Circular Buffer(JPEG 인코딩+push)를 이미 실시간 처리 중인 전체 파이프라인(Detection+
   Tracking+ROI+BestShot+Metadata)에 매 프레임 추가로 얹으면, 그 오버헤드는 실제로 얼마인가?
2. Pre-Roll을 "언제" Circular Buffer에서 꺼내야 하는가? — 지금까지의 실험은 전부 trigger_frame을
   스크립트에 미리 하드코딩해 두고 그 순간에만 꺼냈다. 실제 파이프라인은 Event가 언제 발생할지
   미리 모른다(Two-Pass가 아니라 진짜 온라인). 그리고 "언제 버퍼에서 꺼내는가"의 선택 자체가
   Correctness에 영향을 준다는 것을 이번에 실측으로 확인했다.

## Hypothesis
1. JPEG 인코딩+Circular Buffer push는 YOLO Detection+Tracking(수십 ms)에 비해 훨씬 가벼우므로
   (EXP-021에서 인코딩 자체는 프레임당 수 ms), 전체 파이프라인 P50/P95 Latency에 추가하는
   오버헤드 비율은 한 자릿수 %대일 것이다.
2. Pre-Roll을 "EVENT Tier가 끝난 뒤(Post-Roll 만료 시점)"에 꺼내는 방식(대안 A, 구현이 더
   단순함)도 Circular Buffer 용량이 충분히 크면 문제없을 것이라고 가정할 수 있지만, 이
   프로젝트의 실제 시나리오(짧은 간격으로 반복되는 Intrusion이 EVENT Tier를 길게 연장시킴,
   EXP-017/022에서 이미 확인된 패턴)에서는 EVENT Tier 구간 자체가 Circular Buffer 용량보다
   길어져 Pre-Roll 프레임이 이미 evict된 뒤일 것이라고 예상했다.

## Problem
`run_full_pipeline.py`는 `data/raw/crowded_intersection_1080p.webm`을 입력으로 쓰는데,
이 원격 세션은 egress 정책상 Wikimedia 도메인이 차단되어 그 영상을 받을 수 없다(EXP-010부터
반복 확인된 동일 제약). 원본 영상 없이도 "두 파이프라인을 하나로 합쳤을 때 실제로 벌어지는 일"을
검증하려면, Detection이 동일하게 재현되는 입력이 필요했다 — EXP-017이 이미 검증해 둔 bus.jpg+
Pan/Jitter 3-Phase(IDLE/NORMAL/EVENT, ROI 중앙 대역 x=[430,620]) 시나리오를 재사용했다.

## Dataset
- `data/raw/bus.jpg` (ultralytics 패키지 내장 실제 사진, 810×1080, 사람 4명 이상 포함).
- 900프레임 합성 시나리오 (EXP-017과 동일): Phase 1 [0,300) IDLE, Phase 2 [300,600) NORMAL,
  Phase 3 [600,900) EVENT(사람 무리가 ROI를 두 차례 들락날락하도록 진폭 300px/주기 150프레임
  수평 이동 추가).
- Detection/Tracking은 매 프레임 실제 YOLO11n(`yolo11n.pt`, GitHub Release Asset 경로로 접근
  가능함을 이번 세션에서 재확인)+ByteTrack으로 수행했다(캐시 재사용 아님 — 이번 실험은
  Two-Pass가 아니라 진짜 온라인 단일 패스이므로 미리 캐시된 detection 결과를 쓸 수 없다).

## Environment
- 원격 자동화 세션, `.venv`(Python 3.11.15) 신규 생성 (opencv-python-headless, ultralytics,
  numpy, pytest, lap).
- CPU 전용, 프레임 810×1080 BGR uint8.

## Configuration
```yaml
pre_roll_sec: 10.0   # pre_roll_frames = 250 @ 25fps
post_roll_sec: 10.0  # post_roll_frames = 250
buffer_capacity: 271  # required_capacity_for_pre_roll(250, safety_margin=20)
tier_quality:          # EXP-022 채택값을 이번 세션부터 기본값으로 승격
  idle: 50
  normal: 75
  event: 95
roi_polygon: 중앙 대역 x=[430,620] (EXP-017과 동일)
detector: yolo11n.pt, conf=0.4, imgsz=640, tracker=bytetrack.yaml
```

## Baseline
이전에는 없던 "단일 패스 실시간 통합"이므로 코드 레벨 Baseline은 없다. 대신 두 개의 독립적으로
이미 검증된 기존 수치를 비교 기준으로 삼았다:
- Metadata Pipeline만 있을 때의 Latency: EXP-008 `run_full_pipeline.py` 벤치마크
  (원본 crowded_intersection 1080p 영상, P95=41.8ms, 27.7 FPS).
- Circular Buffer 단독 인코딩 비용: EXP-021/022 (버퍼 push/pop 자체는 별도로만 측정, 실제
  Detection 파이프라인과 동시에 측정한 적은 없음).

## Result

### 1) 통합은 실제로 동작했다 — Track 10개, Event 8개(ENTER 4/EXIT 4), BestShot, Event Clip 전부 SQLite에 함께 기록
`results/EXP-023/summary.json`. ROI Intrusion이 실제 YOLO 검출로 2개 Track(track 1, 3)에서
각 2회씩 총 4회 ENTER를 만들었고, `MetadataStore.update_event_detail()`로 4개 ENTER Event
row 모두에 동일한 `event_clip_path`가 사후에 붙었다(`results/EXP-023/metadata.db`로 직접 조회
확인) — VMS Search가 "이 Event가 발생한 순간의 영상"을 서빙할 수 있는 최소 구조가 만들어졌다.

### 2) Circular Buffer 통합 오버헤드는 전체 파이프라인 대비 약 5.7~6.2%
`results/EXP-023/latency_breakdown.csv` (900프레임, 두 번 재실행해 재현성 확인):

| Stage | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| Detection+Tracking (YOLO11n+ByteTrack) | 54.6 | 72.4 | 81.1 |
| ROI/BestShot/Metadata (SQLite) | 9.6 | 14.8 | 17.8 |
| **Adaptive Recording+Circular Buffer(신규)** | **3.8** | **5.5** | **6.5** |
| 전체(Total) | 66.5 | 88.5 | 99.0 |

Circular Buffer(JPEG 인코딩+push, Tier 판정 포함)가 추가하는 비용은 P50 기준 3.8ms(전체의
5.7%), P95 기준 5.5ms(전체의 6.2%)로, 가설 1과 같이 한 자릿수 %대였다. Detection+Tracking이
P50 기준 전체의 82%를 차지해 여전히 압도적 병목이다. (이 세션의 Detection 자체 P50=54.6ms는
EXP-008이 측정한 원본 1080p 영상 기준(P95=41.8ms 전체)보다 프레임당 느리다 — 입력 영상 종류
[810×1080 정지사진 기반 vs 실제 1920×1080 동영상]과 하드웨어/세션이 달라 직접 비교는 불가능하며,
이번 실험의 핵심 비교는 "같은 실행 내에서 Circular Buffer 유무"이므로 이 차이는 결과 해석에
영향을 주지 않는다.)

### 3) 발견한 문제: Pre-Roll을 "언제" 꺼내야 하는가 — 대안 A vs 대안 B
두 번의 Intrusion(frame 639, 666)과 그다음 두 번(frame 789, 819)이 Post-Roll(250프레임)
간격보다 가깝게 반복되어, EVENT Tier가 frame 639부터 영상이 끝나는 frame 899까지 한 번도
끊기지 않고 이어졌다(261프레임 EVENT, 4건의 ENTER가 Clip 1개로 자동 병합 — EXP-017/PAR-008의
Merge 로직과 동일한 효과가 온라인에서도 재현됨). 이 EVENT 구간 길이(261프레임)가 Circular
Buffer 용량(271프레임)에 근접했다.

이 실험은 Pre-Roll을 꺼내는 시점으로 두 가지를 놓고 구현 전 비교했다:
- **대안 A: EVENT Tier 구간이 끝난 뒤(Post-Roll 만료 시점)에 한 번에 Pre-Roll+본편을 Circular
  Buffer에서 꺼낸다.** 구현이 더 단순하다(Clip 하나를 한 번의 호출로 완성). 그러나 EVENT
  Tier가 오래 지속되면(이번처럼 261프레임) 그 사이에도 매 프레임 Circular Buffer에 새 프레임이
  계속 push되어 오래된 프레임(Pre-Roll 시작 부분)이 먼저 evict된다 — 즉 Pre-Roll을 "나중에"
  꺼내려는 순간 이미 사라져 있을 위험이 있다.
- **대안 B(채택): Tier가 IDLE/NORMAL → EVENT로 전환되는 바로 그 프레임(트리거 순간)에 즉시
  Pre-Roll을 Circular Buffer에서 꺼내고, 그 이후 프레임은 실시간으로 들어오는 그대로 Clip에
  이어 붙인다.** 트리거 이후 같은 EVENT 구간 안에서 추가 ENTER가 발생해도(2번째, 4번째 Intrusion)
  이미 열려 있는 Clip에 Track ID만 추가로 기록해 병합한다.

실제 이 실험의 종료 시점 Circular Buffer 상태로 **대안 A를 사후 검증**했다: 만약 대안 A대로
EVENT 구간이 끝나는 시점(frame 899)에 Pre-Roll을 포함한 전체 Window `[389, 899]`(511프레임)를
그 시점의 Circular Buffer에서 꺼내려 했다면, `FrameCircularBuffer.coverage()`가
**240/511프레임(47.0%) 누락**을 보고했다(`start_frame=389`부터 `628`까지, 버퍼 용량 271이
262(899-628+1)를 조금 넘는 수준이라 오래된 절반이 이미 evict됨). 즉 대안 A는 이번처럼 EVENT
Tier가 길게 이어지는(실제 카지노에서도 사람이 몰리는 구간엔 흔한) 상황에서 Pre-Roll의 거의
절반을 조용히 잃어버렸을 것이다 — FC-008(Naive Stream Copy가 조용히 불완전한 Clip을 만든 것)과
본질적으로 같은 패턴의 문제를, 이번엔 구현 전 비교로 미리 걸러냈다.

## Failure Cases
- **[신규 발견, 재발 방지 조치까지 완료] Post-Roll이 영상/세션이 끝나기 전에 완료되지 못하면
  Event Clip이 잘린 채로 저장된다.** 이번 900프레임 시나리오는 EVENT Tier가 frame 639부터
  시작해 영상 끝(899)까지 한 번도 끊기지 않아, 4건의 ENTER가 병합된 Clip이 `end_frame=899`에서
  `truncated_at_video_end=true`로 강제 종료됐다(Post-Roll 250프레임이 다 채워지기 전에 영상이
  끝남). FC-008처럼 "조용히" 발생하지 않도록 `truncated_at_video_end` 플래그와
  `frames_missing` 카운트를 summary에 명시적으로 남기게 이미 구현했다 — 실제 배포 환경에서는
  카메라 스트림이 계속되므로 이 케이스는 "세션/데모가 인위적으로 짧을 때"만 발생하지만, Edge
  Camera 재시작/전원 차단 시점에 동일한 상황이 재현될 수 있어 재발 방지 대책에 반영한다.

## Analysis
- Circular Buffer 통합 비용(5.7~6.2%)은 예상대로 작았다 — 병목은 여전히 Detection+Tracking이다.
  Edge Camera에 이 전체 스택을 올린다면 Circular Buffer 최적화보다 Detection 자체의 경량화
  (EXP-009 Edge Optimization 참고)가 우선순위가 높다는 기존 결론이 이번 통합 후에도 유지된다.
- "온라인으로 Tier를 계산해야 한다"는 요구사항이 `OnlineTierClassifier`라는 새 추상화를
  만들게 했다 — 기존 `compute_tier_sequence()`(Two-Pass, 배열 전체를 미리 앎)와 동일한 규칙을
  스트리밍으로 재현해야 했고, 두 구현이 실제로 일치하는지 회귀 테스트(랜덤 시퀀스 20개 포함)로
  검증했다.
- Pre-Roll을 꺼내는 "시점"이 Correctness에 미치는 영향은 문서(adaptive.py의 기존 주석)에는
  없었고 이번에 처음 실측했다 — Circular Buffer의 용량 설계(`required_capacity_for_pre_roll`)는
  Pre-Roll 길이만 고려했지 "EVENT Tier가 얼마나 길게 지속될 수 있는가"는 고려 대상이 아니었다.
  이번 실험으로 두 값이 서로 다른 관심사라는 것이 명확해졌다(재발 방지 대책 참고).

## Decision
1. **Pre-Roll은 트리거 순간(Tier가 EVENT로 전환되는 프레임)에 즉시 꺼낸다(대안 B)** — 대안
   A는 실측으로 확인된 47% 프레임 유실 위험 때문에 채택하지 않는다.
2. EXP-022에서 후보로만 비교했던 `TierQualityConfig(idle=50, normal=75, event=95)`를
   이 실험부터 프로젝트 기본값으로 승격한다(`PRODUCTION_TIER_QUALITY`) — EXP-022가 이미
   EVENT 화질 보존과 메모리 절감의 균형점으로 검증했다.
3. `run_full_pipeline.py` 자체는 이번에도 수정하지 않았다 — 그 스크립트가 요구하는 원본
   `crowded_intersection_1080p.webm`은 이번 세션에서도 접근 불가능하기 때문이다. 대신 동일한
   모듈(BestShot/ROI/MetadataStore)을 재사용하는 새 스크립트(`run_exp023_pipeline_integration.py`)로
   통합 로직 자체를 검증했다 — 원본 영상에 접근 가능한 세션에서 `run_full_pipeline.py`에
   이 로직을 이식하는 것을 다음 Action으로 남긴다(가정이 아니라 명시적 Decision으로 기록).

## Next Action
- 원본 영상 접근이 가능한 세션에서 이 실험의 `OnlineTierClassifier`+`FrameCircularBuffer`+
  `TierQualityConfig` 통합 코드를 `run_full_pipeline.py`에 그대로 이식하고 동일 벤치마크를
  재현할 것.
- FastAPI VMS Search API(`src/api`)에 `event_clip_path`를 응답에 포함하는 필드를 추가해
  "특정 Event 전후 영상" 요구사항(지침 17)을 완전히 닫을 것.
- EVENT Tier 지속시간이 Circular Buffer 용량을 초과할 수 있다는 이번 발견을 반영해, 용량 계산
  함수에 "예상 최대 연속 EVENT 길이"를 인자로 받는 확장을 검토할 것(현재는 Pre-Roll 길이만
  고려).
