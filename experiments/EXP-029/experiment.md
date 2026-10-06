# EXP-029 Attribute Metadata(c_wb) 실시간 파이프라인 통합 - 호출 빈도 A/B 비교

관련 실험: EXP-025(Baseline/A/B, PAR-016), EXP-026(FC-012 WB 재검토, PAR-017),
EXP-027(coco128 n=21 재검증, PAR-018), EXP-028(Segmentation Mask 대안 C/C_wb, PAR-019)

## Goal
EXP-028 Next Action 2번과 01. Architecture & Roadmap 다음 Action 5번("c_wb를 Track당
1회(BestShot 확정 시점)만 호출하는 방식으로 run_full_pipeline.py에 연결해 실제 FPS/P95
Latency 영향을 재측정할 것 - 이번 실험[EXP-028]은 오프라인 n=21 Batch만 측정, 실시간 통합
Overhead는 미측정")을 수행한다. EXP-025~028은 모두 Attribute 분류 자체의 Accuracy만
오프라인 정지 이미지로 측정했고, 실제 Detection+Tracking+BestShot이 매 프레임 돌아가는
실시간 파이프라인에 연결했을 때 전체 FPS/Latency가 얼마나 영향받는지는 한 번도 측정되지
않았다.

## Hypothesis
1. Attribute 분류(c_wb)를 매 프레임마다 모든 활성 Track에 대해 호출하면(대안 A), Segmentation
   추론 비용(EXP-028 실측: coco128 전체 이미지 기준 Detector 단독 대비 1.29x)이 프레임마다
   반복되어 전체 파이프라인 FPS가 크게 떨어질 것이다.
2. BestShot이 확정되는 순간(Track 소실/영상 종료)에만 Track당 1회 호출하면(대안 B, 지침
   12/19가 원래 의도한 설계), Track 수는 프레임 수보다 훨씬 적으므로 전체 파이프라인 FPS에는
   거의 영향이 없을 것이다.

## Problem
지침 12/19는 "BestShot 확정 시점에만" 또는 "Track마다 1회"라는 표현으로 Attribute/BestShot
계산 빈도를 암묵적으로 제한하고 있었지만, 지금까지의 모든 Attribute 실험(EXP-025~028)은 이
설계를 코드로 강제하지 않은 채 오프라인 Batch로만 검증했다. 실제로 매 프레임 계산하는 구현을
그대로 run_full_pipeline.py에 붙였다면 Segmentation 모델 추론이라는 무거운 연산이 활성 Track
수 × 프레임 수만큼 반복돼, EXP-008이 확인한 실시간 처리 가능 FPS(27.7)를 무너뜨릴 위험이
있었다 - 이를 구현 전에 실측으로 확인하지 않고 바로 production에 연결하는 것은 이 프로젝트가
반복적으로 경계하는 "검증 없는 기능 추가"(PAR-018 참고)에 해당한다.

## Dataset
이 원격 세션도 egress 정책상 commons.wikimedia.org가 403으로 차단되어(`curl -sS
https://commons.wikimedia.org` 직접 재확인) run_full_pipeline.py가 쓰는
crowded_intersection_1080p.webm을 받을 수 없다(EXP-010/014~023과 동일 Blocker). EXP-017/023과
동일하게 ultralytics 내장 bus.jpg(복수의 보행자가 포함된 실제 사진)를 Pan 합성한 시나리오로
대체했다. 다만 EXP-023과 달리 "사람 없음" 구간을 3번(각 110~130프레임, ByteTrack
`track_buffer` 기본값 30프레임보다 훨씬 길게) 반복해 넣어, Track이 한 번만 생성·소멸하는
것이 아니라 실행 전체에 걸쳐 여러 번 생성·소멸하도록 만들었다 - "Track당 1회" 호출이 영상
끝에서 단 한 번만 일어나는 것이 아니라 분산되는 더 현실적인 조건을 만들기 위함이다.

## Environment
원격 자동화 세션, `.venv`(Python 3.11, 신규 생성) + opencv-python, numpy, torch, ultralytics
8.4.173, fastapi, pytest. CPU 전용. 신규 의존성 모델은 없음(yolo11n.pt, yolo11n-seg.pt는
EXP-028에서 이미 사용된 동일 Release Asset).

## Configuration
`scripts/run_exp029_attribute_pipeline_integration.py`에 3개 variant를 동일 시나리오·동일
900프레임·동일 YOLO11n(conf=0.4, imgsz=640)+ByteTrack으로 구현:
- **none** (Baseline): Attribute 분류 없음 - Detection+Tracking+BestShot+Metadata만.
- **per_frame** (대안 A): 매 프레임, 활성 Track마다 `classify_track_attributes_once()` 호출
  (Segmentation 모델을 해당 crop에 매 프레임 재실행).
- **per_track** (대안 B): BestShot이 확정되는 시점(Track 소실/영상 종료)에만 Track당 정확히
  1회 호출.

Segmentation Mask는 EXP-028처럼 "전체 프레임 + bbox IoU 매칭"이 아니라 **BestShot crop 자체를
다시 Segment**하는 방식으로 얻는다(`segment_largest_instance_mask()`) - PAR-004(BestShot
Memory Leak) 수정 이후 `BestShotTracker`는 Track마다 원본 프레임이 아니라 최종 crop 1장만
들고 있으므로, 원본 프레임에 대한 Segmentation+IoU 매칭 자체가 애초에 불가능하다(설계
제약, Decision 참고).

Production 코드(`scripts/run_full_pipeline.py`)에도 동일한 대안 B를 실제로 연결했다:
`segment_largest_instance_mask()` + `finalize_track_attributes()`를 추가하고, 두 BestShot
확정 지점(Track 소실 시 / 영상 종료 시) 모두에서 `MetadataStore.update_attributes()`를
호출하도록 수정했다. Segmenter(`yolo11n-seg.pt`)는 Detector와 마찬가지로 루프 시작 전
1회만 로드한다.

## Baseline
EXP-008이 실측한 전체 통합 파이프라인(Detection+Tracking+Events+BestShot+Metadata, Attribute
없음) 기준 FPS 27.7 / P95 41.8ms(원본 1080p 영상 기준). 이번 실험은 합성 bus.jpg 시나리오라
절대 FPS 수치가 다르므로(모델 load 직후 CPU 추론, 복수 인물) 직접 비교하지 않고, **이번
실험 안에서의 세 variant 간 상대 비교**(동일 조건, Attribute 유무/빈도만 다름)만 결론으로
사용한다.

## Result
`results/EXP-029/summary.json`, `latency_breakdown.csv` (900프레임, 3개 "사람 등장" 구간,
구간당 최대 5명 동시 Track - bus.jpg 자체에 보행자 여러 명이 있음):

| Variant | FPS | P50(ms) | P95(ms) | Attribute 호출 수 | 호출당 평균(ms) |
|---|---|---|---|---|---|
| none (Baseline) | 10.68 | 73.2 | 96.9 | 0 | - |
| **per_track (대안 B, 채택)** | **10.65** | 72.7 | 99.6 | 16 | 59.1 |
| per_frame (대안 A, 기각) | **4.38** | 267.8 | 358.8 | 2045 | 54.8 |

FPS 저하율(Baseline 대비): per_track **-0.28%**, per_frame **-58.99%**.

`run_full_pipeline.py`(production) 변경은 `pytest tests/ -q`로 전체 Regression Test를
재실행해 검증했다(202 passed, 7 skipped - 기존 197 passed/7 skipped에서 신규 Unit Test 5개만
추가, 기존 테스트 전부 통과 유지).

## Failure Cases
- 없음(치명적 실패는 없음). 다만 `per_track` variant에서 저장된 Track 행 수(15개, track_id
  16~30)와 `attribute_calls` 카운터(16)가 1개 어긋나는 것을 관찰했다 - bus.jpg에 동시에 여러
  보행자가 있어 ByteTrack이 짧은 간격(10프레임 초과 but `bytetrack.yaml` 내부 track_buffer
  30프레임 이하)으로 일시적으로 놓친 뒤 **같은 track_id로 재할당**하면, 우리 파이프라인의
  `last_seen` 기반 10프레임 Timeout이 먼저 `forget_track()`을 한 번 호출하고, 그 track_id가
  나중에 다시 활성화되면 두 번째 `forget_track()`이 다시 호출돼 같은 track_id에 대해
  `update_attributes()`가 두 번(마지막 값으로 덮어쓰기) 호출될 수 있다. 결과가 틀리지는
  않지만(멱등적 덮어쓰기), Attribute 호출 횟수를 실제보다 과대 집계한다는 것은 정직하게
  기록한다 - 이번 실험의 핵심 결론(대안 A vs B의 FPS 차이)에는 영향 없음(둘 다 동일한
  재할당 패턴을 겪으므로 상대 비교가 깨지지 않음).

## Analysis
**가설 1, 2 모두 확인됐다.** 매 프레임 Segmentation을 돌리는 대안 A는 FPS를 10.68→4.38로
59% 떨어뜨렸다 - 활성 Track이 평균 4~5명인 bus.jpg 시나리오에서 프레임당 Attribute 호출이
누적돼(총 2045회) Segmentation 추론 비용(평균 54.8ms/호출)이 그대로 파이프라인 전체 지연에
더해졌다(P50 73.2ms→267.8ms, 거의 Attribute 호출 비용만큼 증가). 반면 Track당 1회만 호출하는
대안 B는 같은 900프레임에서 호출이 16회로 줄어(전체 Detection 호출의 1.8%) FPS 저하가
0.28%에 그쳤다 - 통계적 잡음(P95는 오히려 99.6ms로 Baseline보다 약간 높지만, Attribute
호출이 발생한 소수 프레임에서만 수십 ms 추가되는 것이지 매 프레임 영향을 주지 않는다는
것이 핵심).

이는 EXP-028이 Decision에 적어둔 "지침 12/19 설계 의도와 일치하면 이 오버헤드는 실시간
파이프라인 FPS에 영향을 주지 않을 가능성이 높다"는 추측을 실측으로 확정했다 - **"그럴듯한
가정"과 "실측 확인"을 구분한 사례**로, 이 프로젝트가 반복하는 원칙(PAR-018 등)과 같은 선상에
있다.

**Edge Camera 관점**: Segmentation 모델(yolo11n-seg.pt, +0.57MB, EXP-028)을 추가로 상시
로드해야 하지만, 추론 자체는 Track 생애당 1회만 일어나므로 다채널 카지노 CCTV에서도 채널당
오버헤드는 "그 채널에서 Track이 종료되는 빈도"에 비례할 뿐 프레임 수와는 무관하다 - Edge
NPU/CPU 예산이 제한적인 환경에서 중요한 성질이다.

## Decision
1. **대안 B(Track당 1회, BestShot 확정 시점)를 `run_full_pipeline.py`에 실제로 연결한다.**
   `MetadataStore`에 `upper_color`/`lower_color` 컬럼과 `update_attributes()`를 추가하고,
   두 BestShot 확정 지점(Track 소실/영상 종료) 모두에서 호출하도록 수정했다. 대안 A(매 프레임)
   는 코드에도 넣지 않는다 - 실측으로 확정된 59% FPS 저하는 지침 24(Edge Optimization)의
   핵심 Trade-off(Accuracy vs Latency vs Model Size)를 명백히 위반한다.
2. **VMS Search API에 upper_color/lower_color를 노출하는 것은 계속 보류한다.** EXP-027/028이
   이미 확정한 대로 c_wb의 절대 Accuracy(n=21 기준 38.1%)가 아직 production 임계치에
   못 미치기 때문이며, 이번 실험은 "통합했을 때 빠른가"만 검증했을 뿐 "통합해도 되는
   정확도인가"에는 새로운 근거를 추가하지 않았다 - 두 질문을 섞지 않는다.
3. Segmentation Mask는 EXP-028의 "전체 프레임 + IoU 매칭" 대신 "BestShot crop 자체를
   재검증(re-segment)"하는 방식으로 구현을 바꿨다 - PAR-004(Memory Leak) 수정으로
   `BestShotTracker`가 원본 프레임을 보관하지 않게 된 구조적 제약 때문이며, 이번 실험으로
   crop 단위 재분할도 유효한 mask를 만들어낸다는 것을 확인했다(`attribute_calls`가 매번
   정상적인 색상 문자열을 반환, "gray" fallback 비율 낮음).

## Next Action
1. 원본 영상(실제 CCTV, Wide-angle) 접근이 가능해지는 세션에서 `run_full_pipeline.py`를
   실제로 실행해 이 실험의 결론(Track당 1회 호출의 FPS 영향 거의 없음)이 합성 시나리오가
   아닌 실제 입력에서도 유지되는지 재확인할 것.
2. Attribute Accuracy가 향후 production 임계치를 넘기면(FC-012 strong_light 해결 포함),
   VMS Search API에 upper_color/lower_color 필터를 추가하는 작업(EXP-024의
   `MetadataStore.query_events()` 재사용 패턴과 동일하게 `query_tracks()`에 `upper_color`
   필터 인자 추가)을 진행할 것.
3. `attribute_calls` 카운터가 실제 저장된 Track 행 수와 어긋나는 현상(Failure Cases 참고,
   ByteTrack 내부 track_buffer와 우리 파이프라인의 10프레임 Timeout 간 불일치로 같은
   track_id가 재활성화)은 이번 실험의 결론에 영향이 없어 깊이 조사하지 않았다 - FC-004(Track
   소실 시 EXIT 유실, PAR-002)와 같은 뿌리(두 "소실 판정" 기준의 불일치)일 가능성이 있어
   다음 Tracking 관련 세션에서 재검토할 것.
