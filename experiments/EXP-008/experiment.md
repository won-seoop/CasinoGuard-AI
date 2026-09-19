# EXP-008 Metadata Pipeline + VMS Search + End-to-End Performance Benchmark ✅ 실행 완료

## Goal
Phase 1~7에서 만든 Detection/Tracking/BestShot/Event(Intrusion/LineCrossing/Loitering) 모듈을 하나의 파이프라인으로 통합해 SQLite Metadata Store에 저장하고, FastAPI로 검색 가능하게 만든다. 통합 파이프라인 전체의 실시간성도 함께 측정한다.

## Dataset
crowded_intersection_1080p.webm, 800 frame

## Configuration
- DB: SQLite (08. Technical Decisions 참고 — 단일 프로세스 데모 규모에 서버형 PostgreSQL은 과함)
- ROI/Line/Loitering Threshold: EXP-005~007과 동일 값 재사용 (Loitering만 EXP-007 분석 결과를 반영해 데모용 8초로 조정 — 횡단보도에는 20~30초가 과하다는 결론을 실제로 적용)
- API: FastAPI, `/tracks`, `/tracks/{id}`, `/events`, `/` (간단 Dashboard)

## Result

**Metadata Pipeline**
- Tracks 저장: 164개
- BestShot 저장: 115개 (관측치 5개 미만인 Track은 제외)
- Events 저장: 190개 (INTRUSION_ENTER/EXIT, LINE_CROSSING, LOITERING 통합)

**End-to-End Performance Benchmark** (Detection+Tracking+ROI/Line/Loitering 로직+DB 쓰기+BestShot 후보 수집 전체 포함, BestShot 최종 계산/저장은 제외한 순수 프레임 루프 기준)

| Metric | 값 |
|---|---|
| 처리 Frame 수 | 800 |
| 전체 소요 시간 | 28.8초 |
| FPS | 27.7 |
| P50 Latency | 32.2ms |
| P95 Latency | 41.8ms |
| P99 Latency | 65.4ms |

원본 영상 FPS(23.98)보다 파이프라인 처리 FPS(27.7)가 더 높아 **이 통합 파이프라인은 M3에서 실시간 처리가 가능함**을 확인했다.

**VMS Search API 검증** (FastAPI TestClient로 실제 DB 대상 통합 테스트 7개 작성·통과)
- `/tracks?min_dwell_sec=100` — 특정 시간 이상 체류한 Track 필터링
- `/tracks/{id}` — 특정 Track의 이동 기록(trajectory) + 전체 Event 이력
- `/events?event_type=LOITERING` — 특정 Event Type만 조회
- `/` — Dwell Time 상위 5개, 최근 Event 10개를 보여주는 최소 Dashboard

## Failure Cases
- 초기 구현에서 `add_event()`가 `upsert_track()`보다 먼저 호출되어 SQLite Foreign Key 제약(`FOREIGN KEY constraint failed`) 오류 발생. Track row가 없는 상태에서 Event를 먼저 쓰려고 한 것이 원인 — 순서를 "Track upsert → Event 기록"으로 바꿔서 해결. 사소하지만 "Track과 Event 저장 순서"라는 통합 단계에서만 드러나는 전형적인 버그였다.

## Analysis
개별 Phase(1~7)에서는 각 모듈이 별도로 동작해 병목이 뚜렷하지 않았지만, 통합 후에도 P95 41.8ms(≈24 FPS 상당)로 실시간성을 유지한다는 것을 확인했다. Detection(EXP-002)이 여전히 전체 지연의 대부분을 차지하고, ROI/Line/Loitering 로직과 DB 쓰기의 오버헤드는 상대적으로 작다는 것을 간접적으로 확인했다(EXP-002의 conf=0.4/imgsz=640 FPS 27.4와 이번 통합 파이프라인 FPS 27.7이 거의 같음).

## Decision
Core MVP(Phase 1~7 + Metadata Pipeline + VMS Search + Performance Benchmark)를 이 상태로 완성 처리한다. Edge Optimization(ONNX/INT8), Multi Thread Pipeline 등은 Stretch Goal로 남긴다.

## Next Action
Core MVP 완료 — README 정리 및 Stretch Goal(Person Re-ID 등) 착수 여부는 시간 남는 대로 결정
