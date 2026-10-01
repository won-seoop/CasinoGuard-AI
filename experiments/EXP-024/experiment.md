# EXP-024 VMS Search API event_clip_path 응답 필드 + Event Clip 재생 엔드포인트

관련 실험: EXP-008(Metadata Pipeline/VMS Search API), EXP-023(Adaptive Recording+Circular Buffer
실시간 통합, PAR-014, event_clip_path를 Event detail에 사후 부착)

## Goal
01. Architecture & Roadmap에 EXP-023부터 명시적으로 남아 있던 다음 Action —
"FastAPI VMS Search API(src/api)에 event_clip_path를 응답에 포함하는 필드를 추가해 '특정 Event
전후 영상' 요구사항(지침 17)을 완전히 닫을 것" — 을 실제로 구현하고 검증한다.

## Hypothesis
`MetadataStore.update_event_detail()`(EXP-023/PAR-014)이 Event의 `detail` JSON에
`event_clip_path`를 이미 붙이고 있으므로, VMS Search API도 당연히 그 경로를 응답에서 쓸 수 있을
것이라고 가정했다. 실제로 `src/api/main.py`를 읽어보기 전까지는 이 가정을 검증하지 않았다.

## Problem
`src/api/main.py`의 `/events`와 `/tracks/{track_id}`는 `MetadataStore`를 전혀 쓰지 않고
`sqlite3.connect()`로 직접 `SELECT *`를 실행한 뒤 `dict(row)`로만 반환한다. `events.detail`
컬럼은 TEXT(JSON 문자열)로 저장되는데, 이 raw SQL 경로는 이를 `json.loads()`하지 않고 그대로
반환한다 — 즉 `event_clip_path`가 **문자열로 인코딩된 JSON 안에 숨어 있어** 클라이언트가 직접
파싱해야만 꺼낼 수 있었다. 더 심각하게는, 그 경로를 안다 해도 **실제 영상 파일을 내려주는
엔드포인트가 아예 존재하지 않았다** — 지침 17의 "특정 Event 전후 영상" 요구사항(결과 표시:
BestShot, Timestamp, Track ID, Event Type, Camera ID, **Video 위치**)을 API 레벨에서 전혀
충족하지 못하는 상태였다. (`tests/test_metadata_store.py`가 Store 레벨에서는 `detail`에
`event_clip_path`가 올바르게 들어간다는 것만 검증했고, API 레벨 테스트인 `tests/test_api.py`는
이 필드를 전혀 건드리지 않았다 — 즉 "Store는 맞는데 API가 노출 안 함"이라는 계층 간 간극을
기존 테스트가 잡아내지 못했다.)

## Dataset
- `data/raw/bus.jpg` (ultralytics 패키지 내장 실제 사진). 이 원격 세션도 Wikimedia 등 외부
  도메인이 egress 정책상 차단되어(`curl commons.wikimedia.org` → 403, EXP-010부터 반복
  확인된 동일 제약) `crowded_intersection_1080p.webm`을 받을 수 없었다. `yolo11n.pt`는
  GitHub Release Asset 경로로 정상 다운로드됨을 재확인했다.
- `scripts/run_exp023_pipeline_integration.py`를 그대로 재실행해(코드 수정 없이) 실제
  YOLO11n+ByteTrack 검출 결과가 반영된 `results/EXP-023/metadata.db`와
  `results/EXP-023/event_clips/event_clip_1.mp4`(27,033,338 bytes)를 재생성했다 — API가
  "실제로 생성된 Event Clip"을 정상적으로 서빙하는지 end-to-end로 눈으로 확인하기 위함이다.
- API 레벨 pytest(아래 Result)는 이 무거운 YOLO 재실행에 의존하지 않도록 `MetadataStore`로
  직접 만든 작은 fixture DB를 사용한다 — section 32 원칙("AI 모델 자체보다 직접 작성한 로직을
  테스트하는 것을 우선") 그대로.

## Environment
- 원격 자동화 세션, `.venv`(Python 3.11.15) 신규 생성
  (opencv-python-headless, numpy, fastapi, "uvicorn[standard]", pytest, lap, ultralytics).
- CPU 전용.

## Configuration
변경 없음(기존 EXP-023 Configuration 그대로 재사용 — pre_roll/post_roll 10초, buffer_capacity
271, tier_quality idle=50/normal=75/event=95).

## Baseline
API 수정 전 동작(수정 전 커밋으로 직접 재현):
```
GET /events
{"event_id": 1, "event_type": "INTRUSION_ENTER", "frame_idx": 639,
 "detail": "{\"zone\": \"restricted_zone\", \"event_clip_path\": \"results/EXP-023/event_clips/event_clip_1.mp4\"}"}
```
`detail`이 파싱되지 않은 문자열로 반환되고, `event_clip_path`를 꺼내 쓸 수 있는 전용
엔드포인트가 없다 (`GET /events/{id}/clip` 자체가 존재하지 않아 404 Not Found — FastAPI
라우트 부재).

## Result

### 1) 구현
- `src/metadata/store.py`: `MetadataStore.query_events()`에 `has_clip: bool | None` 파라미터를
  추가하고, 모든 반환 row에 `event_clip_path`를 `detail`에서 꺼내 최상위 필드로도 노출하도록
  수정.
- `src/api/main.py`:
  - `/events`, `/tracks/{track_id}`가 raw SQL 대신 `MetadataStore.query_events()`를 재사용하도록
    변경(이미 테스트된 JSON 파싱 로직을 그대로 재사용, 중복 제거).
  - `/events`에 `has_clip` Query 파라미터 추가(이벤트 클립이 있는/없는 Event만 필터링).
  - 신규 `GET /events/{event_id}/clip` 엔드포인트 추가: `event_clip_path`가 없거나 파일이
    디스크에 없으면 404, 있으면 `FileResponse(media_type="video/mp4")`로 실제 영상 바이트를
    서빙 — 브라우저에서 바로 재생 가능.
  - Dashboard(`GET /`)의 최근 Event 10개 표에 "▶ clip" 링크 컬럼 추가(클립이 있는 Event만).

### 2) 실제 재생성한 EXP-023 데이터로 End-to-End 수동 검증 (TestClient, DB_PATH만 교체)
`results/EXP-023/metadata.db`(8 Event: ENTER 4 / EXIT 4, 그중 ENTER 4건에 동일한
`event_clip_path`가 EXP-023의 Pre-Roll 트리거 즉시 추출 로직으로 붙어 있음)를 가리키도록 API의
`DB_PATH`만 바꿔 실제로 호출했다:

| 호출 | 결과 |
|---|---|
| `GET /events` | count=8, ENTER 4건 모두 `event_clip_path="results/EXP-023/event_clips/event_clip_1.mp4"`, EXIT 4건 모두 `null` |
| `GET /events?has_clip=true` | count=4 (ENTER 4건만) |
| `GET /events?has_clip=false` | count=4 (EXIT 4건만) |
| `GET /events/1/clip` | 200, `content-type: video/mp4`, 27,033,338 bytes = 실제 파일 크기와 정확히 일치 |
| `GET /events/3/clip` (EXIT, 클립 없음) | 404 `{"detail": "no clip recorded for this event"}` |
| `GET /events/9999/clip` (존재하지 않는 event_id) | 404 `{"detail": "event not found"}` |
| `GET /tracks/1` | `events` 배열의 ENTER row에도 `event_clip_path`가 동일하게 포함됨 |
| `GET /` (Dashboard) | 최근 Event 표에 "▶ clip" 링크 렌더링 확인 |

### 3) pytest (`tests/test_api_event_clip.py`, 신규 8개 + `tests/test_metadata_store.py` 신규 3개)
가벼운 fixture DB로 다음을 검증(YOLO 재실행 불필요, 0.6초 내 전체 통과):
- `event_clip_path`가 응답에서 최상위 필드로 노출되고, 클립이 없는 Event는 `null`
- `has_clip=true/false` 필터가 정확히 분리
- `/tracks/{id}`에 임베드된 Event에도 `event_clip_path`가 포함
- `/events/{id}/clip`이 실제 파일 바이트를 그대로 서빙(`content-type`, byte-for-byte 일치)
- 클립 없는 Event / 존재하지 않는 event_id / DB에는 경로가 있지만 파일이 실제로 없는 경우 모두
  404
- Dashboard에 clip 링크가 조건부로 렌더링됨

**Regression**: `python3 -m pytest tests/ -q` → 153 passed, 7 skipped(기존에도 skip되던
`results/metadata_pipeline/metadata.db` 의존 테스트 7개 — 이 세션도 원본 영상을 받을 수 없어
`run_full_pipeline.py`를 돌릴 수 없는 동일한 제약, 이번 변경으로 새로 생긴 skip이 아님).
기존 90여 개 Unit Test 전부 그대로 통과(회귀 없음).

## Failure Cases
- **[신규 발견, FC-011로 등록] `event_clip_path`가 Event의 `detail` JSON 안에만 있고 API가
  파싱하지 않아 VMS가 사실상 쓸 수 없었고, Event Clip을 실제로 재생/다운로드하는 엔드포인트
  자체가 없었다.** Store 레벨 테스트(`test_metadata_store.py`)는 `detail` dict 안에
  `event_clip_path`가 들어가는 것만 확인했고, API 레벨 테스트는 이 필드를 전혀 다루지 않아
  "Store는 맞는데 API가 그 값을 밖으로 못 꺼낸다"는 계층 간 간극이 EXP-023 이후 한 세션 동안
  감지되지 않았다. 이번 EXP-024로 수정 완료.

## Analysis
- 근본 원인은 API가 `MetadataStore`라는 이미 테스트된 추상화를 쓰지 않고 자체 raw SQL을
  따로 유지한 것이었다. `detail` JSON 파싱처럼 "한 곳에서만 올바르게 구현되어 있고 다른 곳에서는
  재구현(그리고 누락)되는" 패턴은 Metadata 스키마가 확장될 때(지침 19 Attribute Metadata의
  `upper_color`/`lower_color`/`bag`/`hat` 등도 같은 `detail`류 JSON 패턴을 쓸 가능성이 높다)
  반복될 수 있는 위험이다.
- Pre-Roll 추출 시점(EXP-023/PAR-014)과 Clip 재생(EXP-024)은 서로 다른 계층의 문제였지만,
  둘 다 "Circular Buffer/Clip 파일이 실제로 완전하게 만들어지는가"와 "그 결과를 외부가 실제로
  쓸 수 있는가"라는 이어지는 질문이었다 — EXP-023이 전자를, EXP-024가 후자를 닫았다.

## Decision
1. **API가 새 Metadata 필드를 노출할 때는 반드시 `MetadataStore`의 쿼리 메서드를 거친다
   (대안 B, 채택)** — raw SQL에서 그때그때 JSON을 파싱하는 방식(대안 A)은 이번처럼 특정
   필드가 조용히 누락되는 문제를 반복시킬 위험이 크고, Store 레벨에는 이미 같은 로직을
   테스트하는 pytest가 있어 재사용이 검증 비용도 더 낮다. API가 필요로 하는 컬럼이
   `query_tracks()`/`query_events()`가 반환하는 것과 다르면(`/tracks`가 필요로 하는
   `bestshot_score` 등), Store 쪽 쿼리 메서드를 확장하는 쪽으로 해결하고 API에 별도 SQL을
   남기지 않는다.
2. **Event Clip은 경로 문자열만 반환하지 않고 실제로 파일을 서빙한다(`FileResponse`)** —
   지침 17이 요구하는 "Event 전후 영상"은 VMS가 그 영상을 재생할 수 있어야 한다는 뜻이므로,
   경로만 알려주고 파일 접근은 클라이언트가 알아서 하게 하는 방식은 요구사항을 절반만
   충족한다고 판단했다.

## Next Action
- 원본 영상 접근이 가능한 세션에서 `run_full_pipeline.py` 자체에 EXP-023의
  `OnlineTierClassifier`+`FrameCircularBuffer` 통합을 이식하면, `results/metadata_pipeline/metadata.db`에도
  실제 `event_clip_path`가 채워지고 `tests/test_api.py`의 기존 skip도 자연히 해소될 것이다.
  (이번 EXP-024의 API 변경 자체는 그 이식과 독립적으로 이미 완결됐다.)
- 지침 19 Attribute Metadata(`upper_color`/`lower_color`/`bag`/`hat`)를 구현할 때, 이번
  Decision 1을 그대로 적용해 API가 Store 쿼리 메서드를 거치도록 설계할 것(재발 방지).
