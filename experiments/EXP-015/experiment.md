# EXP-015 Dwell Time(누적 체류 시간) 0-버그 재현 및 DwellCounter 수정

관련 Failure Case: FC-006 (Track 체류시간 dwell_frames가 전부 0으로 저장됨)
관련 PAR: PAR-006

## Goal

`run_full_pipeline.py`가 저장하는 `tracks.dwell_frames`가 항상 0이라는 FC-006을
이 저장소의 실제 코드에 대해 재현하고, 원인을 정확히 특정한 뒤 VMS Search가
요구하는 "ROI 안에 총 얼마나 있었는가"(재방문 포함 누적)를 정확히 계산하도록
수정한다.

## Hypothesis

FC-006은 단순히 "두 줄의 호출 순서가 바뀐 버그"가 아니라, `LoiteringDetector`의
`enter_frame`이 "지금 이 순간까지의 **연속** 체류"만 표현하도록 설계되어 있어서
ROI를 벗어나는 순간(`update()`가 `not inside` 분기에서 즉시 `enter_frame`을 pop)
이전 체류 기록 자체가 사라진다는 **구조적 문제**일 것이다. 따라서 호출 순서만
바꾸는 정도(Alternative A)로는 "ROI에 머물다 사라지기 전에 이미 ROI를 벗어난"
일반적인 케이스를 고치지 못하고, "ROI 안에 있는 상태로 그대로 사라지는"(Occlusion)
좁은 케이스만 고칠 것이라 예상한다.

## Problem (코드 근거)

`scripts/run_full_pipeline.py`(수정 전) 103\~112번 줄:

```python
for tid in list(last_seen.keys()):
    if idx - last_seen[tid] > 10:
        ...
        loiter_det.forget_track(tid)                       # (1) enter_frame을 먼저 pop
        dwell = idx - loiter_det.enter_frame.get(tid, idx)  # (2) 이미 없으므로 기본값(idx) 사용 -> dwell=0
        store.set_dwell(tid, dwell)
```

`src/events/loitering.py`의 `update()`는 ROI 밖으로 나가는 즉시 다음을 실행한다:

```python
if not inside:
    self.enter_frame.pop(track_id, None)
    self.already_fired.pop(track_id, None)
    return None
```

즉 Track이 ROI를 잠깐이라도 벗어나면(카지노 CCTV에서는 사람이 구역을 드나드는
것이 정상적인 행동) `enter_frame`이 그 즉시 사라지고, 이후 10프레임 이상
미관측되어 `forget_track()`이 호출되는 시점에는 이미 `enter_frame`이 없다.
호출 순서를 바로잡아도(Alternative A) 이 케이스는 여전히 고쳐지지 않는다.

## Dataset / Environment

- Part A: 통제된 시나리오(코드로 직접 구동, Ground Truth 확정) — 외부 데이터 불필요
- Part B: ultralytics 패키지 내장 실제 사진 `bus.jpg`(사람 4명, YOLO 공식 샘플)에
  Pan/밝기 Jitter를 적용한 300프레임 합성 영상 (EXP-010과 동일한 대체 입력)
- YOLO11n + ByteTrack(`bytetrack.yaml`), conf=0.4, imgsz=640
- Python 3.11, opencv-python-headless, ultralytics 8.4.158 (이번 세션에 새로 설치)

**Decision (네트워크 제약)**: 이번 원격 자동화 세션도 EXP-010/EXP-014와 동일하게
egress 정책상 Wikimedia Commons 등 기존 원본 영상 도메인에 접근할 수 없었다
(commons.wikimedia.org 403). PyPI는 접근 가능해 `opencv-python-headless`,
`ultralytics` 등은 새로 설치했다. 실제 카메라 영상 대신 bus.jpg+Pan/Jitter를
사용하는 것은 EXP-010의 선례를 그대로 따른 것이며, 이번 실험의 목적(Dwell 집계
로직의 정확성 검증)은 Detection 정확도와 무관하므로 대체 입력으로 충분하다.

## Configuration

- ROI(Part A): `[(0,0),(100,0),(100,100),(0,100)]`, 안/밖 점: `(50,50)` / `(500,500)`
- ROI(Part B): 프레임 왼쪽 60% 영역(`[(0,0),(0.6w,0),(0.6w,h),(0,h)]`) — bus.jpg 속
  인물들이 Pan에 따라 이 경계를 넘나들도록 설계
- forget_track 임계값: 10프레임 초과 미관측 (run_full_pipeline.py와 동일)
- 시나리오 3종(Part A):
  1. `occluded_while_inside`: 40프레임 연속 ROI 내부 → Occlusion(15프레임 미관측)
  2. `left_roi_before_vanishing`: 30프레임 내부 → 20프레임 ROI 밖 배회 → Occlusion(15프레임)
  3. `multi_visit_reentry`: 내부 20 → 밖 15 → Occlusion 13(재소실, forget 1회) → 내부 20(재입장) → 밖 5 → Occlusion 15(2차 forget)

## Baseline

기존 `run_full_pipeline.py` 코드(위 Problem 섹션의 순서)를 "Buggy"로 그대로
재현한다.

## Result

### Part A — 통제된 시나리오 (Ground Truth 대비)

| Scenario | Ground Truth | Buggy(기존) | Alt A(순서만 교정) | Alt B(DwellCounter) |
|---|---|---|---|---|
| occluded_while_inside | 40 | 0 | 50 (거의 근접, 대신 Occlusion 구간까지 포함해 과대) | **40 (정확)** |
| left_roi_before_vanishing | 30 | 0 | 0 (여전히 실패) | **30 (정확)** |
| multi_visit_reentry | 40(=20+20) | 0 | 0 (여전히 실패, 마지막 방문만 남고 1차 방문 20프레임 유실) | **40 (정확)** |

원본 CSV: `results/EXP-015/part_a_scenarios.csv`

### Part B — 실제 YOLO11n+ByteTrack Detection (bus.jpg Pan/Jitter, 300프레임)

| Metric | Buggy(기존) | DwellCounter(수정) |
|---|---|---|
| 총 Track 수 | 6 | 6 |
| dwell_frames > 0인 Track 수 | **0** | **5** |
| 비율 | 0.0% | 83.3% |

Track별 상세: `results/EXP-015/part_b_per_track.csv` (Track 1은 ROI 진입 없이
전체 구간 ROI 밖에 있어 fixed 값도 0 — 정상 동작).

## Failure Cases

- Alternative A(호출 순서만 교정)는 "occluded_while_inside" 시나리오만 부분적으로
  개선하고(그나마도 Occlusion 구간까지 포함해 정확한 값은 아님), 사람이 ROI를
  정상적으로 드나드는 대부분의 실제 상황(`left_roi_before_vanishing`,
  `multi_visit_reentry`)에서는 여전히 dwell=0을 반환한다. 즉 이번 세션에서 처음
  검토한 "간단한 순서 버그"라는 가설은 기각되었고, 실제로는 상태 책임이 잘못
  섞여 있는 구조적 문제였다.

## Analysis

`LoiteringDetector.enter_frame`은 "Loitering 이벤트를 연속 체류 기준으로 발생시킨다"는
Phase 7의 목적에는 정확히 맞는 설계다. 문제는 `run_full_pipeline.py`가 이 상태를
그대로 재사용해 "VMS Search용 누적 체류 시간"까지 계산하려 한 것이다. 서로 다른
두 책임(① 연속 체류 임계값 판정, ② 누적 총 체류 시간 집계)을 하나의 상태
변수로 처리하면 어느 한쪽의 요구사항(여기서는 ②)이 반드시 깨진다.

## Decision

**대안 A (호출 순서 교정)**: `dwell = idx - loiter_det.enter_frame.get(tid, idx)` 줄을
`forget_track()` 호출보다 먼저 옮기는 1줄짜리 수정. 구현 비용이 가장 낮다는
장점이 있었지만, Part A 실측에서 실제 케이스의 2/3(정상적으로 구역을 드나드는
경우, 재방문하는 경우)에서 여전히 0 또는 과소 계상되는 것이 확인되어 채택하지
않았다. "구역에 오래 머문 사람을 찾는다"는 원래 요구사항을 충족하지 못한다.

**대안 B (DwellCounter 분리, 채택)**: `src/events/dwell.py`에 `DwellCounter`를
새로 만들어 "ROI 안에서 관측된 프레임 수를 무조건 누적"하는 책임만 갖게 하고,
`LoiteringDetector`의 연속-스트릭 상태와는 완전히 분리했다. ROI 이탈/재방문에도
리셋되지 않고, Track이 사라져도(forget) pop하지 않아 나중에 같은 ID로 재관측돼도
이전 누적치가 덮어써지지 않는다. Part A/B 모두에서 Ground Truth와 정확히 일치했다.

메모리 측면에서 PAR-004(BestShot 후보 무제한 누적 Memory Leak)와 겉보기엔 비슷한
패턴(계속 쌓이기만 하는 dict)처럼 보일 수 있어 별도로 검토했다. PAR-004의 문제는
Track 하나당 프레임마다 crop **이미지**(수십\~수백 KB)를 리스트에 추가해 처리
시간에 비례해 무한히 증가한 반면, `DwellCounter.counts`는 Track ID 당 `int` 값
하나(수십 바이트)만 유지하고 크기는 "영상에 실제로 등장한 고유 인물 수"에
비례한다. 이는 이미 영구 저장되는 `metadata.db`의 `tracks` 테이블 행 수와 같은
차수이므로 별도의 leak으로 보지 않았다. 다만 초장시간(수개월) 무중단 운영 시
누적 고유 인물 수가 매우 커지면 재검토가 필요하다는 점은 재발 방지 항목에
남긴다.

## Verification

- `tests/test_dwell_counter.py` 5개 신규 Unit Test (연속 체류/Occlusion, ROI 이탈 후
  소실, 재방문 누적, 미관측 Track 기본값 0, Track 간 독립성)
- 기존 `tests/` 전체 58 passed / 7 skipped(사전 skip, 무관) — Regression 없음 확인
- `scripts/run_exp015_dwell_time_fix.py` Part A/B로 실측 재현 및 개선 확인
- `scripts/run_full_pipeline.py`에 `DwellCounter` 실제 반영 (buggy 1-line 계산 제거)

## Next Action

- FC-006 Notion 기록을 "재등장 Track dwell 덮어쓰기 미수정" → "DwellCounter로 수정
  완료"로 갱신
- `run_full_pipeline.py`를 실제 원본 영상(`crowded_intersection_1080p.webm`)으로
  재실행해 실측 dwell 분포를 확인하는 것은 Wikimedia 접근이 가능한 세션으로 이월
  (이번 세션 네트워크 제약, 상단 Decision 참고)
- Heatmap/Crowd Analysis 또는 Person Re-ID 등 나머지 Stretch Goal은 다음 세션에서 계속
