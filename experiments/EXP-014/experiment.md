# EXP-014 Line Crossing 왕복 중복 이벤트(FC-007) — band_px Hysteresis A/B ✅ 실행 완료

## Goal
FC-007(가상선 근처 박스 흔들림으로 Line Crossing 왕복 중복 이벤트 발생)을 실제로 재현하고,
검토된 두 대안(A: Track별 최소 재발생 간격 Cooldown, B: 선 양쪽 불감대 Hysteresis)을
동일 궤적으로 정량 비교해 `LineCrossingDetector`에 적용할 방식을 결정한다.
(01. Architecture & Roadmap "다음: FC-007 해결 A/B 비교", EXP-012 Next Action
"같은 band를 LineCrossingDetector에도 적용할지 A/B"에서 이어지는 작업.)

## Hypothesis
선까지의 부호 있는 거리 `|d| >= band_px`일 때만 확정 side를 바꾸는 Hysteresis(B)가,
이벤트 발생 후 일정 프레임 동안 무시하는 시간 기반 Cooldown(A)보다 "왜 억제되는지"가
기하학적으로 명확하고 프레임레이트 가정에 의존하지 않으므로 더 적합할 것이다.

## Problem
`src/events/line_crossing.py`의 `side_of_line()`은 외적의 부호만으로 선의 양쪽을 구분하고
(dead zone이 부동소수점 오차 수준인 1e-6뿐), `LineCrossingDetector.update()`는 side가
바뀔 때마다 무조건 이벤트를 낸다. 사람이 선 근처에서 멈추거나 Detection bbox가 프레임마다
몇 픽셀 흔들리면 side가 반복적으로 뒤집혀 같은 통과에 대해 A_TO_B/B_TO_A가 번갈아 발생한다.

## Dataset
이 원격 세션은 egress 정책상 Wikimedia 등 외부 도메인이 여전히 403으로 차단되어
(EXP-010/PAR-004와 동일 제약, curl로 재확인함) 기존 crowded_intersection_1080p.webm을
다시 받을 수 없었다. EXP-010과 동일하게 ultralytics 내장 실제 사진(bus.jpg, 사람 4명)에
Pan Jitter(동일 파라미터: `max_shift=12px`, `sin(idx*0.037)`/`cos(idx*0.053)`)를 주어
"카메라 고정, 사람 bbox가 프레임마다 흔들리는" 상황을 재현했다. Detection 자체는
900프레임 전부 실제 YOLO11n 추론 결과이므로, "실제 검출 bbox의 프레임 간 흔들림이 선
통과 판정에 미치는 영향"이라는 이 실험의 목적에는 이 대체 입력으로 충분하다(정확도
벤치마크가 아니라 이벤트 로직 안정성 검증이 목적).

## Environment
원격 컨테이너 Linux, Python 3.11(venv), PyTorch/Ultralytics(latest, pip 설치),
YOLO11n pretrained, ByteTrack(`bytetrack.yaml`), conf=0.4, imgsz=640.

## Configuration
- 900프레임(1차 패스로 YOLO/ByteTrack 추론 후 bbox를 캐싱, 2차 패스에서 같은 bbox 시퀀스로
  6개 detector 변형을 동시에 재생 — 동일 궤적에서 로직만 비교하기 위함)
- 가상선: 관측된 bottom-center y좌표의 중앙값(y=874.1, std=17.97)을 지나는 수평선
- `band_px in {0, 3, 5, 10, 20}` — B(Hysteresis)의 폭 sweep
- `cooldown_altA`: A(Cooldown) — band_px=0 판정 위에 "이벤트 후 15프레임(0.6초, FC-007
  Notion 관찰 기준과 동일 window) 동안 같은 Track 이벤트 무시"를 얹은 wrapper
- 중복 판정 기준: 같은 Track에서 direction이 반대인 이벤트가 0.6초(15프레임, 25fps 가정)
  이내에 연속 발생하면 "중복 반전 쌍" (Roadmap에 기록된 FC-007 관찰 기준과 동일 정의)

## Baseline
`band_px=0` (기존 코드와 동일한 동작 — side가 바뀌면 무조건 이벤트).

## Result

| Variant | 총 이벤트 | 중복 반전 쌍(0.6초 이내) |
|---|---|---|
| band_0 (Baseline, 기존 동작) | 57 | **24** |
| band_3 | 32 | 0 |
| **band_5 (채택)** | **31** | **0** |
| band_10 | 12 | 0 |
| band_20 | 0 | 0 |
| cooldown_altA (검토했지만 미채택) | 33 | 0 |

원본 데이터: `results/EXP-014/summary.json`, `results/EXP-014/summary.csv`

## Failure Cases
- **FC-007이 이 세션의 대체 입력으로도 실제로 재현됨**: band_0에서 57건 중 24쌍이 0.6초
  이내 방향 반전 — Roadmap에 기록된 실내 장면 관찰(76건 중 26쌍)과 같은 성격의 문제가
  다른 데이터에서도 나타남을 확인. 근본 원인이 특정 영상의 특이값이 아니라 로직 자체의
  구조적 문제임을 뒷받침한다.
- band_20은 총 이벤트가 0건까지 떨어짐 — 이 장면의 사람 위치 흔들림 폭(std≈18px)보다
  band가 크면 어떤 Track도 "확정된 반대쪽"에 도달하지 못해 진짜 통과까지 전부 놓친다.
  즉 band_px는 무조건 클수록 좋은 것이 아니라 Trade-off가 있음을 실측으로 확인.

## Analysis
- band_3부터 이미 중복 쌍이 24→0으로 완전히 사라지고, band_5까지는 총 이벤트 수(32→31)가
  거의 그대로 유지된다 — 이 장면에서는 3~5px 폭이면 흔들림은 억제하면서 진짜 이동은
  거의 그대로 통과시킨다는 뜻으로 해석된다. band_10부터 총 이벤트가 57의 약 1/5(12건)로
  급감해 진짜 통과까지 놓치기 시작하는 것으로 보인다(단, 이 장면 전용 관찰이며 GT는 없음).
- Cooldown(A)도 33건/중복 0으로 band_5와 비슷한 수준의 억제 효과를 보였다. 즉 이번
  데이터에서는 두 방식 모두 "작동은" 한다 — 그러나 설계상 차이가 있다(아래 Decision).
- **한계(정직하게 명시)**: 정답 라벨이 없어 band_5가 제거한 26건(57→31)이 전부 중복
  오탐인지, 그중 일부가 진짜 통과인지는 확정할 수 없다(EXP-012 People Counting에서도
  동일하게 명시된 한계). band_px=5는 이 장면(1080px 프레임, 사람 크기 기준)에 대한 값이며
  해상도/카메라 거리가 다르면 재조정이 필요하다(Notion FC-007 "미확인"에 이미 기록된 우려).

## Decision
**B(선 양쪽 불감대 Hysteresis, band_px)를 채택**하고 `LineCrossingDetector`에
`band_px: float = 0.0` 필드로 추가했다(기본값 0 유지 → 기존 12개 테스트 전부 무수정 통과,
하위 호환). `scripts/run_full_pipeline.py`(실제 통합 파이프라인)에는 `band_px=5`를 적용.

**A(시간 기반 Cooldown)를 검토했지만 채택하지 않은 이유**:
1. 정량적으로 이번 데이터에서 B와 거의 동등한 효과(33건 vs 31건, 중복 둘 다 0)였지만,
   Cooldown은 "몇 프레임" 값이 FPS에 의존적이다 — 카메라/코덱마다 FPS가 다른 실제
   CCTV 환경에서는 프레임 수가 아니라 초 단위로 다시 환산해야 하고, 처리 FPS가
   변하면(예: Edge Camera에서 프레임 드랍) 같은 15프레임이 더 이상 0.6초가 아니게 된다.
2. Cooldown은 "왜 억제되었는지"에 대한 물리적 근거가 없다 — 그냥 "최근에 이벤트가
   있었으니 무시"할 뿐이라 진짜로 빠르게 왕복한 사람(예: 문 앞에서 망설이다 들어간
   사람)의 두 번째 이벤트도 동일하게 묻어버린다.
3. band_px는 "선에서 실제로 몇 픽셀 떨어졌는가"라는 기하학적 의미가 있어 카메라
   설치 위치/해상도에 맞춰 튜닝하는 근거가 더 명확하고, 이미 EXP-012(People Counting)가
   같은 설계를 사용하고 있어 Codebase 안에서 "선 관련 판정은 band_px 불감대" 라는
   일관된 패턴을 유지할 수 있다.

## Regression Test
- `tests/test_line_crossing.py` 기존 7개 테스트 **수정 없이 그대로 통과** (band_px 기본값
  0으로 하위 호환 유지 확인).
- 신규 5개 테스트 추가: FC-007 재현(band_px=0에서 흔들림이 이벤트를 반복 발생시킴),
  band_px=5에서 같은 흔들림이 억제됨, band_px가 있어도 진짜 통과는 여전히 감지됨,
  `band_px` 기본값 검증, `signed_distance()`가 실제 픽셀 스케일과 일치하는지 검증.
- 전체 회귀: `pytest tests/ -q` → **53 passed, 7 skipped**(선택적 의존성 없는 환경에서
  skip되는 것으로 기존과 동일한 패턴, PAR-004 시점과 동일한 48+5=53).

## Next Action
FC-006(재등장 Track dwell 덮어쓰기, 로컬 세션에서 발견/미수정 상태로 Notion에 기록됨)
또는 Heatmap/Crowd Analysis. band_px=5는 미검증 임시값이므로, MOT20-01 GT를 확보해
정답 대비 통과 횟수(MAE)로 최적 band를 다시 검증하는 것은 향후 Action으로 남김
(EXP-012와 동일한 한계).
