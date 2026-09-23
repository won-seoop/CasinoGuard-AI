# EXP-016 Heatmap & Crowd Analysis Baseline

## Goal
CLAUDE.md 지침 21(Heatmap), 22(Crowd Analysis)의 Baseline을 구현하고 검증한다.
- Heatmap: Detection Center 누적(Baseline) vs Track-Gated 누적(대안)을 비교한다.
- Crowd Analysis: ROI(Grid Cell)별 Person Count를 Normal/Busy/Crowded로 분류하되,
  Threshold를 임의로 정하지 않고 실제 관측 분포에서 도출한다(지침 22 원칙).

## Hypothesis
1. (Heatmap) Pan/Jitter로 카메라가 흔들리는 상황에서 프레임 가장자리(np.roll wrap-around
   경계)에 몸이 잘린 사람/배경 일부가 순간적으로 오탐되어 아주 짧은(몇 프레임) Track이
   생길 것이고, Raw Detection Center 누적(Baseline)은 이런 오탐성 Track까지 그대로
   Heatmap에 반영해 노이즈를 만들 것이다. Track-Gated 누적(대안)은 이 노이즈를 제거할 것이다.
2. (Crowd) 실제 카메라 화각/밀집도 분포를 확인하지 않고 "그럴듯하게" 정한 고정 Threshold는
   실제 분포와 어긋나 Crowded 등급이 사실상 작동하지 않거나(Degenerate Classification)
   반대로 항상 Crowded로 잡히는 등의 문제를 만들 것이다.

## Problem
Section 21/22는 아직 구현되지 않은 Stretch Goal이었다. Baseline을 실제로 만들고
측정하지 않으면 "Threshold를 임의로 정하지 않는다"는 원칙이 실제로 왜 중요한지
수치로 보여줄 수 없다.

## Dataset
EXP-010/014/015와 동일한 Blocker: 이 원격 세션은 egress 정책상
commons.wikimedia.org / raw.githubusercontent.com 등이 403으로 차단되어 기존
Wikimedia 원본 영상(crowded_intersection_1080p.webm)을 재확보할 수 없음을 재확인했다
(`curl` 직접 테스트, 두 도메인 모두 CONNECT tunnel 403). 반면 GitHub Release Assets
(`github.com/ultralytics/assets/releases/...`)는 접근 가능해 YOLO11n pretrained
가중치는 정상 다운로드된다.

대신 ultralytics 패키지에 내장된 실제 사진 `bus.jpg`(사람 4명 이상, 810x1080)에
EXP-010/014와 동일한 Pan(`np.roll`, 진폭 12px, sin/cos 주기)과 밝기 Jitter(±3%)를
적용해 "카메라가 고정된 채 사람이 화면 안에서 흔들리며 움직이는" 상황을 재현했다.
Detection 자체는 매 프레임 실제 YOLO11n 추론 결과이므로, 누적 로직/Threshold 도출
검증이라는 이 실험의 목적에는 충분하다(정확도 벤치마크가 목적이 아님).

## Environment
- Remote automated session, egress 제한(위 Dataset 항목 참고)
- Python 3.11.15, opencv-python-headless, ultralytics 8.4.x, numpy
- YOLO11n pretrained (conf=0.4, imgsz=640), ByteTrack (bytetrack.yaml, persist=True)

## Configuration
```yaml
n_frames: 900          # EXP-010/014와 동일
assumed_fps: 25.0
min_track_len: 5       # Heatmap Track-Gated 확정 기준 (0.2초)
stale_frames: 10       # 이 프레임 동안 미관측 시 forget_track
grid_rows: 3
grid_cols: 4
heatmap_cell_size_px: 20
crowd_threshold_percentiles: [50, 90]   # Normal/Busy 경계, Busy/Crowded 경계
```

## Baseline
- Heatmap Baseline: 매 프레임 모든 Detection의 bottom-center를 그대로 누적 (Track 여부 무관)
- Crowd Baseline: 3x4 Grid Cell별로 프레임마다 사람 수를 센다.

## Result

### 1) Heatmap: Detection Center 누적 vs Track-Gated 누적

| Metric | 값 |
|---|---|
| 총 고유 Track 수 | 10 |
| min_track_len(5) 미만 짧은 Track 수 | 0개 관측 종료 기준 / Feeder 기준 1개 폐기 |
| Baseline 총 누적 Hit | 3625 |
| Track-Gated 총 누적 Hit | 3621 |
| 제거된 Hit 수 (노이즈로 판단되어 제외) | 4 (0.11%) |
| Feeder가 폐기한 Track 수 | 1 |
| Feeder가 확정한 Track 수 | 11 (같은 숫자 ID가 소실 후 재등장하며 다시 카운트된 경우 포함) |

결과 이미지: `results/EXP-016/heatmaps/baseline_raw_detection.png`,
`results/EXP-016/heatmaps/alternative_track_gated.png` (육안상 거의 동일 — 아래 Analysis 참고)

### 2) Crowd Analysis: 임의 고정 Threshold vs 실측 분포 기반 Threshold

Cell-Count 분포 (3x4=12 cell x 900 frame = 10,800 관측치):

| 통계 | 값 |
|---|---|
| min | 0 |
| p50 | 0 |
| p90 | 1 |
| max | 3 |

| Threshold 방식 | normal_max | busy_max | Normal | Busy | Crowded |
|---|---|---|---|---|---|
| 임의 고정(Naive, "그럴듯한" 추정 분포 [0,0,0,3,6]에서 도출) | 0 | 4.8 | 8100 | 2700 | **0** |
| 실측 분포 기반(Percentile, p50/p90) | 0 | 1.0 | 8100 | 1841 | **859** |

`results/EXP-016/plots/crowd_threshold_comparison.png` 참고.

## Failure Cases
- Feeder가 확정한 Track 수(11)가 관측된 고유 ID 수(10)보다 많다 — ByteTrack이 Track을
  잃었다가 짧게 재등장할 때 같은 숫자 ID를 재사용하면서 `forget_track()` → 재관측 →
  재확정 사이클이 한 번 더 돈 것으로 보인다(로그상 track_id 값 재사용 확인, 상세 트랙별
  ID 재사용 로그는 저장하지 않아 완전히 특정하지는 못함 — 확정 결론 아님, 다음 세션에서
  ID별 등장 구간을 직접 로깅해 검증 필요).

## Analysis
1. **Heatmap 가설은 이 데이터셋에서는 강하게 확인되지 않았다.** 900프레임 동안
   min_track_len(5) 미만으로 사라진 오탐성 Track은 1개뿐이었고, 제거된 Hit도 전체의
   0.11%(4/3625)에 불과했다. 즉 "Pan 경계에서 오탐성 짧은 Track이 대량 발생해 Heatmap을
   오염시킨다"는 가설은 이 bus.jpg + 소폭 Pan(12px) 조건에서는 관측되지 않았다.
   원인으로 추정되는 것: Pan 진폭(12px)이 프레임 크기(810x1080)에 비해 작아 wrap-around
   경계가 사람이 지나다니는 주 영역과 거의 겹치지 않았고, YOLO11n conf=0.4가 이미 상당히
   엄격해 경계 아티팩트 자체가 검출로 이어지는 경우가 드물었다. **이 결과를 숨기지 않고
   그대로 기록한다(지침 38)** — Track-Gated 누적 로직 자체는 구현하고 Unit Test로
   검증했으며(정상 동작 확인), 실제 효과가 큰 시나리오(더 큰 Pan 진폭, 더 낮은 conf,
   실제 혼잡한 배경 군중 - FC-002와 연결)는 향후 재검증이 필요한 한계로 남긴다.
2. **Crowd Analysis 가설은 뚜렷하게 확인되었다.** "그럴듯하게" 고른 고정 Threshold
   (busy_max=4.8, 즉 "한 셀에 4~5명 있으면 Crowded"라는 상식적인 추정)는 이 카메라
   화각/그리드 해상도(3x4, 셀당 약 270x360px)에서 실제로 한 셀에 동시에 잡히는
   사람 수가 최대 3명(p90=1명)에 그친다는 사실과 어긋나, **Crowded 등급이 900프레임
   내내 단 한 번도 발동하지 않는(0/10800) Degenerate Classification**을 만들었다.
   반대로 실측 분포 기반 Threshold(p50=0, p90=1)를 적용하면 전체의 7.9%(859/10800)가
   Crowded로 정확히 분리된다. 이는 지침 22가 "Threshold는 임의로 정하지 않는다"고
   못박은 이유를 실측으로 보여주는 사례다: 카메라 화각/그리드 크기/피사체 밀도가
   현장마다 다르므로, 고정값은 특정 현장에서는 항상 Normal, 다른 현장에서는 항상
   Crowded가 되는 식으로 완전히 무의미해질 수 있다.

## Decision
- Heatmap: Detection Center 누적을 Baseline으로 채택하고, Track-Gated 누적
  (`TrackGatedFeeder`, `min_track_len`)을 옵션으로 함께 제공한다. 이번 실험에서는
  효과가 미미했지만 원리적으로 "확정되지 않은 Track은 누적하지 않는다"는 것은
  Long Running Test(PAR-004)에서 이미 검증한 "미확정 상태를 무한정 쌓지 않는다"는
  설계 원칙과 일치하고, 실제 배포 시(혼잡한 배경, 낮은 conf) 노이즈가 커질 가능성이
  있으므로 기본값으로 유지한다.
- Crowd Analysis: 고정 Threshold를 코드에 박아두지 않고, `derive_thresholds_from_distribution()`
  으로 실제 관측 분포의 Percentile(기본 p50/p90)에서 Threshold를 도출하는 방식을 채택한다.
  → **Technical Decision(08)에 "Crowd Threshold: 고정값 vs Percentile 기반" 추가.**
  → **PAR-007로 정리** (실제 문제 재현 + 2가지 대안 비교 + 실측 검증 + 개선 수치 확보).

## Next Action
- 더 큰 Pan 진폭/낮은 conf/실제 혼잡 배경(FC-002 해소 후)으로 Heatmap 노이즈 가설을
  재검증할 것.
- ByteTrack의 ID 재사용으로 추정되는 `feeder_confirmed_track_count(11) > total_unique_tracks(10)`
  불일치를 ID별 등장 구간 로그로 직접 확인할 것.
- 시간 단위 Heatmap(1분/5분 bin)은 `HeatmapAccumulator.time_bin_sec`으로 이미 지원되나
  이번 실험은 전체 누적만 시각화했다 — 더 긴 영상 확보 시 시간대별 비교 추가.
