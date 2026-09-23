"""EXP-016: Heatmap & Crowd Analysis Baseline (지침 21, 22번).

Goal:
- Heatmap: Track 위치를 시간에 따라 누적해 사람이 많이 머무는 공간을 시각화한다.
  지침 21번이 명시한 "Detection Center 누적 vs Track Trajectory 누적" 비교를 수행한다.
- Crowd Analysis: ROI(Grid Cell)별 Person Count를 Normal/Busy/Crowded로 분류한다.
  지침 22번이 명시한 대로 Threshold는 임의로 정하지 않고 실제 Dataset의 Distribution에서 뽑는다.

Blocker / Decision (EXP-010/014/015와 동일):
이 원격 세션은 egress 정책상 Wikimedia 등 외부 일반 도메인이 차단되어 기존
crowded_intersection_1080p.webm을 다시 받을 수 없다(commons.wikimedia.org 403 확인).
대신 ultralytics 내장 실제 사진(bus.jpg, 사람 4명 이상 포함)에 EXP-010/014와 동일한
Pan/밝기 Jitter를 주어 "카메라가 고정된 채 사람이 화면 안에서 흔들리며 움직이는" 상황을
재현한다. Detection 자체는 매 프레임 실제 YOLO11n 추론 결과이므로, 누적 로직/노이즈
필터링/Threshold 도출이라는 이 실험의 목적에는 이 대체 입력으로 충분하다.

측정할 실제 문제(가설):
Pan은 np.roll로 구현되어 프레임 가장자리에서 반대편 내용이 wrap-around로 나타난다.
이 wrap 경계에서 몸이 잘린 사람이나 배경 일부가 순간적으로 사람으로 오탐되어
아주 짧게(몇 프레임) 존재했다가 사라지는 Track이 생길 수 있다. Raw Detection Center
누적(Baseline)은 이런 짧은 오탐성 Track까지 그대로 Heatmap에 반영해 노이즈를 만들 것이다.
이를 Track이 min_track_len 이상 관측된 뒤에만 누적하는 Track-Gated 누적(대안)과
정량 비교한다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from analytics.crowd import CrowdAnalyzer, classify, derive_thresholds_from_distribution  # noqa: E402
from analytics.heatmap import HeatmapAccumulator, TrackGatedFeeder  # noqa: E402
from events.roi_state_machine import bottom_center  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_CANDIDATES = [
    Path(sys.prefix) / "lib/python3.11/dist-packages/ultralytics/assets/bus.jpg",
    Path(sys.prefix) / "lib/python3.11/site-packages/ultralytics/assets/bus.jpg",
]
N_FRAMES = 900  # EXP-010/014와 동일 (약 36초 @ 25fps 가정, Pan 여러 주기 포함)
ASSUMED_FPS = 25.0
MIN_TRACK_LEN = 5  # 0.2초 @ 25fps. ByteTrack 기본 트랙 확정 관성보다 짧게 잡아 "노이즈만" 걸러지는지 관찰.
GRID_ROWS, GRID_COLS = 3, 4
STALE_FRAMES = 10  # 이 프레임 동안 안 보이면 forget_track


def find_asset() -> Path:
    for p in ASSET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"ultralytics 샘플 이미지를 찾을 수 없음: {ASSET_CANDIDATES}")


def make_frame(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    """EXP-010/014와 동일한 Pan/밝기 Jitter (재현성을 위해 같은 파라미터 사용)."""
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037))
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    frame = np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return frame


def run_detection(base_img: np.ndarray) -> list[list[tuple[int, tuple[float, float, float, float]]]]:
    model = YOLO("yolo11n.pt")
    boxes_per_frame = []
    for idx in range(N_FRAMES):
        frame = make_frame(base_img, idx)
        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False
        )[0]
        active: list[tuple[int, tuple[float, float, float, float]]] = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            active = [(tid, tuple(b)) for tid, b in zip(ids, boxes)]
        boxes_per_frame.append(active)
    return boxes_per_frame


def track_length_stats(boxes_per_frame: list) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for active in boxes_per_frame:
        for tid, _ in active:
            lengths[tid] = lengths.get(tid, 0) + 1
    return lengths


def run_heatmap_comparison(boxes_per_frame: list, frame_w: int, frame_h: int) -> dict:
    baseline_acc = HeatmapAccumulator(frame_w, frame_h, cell_size=20)
    gated_acc = HeatmapAccumulator(frame_w, frame_h, cell_size=20)
    feeder = TrackGatedFeeder(gated_acc, min_track_len=MIN_TRACK_LEN)

    last_seen: dict[int, int] = {}
    for idx, active in enumerate(boxes_per_frame):
        t_sec = idx / ASSUMED_FPS
        for tid, box in active:
            x, y = bottom_center(box)
            baseline_acc.add_point(x, y, timestamp_sec=t_sec)
            feeder.observe(tid, x, y, timestamp_sec=t_sec)
            last_seen[tid] = idx
        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > STALE_FRAMES:
                feeder.forget_track(tid)
                del last_seen[tid]
    # 영상 종료 시점까지 살아있던 Track도 정리 (버퍼 상태 확정)
    for tid in list(last_seen.keys()):
        feeder.forget_track(tid)

    lengths = track_length_stats(boxes_per_frame)
    short_tracks = {tid: n for tid, n in lengths.items() if n < MIN_TRACK_LEN}
    sustained_tracks = {tid: n for tid, n in lengths.items() if n >= MIN_TRACK_LEN}

    return {
        "baseline_total_hits": baseline_acc.total_hits(),
        "gated_total_hits": gated_acc.total_hits(),
        "hits_removed": baseline_acc.total_hits() - gated_acc.total_hits(),
        "hits_removed_pct": (
            100.0 * (baseline_acc.total_hits() - gated_acc.total_hits()) / baseline_acc.total_hits()
            if baseline_acc.total_hits() > 0
            else 0.0
        ),
        "total_unique_tracks": len(lengths),
        "short_track_count": len(short_tracks),
        "sustained_track_count": len(sustained_tracks),
        "feeder_discarded_track_count": feeder.discarded_track_count,
        "feeder_discarded_point_count": feeder.discarded_point_count,
        "feeder_confirmed_track_count": feeder.confirmed_track_count,
        "baseline_acc": baseline_acc,
        "gated_acc": gated_acc,
    }


def run_crowd_analysis(boxes_per_frame: list, frame_w: int, frame_h: int, gated_confirmed_only_points: list) -> dict:
    analyzer = CrowdAnalyzer(frame_w, frame_h, GRID_ROWS, GRID_COLS)

    raw_cell_counts: list[int] = []
    for active in boxes_per_frame:
        points = [bottom_center(box) for _, box in active]
        grid = analyzer.count_frame(points)
        raw_cell_counts.extend(grid.flatten().tolist())

    thresholds = derive_thresholds_from_distribution(raw_cell_counts, normal_pct=50.0, busy_pct=90.0)

    # 대안 비교: 임의로 정한 고정 Threshold (지침 22가 하지 말라고 명시한 방식) — 실제로 얼마나 다른 분류가 나오는지 비교용
    naive_fixed = derive_thresholds_from_distribution([0, 0, 0, 3, 6], normal_pct=50.0, busy_pct=90.0)  # 임의 추정값

    classification_counts_percentile = {"Normal": 0, "Busy": 0, "Crowded": 0}
    classification_counts_naive = {"Normal": 0, "Busy": 0, "Crowded": 0}
    for c in raw_cell_counts:
        classification_counts_percentile[classify(c, thresholds)] += 1
        classification_counts_naive[classify(c, naive_fixed)] += 1

    return {
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "cell_count_distribution": {
            "min": float(np.min(raw_cell_counts)) if raw_cell_counts else 0.0,
            "p50": float(np.percentile(raw_cell_counts, 50)) if raw_cell_counts else 0.0,
            "p90": float(np.percentile(raw_cell_counts, 90)) if raw_cell_counts else 0.0,
            "max": float(np.max(raw_cell_counts)) if raw_cell_counts else 0.0,
        },
        "thresholds_percentile": {"normal_max": thresholds.normal_max, "busy_max": thresholds.busy_max},
        "thresholds_naive_fixed": {"normal_max": naive_fixed.normal_max, "busy_max": naive_fixed.busy_max},
        "classification_counts_percentile": classification_counts_percentile,
        "classification_counts_naive": classification_counts_naive,
    }


def save_heatmap_image(acc: HeatmapAccumulator, base_img: np.ndarray, out_path: Path) -> None:
    heat = acc.to_normalized_image()
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_img, 0.5, color, 0.5, 0)
    cv2.imwrite(str(out_path), overlay)


def save_crowd_classification_plot(crowd_result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Normal", "Busy", "Crowded"]
    percentile_vals = [crowd_result["classification_counts_percentile"][k] for k in labels]
    naive_vals = [crowd_result["classification_counts_naive"][k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, naive_vals, width, label="Naive Fixed Threshold")
    ax.bar(x + width / 2, percentile_vals, width, label="Percentile-based Threshold")
    ax.set_ylabel("Cell-frame count")
    ax.set_title("Crowd Classification: Naive Fixed vs Percentile-based Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run() -> dict:
    base_img = cv2.imread(str(find_asset()))
    frame_h, frame_w = base_img.shape[:2]

    boxes_per_frame = run_detection(base_img)
    heatmap_result = run_heatmap_comparison(boxes_per_frame, frame_w, frame_h)
    crowd_result = run_crowd_analysis(boxes_per_frame, frame_w, frame_h, [])

    out_dir = ROOT / "results" / "EXP-016"
    (out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    save_heatmap_image(heatmap_result["baseline_acc"], base_img, out_dir / "heatmaps" / "baseline_raw_detection.png")
    save_heatmap_image(heatmap_result["gated_acc"], base_img, out_dir / "heatmaps" / "alternative_track_gated.png")
    save_crowd_classification_plot(crowd_result, out_dir / "plots" / "crowd_threshold_comparison.png")

    summary = {
        "n_frames": N_FRAMES,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "min_track_len": MIN_TRACK_LEN,
        "heatmap": {k: v for k, v in heatmap_result.items() if k not in ("baseline_acc", "gated_acc")},
        "crowd": crowd_result,
    }
    return summary


if __name__ == "__main__":
    result = run()
    out_dir = ROOT / "results" / "EXP-016"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "heatmap_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in result["heatmap"].items():
            writer.writerow([k, v])
    print(json.dumps(result, indent=2))
    print("saved to", out_dir)
