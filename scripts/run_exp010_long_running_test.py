"""EXP-010: Long Running Stability Test.

지침 28번: CCTV 시스템은 장시간 실행되므로 Memory/FPS/Dropped Frame 변화를 측정해야 한다.

Blocker / Decision (experiment.md Decision 섹션에도 기록):
이번 세션(원격 실행 환경)은 egress 네트워크 정책상 Wikimedia Commons 등
외부 일반 도메인 접근이 차단되어(403), 기존 EXP-001~009에서 사용한
crowded_intersection_1080p.webm / pedestrian_crossing_480p.ogg 원본 영상을
다시 받을 수 없었다. 대신 ultralytics 패키지에 기본 포함된 실제 사진
bus.jpg(사람 4명이 버스를 기다리는 실제 사진, YOLO 공식 샘플)를 기반으로,
프레임마다 작은 Pan/Zoom/밝기 Jitter를 주어 "카메라가 정지된 채 장시간
동일 장면을 촬영하는" 상황을 재현했다. 이는 카지노 CCTV에서 흔한
"딜러/캐셔가 한 자리에 오래 머무는" 시나리오와 유사하며, Detection 정확도
검증이 아니라 파이프라인의 장시간 Memory/FPS 안정성 검증이 목적이므로
실제 사람이 찍힌 정지 장면을 반복 처리하는 것으로 충분하다.

측정: RSS Memory(MB), FPS(rolling), P95 Latency(rolling), 누적 Track 수,
BestShot 후보 버퍼에 쌓인 관측치 총 개수(메모리 누적의 직접적 proxy),
예외 발생 횟수.

안전장치: RSS가 시작 대비 SAFE_RSS_GROWTH_MB 이상 증가하거나 MAX_FRAMES에
도달하면 즉시 종료한다(컨테이너 OOM 방지).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bestshot.scorer import bestshot_score  # noqa: E402
from bestshot.tracker import BestShotTracker  # noqa: E402
from events.line_crossing import LineCrossingDetector  # noqa: E402
from events.loitering import LoiteringDetector  # noqa: E402
from events.roi_state_machine import ROIStateMachine, bottom_center  # noqa: E402
from metadata.store import MetadataStore  # noqa: E402

PERSON_CLASS_ID = 0
ASSET_IMG = Path("/usr/local/lib/python3.11/dist-packages/ultralytics/assets/bus.jpg")

MAX_FRAMES = 20000
SAFE_RSS_GROWTH_MB = 3072.0  # 시작 대비 3GB 이상 증가하면 즉시 중단 (OOM 방지)
SAMPLE_EVERY = 200


def rss_mb() -> float:
    """psutil 등 추가 의존성 없이 /proc/self/status에서 RSS를 직접 읽는다 (Linux 전용)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def make_frame_generator(base_img: np.ndarray, frame_idx: int) -> np.ndarray:
    """정지 장면 + 작은 Pan/밝기 Jitter로 "카메라 고정, 장시간 촬영" 상황을 재현.
    (Decision: 실제 영상 다운로드 불가로 인한 대체 입력, 상단 docstring 참고)
    """
    h, w = base_img.shape[:2]
    max_shift = 12
    dx = int(max_shift * np.sin(frame_idx * 0.037))
    dy = int(max_shift * np.cos(frame_idx * 0.053))
    shifted = np.roll(base_img, shift=(dy, dx), axis=(0, 1))
    brightness = 1.0 + 0.03 * np.sin(frame_idx * 0.011)
    frame = np.clip(shifted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return frame


def run(use_fix: bool, max_frames: int, out_csv: Path) -> dict:
    base_img = cv2.imread(str(ASSET_IMG))
    frame_h, frame_w = base_img.shape[:2]
    model = YOLO("yolo11n.pt")

    roi_polygon = [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]  # 전체 프레임을 ROI로 (dwell 측정 목적)
    roi_sm = ROIStateMachine(polygon=roi_polygon)
    line_det = LineCrossingDetector((0, frame_h // 2), (frame_w, frame_h // 2))
    loiter_det = LoiteringDetector(polygon=roi_polygon, threshold_frames=999999)  # 이번 실험은 Loitering 자체가 목적이 아님

    db_path = ROOT / "results" / "EXP-010" / f"metadata_{'fixed' if use_fix else 'baseline'}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    store = MetadataStore(db_path)

    # Baseline: 지침 개선 전 run_full_pipeline.py와 동일하게 관측치 전체를 리스트로 누적한다.
    bestshot_candidates: dict[int, list[dict]] = {}
    bestshot_tracker = BestShotTracker(frame_w, frame_h) if use_fix else None

    last_seen: dict[int, int] = {}
    exception_count = 0
    rows = []

    start_rss = rss_mb()
    t_start = time.perf_counter()
    latencies: list[float] = []
    idx = 0
    stop_reason = "max_frames_reached"

    while idx < max_frames:
        frame = make_frame_generator(base_img, idx)
        t0 = time.perf_counter()
        try:
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False
            )[0]
        except Exception:
            exception_count += 1
            idx += 1
            continue
        latencies.append(time.perf_counter() - t0)

        active_boxes = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            active_boxes = list(zip(ids, boxes, confs))
        all_boxes_this_frame = [b for _, b, _ in active_boxes]

        for tid, box, conf in active_boxes:
            box_t = tuple(box)
            point = bottom_center(box_t)
            last_seen[tid] = idx
            roi_sm.update(tid, box_t, idx)
            line_det.update(tid, point, idx)
            loiter_det.update(tid, point, idx)
            store.upsert_track(tid, "person", idx, conf, box_t, point, None)

            x1, y1, x2, y2 = [max(0, int(v)) for v in box_t]
            crop = frame[y1:y2, x1:x2].copy()
            others = [b for b in all_boxes_this_frame if b != box]

            if use_fix:
                score = bestshot_score(conf, crop, box_t, others, frame_w, frame_h)
                bestshot_tracker.observe(tid, idx, crop, score["total"])
            else:
                bestshot_candidates.setdefault(tid, []).append({"frame_idx": idx, "crop": crop})

        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                roi_sm.forget_track(tid, frame_idx=idx)
                line_det.forget_track(tid)
                loiter_det.forget_track(tid)
                if use_fix:
                    bestshot_tracker.forget_track(tid)
                del last_seen[tid]

        if idx % SAMPLE_EVERY == 0 and idx > 0:
            elapsed = time.perf_counter() - t_start
            window = latencies[-SAMPLE_EVERY:]
            fps_roll = len(window) / sum(window) if sum(window) else 0.0
            p95 = float(np.percentile(np.array(window) * 1000, 95)) if window else 0.0
            cur_rss = rss_mb()
            if use_fix:
                buffered_obs = bestshot_tracker.buffered_count()
            else:
                buffered_obs = sum(len(v) for v in bestshot_candidates.values())
            row = {
                "elapsed_sec": round(elapsed, 1),
                "frame_idx": idx,
                "rss_mb": round(cur_rss, 1),
                "rss_growth_mb": round(cur_rss - start_rss, 1),
                "fps_rolling": round(fps_roll, 2),
                "p95_ms_rolling": round(p95, 2),
                "n_tracks_total": len(store.query_tracks()),
                "buffered_observations": buffered_obs,
                "exception_count": exception_count,
            }
            rows.append(row)
            print(row)

            if cur_rss - start_rss > SAFE_RSS_GROWTH_MB:
                stop_reason = "safety_abort_rss_growth"
                idx += 1
                break

        idx += 1

    # 종료 시점 정리: 아직 활성 상태인 Track들도 forget 처리
    for tid in list(last_seen.keys()):
        roi_sm.forget_track(tid, frame_idx=idx)
        if use_fix:
            bestshot_tracker.forget_track(tid)

    total_elapsed = time.perf_counter() - t_start
    store.close()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "mode": "fixed" if use_fix else "baseline",
        "frames_processed": idx,
        "wall_clock_sec": round(total_elapsed, 1),
        "stop_reason": stop_reason,
        "start_rss_mb": round(start_rss, 1),
        "end_rss_mb": round(rows[-1]["rss_mb"], 1) if rows else round(start_rss, 1),
        "rss_growth_mb": round((rows[-1]["rss_mb"] if rows else start_rss) - start_rss, 1),
        "exception_count": exception_count,
    }
    print("SUMMARY:", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "fixed"], required=True)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "EXP-010"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"timeseries_{args.mode}.csv"

    summary = run(use_fix=(args.mode == "fixed"), max_frames=args.max_frames, out_csv=out_csv)

    import json

    with open(out_dir / f"summary_{args.mode}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
