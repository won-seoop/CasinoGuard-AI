"""EXP-007: Loitering Detection — Dwell Time Threshold 비교.

지침 15번: Threshold 5/10/20/30초를 비교한다.
crowded_intersection 영상은 실제 원본 FPS가 23.976이므로 frame 수로 환산해서 사용한다.
같은 ROI(EXP-005와 동일한 횡단보도 Polygon)를 재사용한다.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from events.loitering import LoiteringDetector  # noqa: E402
from events.roi_state_machine import bottom_center  # noqa: E402

PERSON_CLASS_ID = 0
ROI_POLYGON = [(950, 650), (1750, 650), (1850, 1080), (700, 1080)]  # EXP-005와 동일
THRESHOLDS_SEC = [5, 10, 20, 30]


def collect_track_points(model: YOLO, video_path: Path, max_frames: int):
    """(frame_idx, track_id, bottom_center_point) 리스트를 한 번의 추론으로 수집해
    여러 Threshold에 재사용한다 (같은 영상을 4번 다시 돌리지 않기 위한 효율화).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    records = []
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            for tid, box in zip(ids, boxes):
                records.append((idx, tid, bottom_center(tuple(box))))
        idx += 1
    cap.release()
    return records, fps, idx


def run_threshold(records, fps, threshold_sec, total_frames):
    threshold_frames = int(round(threshold_sec * fps))
    det = LoiteringDetector(polygon=ROI_POLYGON, threshold_frames=threshold_frames)
    events = []
    seen_this_frame = {}
    max_frame_idx = 0
    for frame_idx, tid, point in records:
        max_frame_idx = max(max_frame_idx, frame_idx)
        ev = det.update(tid, point, frame_idx)
        if ev is not None:
            events.append({"threshold_sec": threshold_sec, "track_id": tid, "frame_idx": frame_idx, "dwell_sec": round(ev.dwell_frames / fps, 2)})
    return threshold_frames, events


def main():
    out_dir = ROOT / "results/EXP-007"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"

    max_frames = 800  # 약 33초 분량 (30초 Threshold까지 테스트하기 위해 EXP-005/006보다 길게 처리)
    print(f"Collecting tracking points over {max_frames} frames...")
    records, fps, total_frames = collect_track_points(model, video_path, max_frames)
    print(f"fps={fps:.2f}, processed frames={total_frames}, track observations={len(records)}")

    all_events = []
    summary_rows = []
    for th_sec in THRESHOLDS_SEC:
        th_frames, events = run_threshold(records, fps, th_sec, total_frames)
        all_events.extend(events)
        summary_rows.append({"threshold_sec": th_sec, "threshold_frames": th_frames, "loitering_events": len(events)})
        print(f"threshold={th_sec}s ({th_frames} frames) -> {len(events)} loitering event(s)")

    with open(out_dir / "loitering_events.csv", "w", newline="") as f:
        if all_events:
            writer = csv.DictWriter(f, fieldnames=list(all_events[0].keys()))
            writer.writeheader()
            writer.writerows(all_events)

    with open(out_dir / "threshold_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {out_dir}/loitering_events.csv, {out_dir}/threshold_summary.csv")


if __name__ == "__main__":
    main()
