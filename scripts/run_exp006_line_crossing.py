"""EXP-006: Line Crossing Detection.

crowded_intersection 영상에 가상선을 긋고, 방향(A_TO_B/B_TO_A)별 통과 인원을 센다.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from events.line_crossing import LineCrossingDetector  # noqa: E402
from events.roi_state_machine import bottom_center  # noqa: E402

PERSON_CLASS_ID = 0
# 횡단보도를 가로지르는 가상선 (도로와 인도 경계를 대략적으로 지정)
LINE_A, LINE_B = (700, 850), (1900, 780)


def main():
    out_dir = ROOT / "results/EXP-006"
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"
    cap = cv2.VideoCapture(str(video_path))

    detector = LineCrossingDetector(LINE_A, LINE_B)
    events = []
    last_seen = {}
    max_frames = 300
    idx = 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]

        ids, boxes = [], []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            for tid, box in zip(ids, boxes):
                last_seen[tid] = idx
                ev = detector.update(tid, bottom_center(tuple(box)), idx)
                if ev is not None:
                    events.append({"frame_idx": idx, "track_id": tid, "direction": ev.direction})

        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                detector.forget_track(tid)
                del last_seen[tid]

        if idx % 30 == 0:
            vis = frame.copy()
            cv2.line(vis, LINE_A, LINE_B, (255, 0, 255), 3)
            for tid, box in zip(ids, boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"id:{tid}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(str(snap_dir / f"frame_{idx:04d}.jpg"), vis)

        idx += 1

    cap.release()

    with open(out_dir / "line_crossing_events.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "track_id", "direction"])
        writer.writeheader()
        writer.writerows(events)

    a_to_b = sum(1 for e in events if e["direction"] == "A_TO_B")
    b_to_a = sum(1 for e in events if e["direction"] == "B_TO_A")
    print(f"Frames processed: {idx}")
    print(f"Line Crossing 이벤트: A_TO_B={a_to_b}, B_TO_A={b_to_a}, 총={len(events)}")
    print(f"\nSaved: {out_dir}/line_crossing_events.csv, snapshots in {snap_dir}")


if __name__ == "__main__":
    main()
