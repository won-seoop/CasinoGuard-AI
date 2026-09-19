"""EXP-005: Intrusion Detection — Polygon ROI + State Machine.

Baseline(안 좋은 방식): Track이 ROI 안에 있는 "모든 프레임"을 이벤트로 카운트 -> 중복 폭발
개선(State Machine): 상태 전이(OUTSIDE->INSIDE) 시점에만 이벤트 발생

두 방식의 이벤트 개수를 직접 비교해서 중복 방지 효과를 정량적으로 보여준다.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from events.roi_state_machine import ROIStateMachine, bottom_center, point_in_polygon  # noqa: E402

PERSON_CLASS_ID = 0

# crowded_intersection_1080p(1920x1080)의 횡단보도 영역을 대략적으로 지정
# (카지노의 "통제 구역 진입 감지"를 시뮬레이션하는 데모 ROI)
ROI_POLYGON = [(950, 650), (1750, 650), (1850, 1080), (700, 1080)]


def main():
    out_dir = ROOT / "results/EXP-005"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"
    cap = cv2.VideoCapture(str(video_path))

    sm = ROIStateMachine(polygon=ROI_POLYGON)
    events = []
    naive_event_count = 0  # Baseline: ROI 안에 있는 모든 프레임을 이벤트로 카운트했다면?
    max_frames = 300
    idx = 0
    last_seen = {}

    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]

        active_ids_this_frame = set()
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]
            boxes = result.boxes.xyxy.tolist()
            for tid, box in zip(ids, boxes):
                active_ids_this_frame.add(tid)
                last_seen[tid] = idx
                if point_in_polygon(bottom_center(tuple(box)), ROI_POLYGON):
                    naive_event_count += 1  # Baseline 카운트

                ev = sm.update(tid, tuple(box), idx)
                if ev is not None:
                    events.append({"frame_idx": idx, "track_id": tid, "event_type": ev.event_type})

        # 10프레임 이상 안 보인 Track은 상태 정리 (Long Running 시 메모리 누적 방지)
        for tid in list(last_seen.keys()):
            if idx - last_seen[tid] > 10:
                sm.forget_track(tid)
                del last_seen[tid]

        if idx % 30 == 0:
            vis = frame.copy()
            cv2.polylines(vis, [__import__("numpy").array(ROI_POLYGON)], True, (0, 0, 255), 3)
            for tid, box in zip(ids if result.boxes is not None and result.boxes.id is not None else [], result.boxes.xyxy.tolist() if result.boxes is not None else []):
                x1, y1, x2, y2 = [int(v) for v in box]
                inside = point_in_polygon(bottom_center((x1, y1, x2, y2)), ROI_POLYGON)
                color = (0, 0, 255) if inside else (0, 255, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"id:{tid}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imwrite(str(snap_dir / f"frame_{idx:04d}.jpg"), vis)

        idx += 1

    cap.release()

    with open(out_dir / "intrusion_events.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "track_id", "event_type"])
        writer.writeheader()
        writer.writerows(events)

    enter_count = sum(1 for e in events if e["event_type"] == "ENTER")
    exit_count = sum(1 for e in events if e["event_type"] == "EXIT")

    print(f"Frames processed: {idx}")
    print(f"State-Machine 기반 이벤트: ENTER={enter_count}, EXIT={exit_count}, 총={len(events)}")
    print(f"Baseline(중복 방지 없음) 방식이었다면 발생했을 이벤트 수: {naive_event_count}")
    print(f"중복 억제율: {(1 - len(events)/naive_event_count)*100:.1f}%" if naive_event_count else "N/A")
    print(f"\nSaved: {out_dir}/intrusion_events.csv, snapshots in {snap_dir}")


if __name__ == "__main__":
    main()
