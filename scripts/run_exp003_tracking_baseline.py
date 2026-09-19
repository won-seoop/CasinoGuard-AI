"""EXP-003: Multi Object Tracking Baseline (ByteTrack, Ultralytics 내장).

공식 GT가 있는 MOT17은 압축파일이 5.8GB로 이번 세션에서는 다운로드하지 않는다
(용량 제약, 지침 31 Ground Truth 원칙에 따라 "없는 것"을 솔직하게 표시).
대신 직접 확보한 실제 영상에 Tracking을 적용하고, GT가 없는 상태에서
측정 가능한 지표(고유 Track ID 수, 동시 Track 수, FPS)만 정직하게 측정한다.
ID Switch/Fragmentation은 저장된 프레임을 눈으로 직접 검토해 정성적으로 기록한다.
"""

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

PERSON_CLASS_ID = 0


def run_tracking(model: YOLO, video_path: Path, tracker: str, max_frames: int, snapshot_dir: Path, snapshot_every: int = 30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    concurrent_counts: list[int] = []
    latencies: list[float] = []
    # 각 track_id가 마지막으로 관측된 프레임 idx (사라짐/재등장 확인용)
    last_seen_frame: dict[int, int] = {}
    reappear_events = 0  # 한동안(>=10프레임) 안 보이다가 다시 나타난 ID 수 (Track Fragmentation 후보)

    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = model.track(
            frame, persist=True, tracker=tracker, classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False
        )[0]
        latencies.append(time.perf_counter() - t0)

        ids = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = [int(i) for i in result.boxes.id.tolist()]

        for tid in ids:
            if tid in last_seen_frame and idx - last_seen_frame[tid] >= 10:
                reappear_events += 1
            last_seen_frame[tid] = idx
        seen_ids.update(ids)
        concurrent_counts.append(len(ids))

        if idx % snapshot_every == 0:
            annotated = result.plot()
            cv2.imwrite(str(snapshot_dir / f"frame_{idx:04d}.jpg"), annotated)

        idx += 1

    cap.release()
    total_time = sum(latencies)
    return {
        "video": video_path.name,
        "frames": idx,
        "unique_track_ids": len(seen_ids),
        "max_concurrent_tracks": max(concurrent_counts) if concurrent_counts else 0,
        "avg_concurrent_tracks": round(sum(concurrent_counts) / len(concurrent_counts), 2) if concurrent_counts else 0,
        "reappear_events_ge10frames": reappear_events,
        "fps": round(idx / total_time, 2) if total_time else 0,
    }


def main():
    out_dir = ROOT / "results/EXP-003"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")

    videos = {
        "normal_480p": ROOT / "data/raw/pedestrian_crossing_480p.ogg",
        "crowded_1080p": ROOT / "data/raw/crowded_intersection_1080p.webm",
    }

    rows = []
    for name, path in videos.items():
        snap_dir = out_dir / "snapshots" / name
        res = run_tracking(model, path, tracker="bytetrack.yaml", max_frames=200, snapshot_dir=snap_dir)
        if res:
            rows.append(res)
            print(res)

    with open(out_dir / "tracking_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_dir}/tracking_summary.csv, snapshots in {out_dir}/snapshots/")


if __name__ == "__main__":
    main()
