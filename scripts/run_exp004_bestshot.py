"""EXP-004: BestShot — Track별 대표 프레임 선택.

Baseline: Random Frame
대안 A: Highest Detection Confidence
대안 B: 자체 BestShot Score (Confidence+Sharpness+Size+Occlusion+Position)

각 방식으로 고른 대표 프레임을 Track별 Image Grid로 저장해서 비교한다.
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bestshot.scorer import bestshot_score  # noqa: E402

PERSON_CLASS_ID = 0
random.seed(42)


def collect_track_observations(model: YOLO, video_path: Path, max_frames: int):
    """Track별로 (frame_idx, bbox, conf, crop) 관측치를 모은다."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}, 0, 0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    observations: dict[int, list[dict]] = defaultdict(list)
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON_CLASS_ID], conf=0.4, imgsz=640, verbose=False)[0]

        boxes_this_frame = []
        if result.boxes is not None and result.boxes.id is not None:
            xyxy = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            ids = [int(i) for i in result.boxes.id.tolist()]
            for tid, box, conf in zip(ids, xyxy, confs):
                boxes_this_frame.append((tid, tuple(box), conf))

        all_boxes = [b for _, b, _ in boxes_this_frame]
        for tid, box, conf in boxes_this_frame:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame[y1:y2, x1:x2].copy()
            others = [b for b in all_boxes if b != box]
            observations[tid].append(
                {"frame_idx": idx, "bbox": box, "conf": conf, "crop": crop, "others": others}
            )
        idx += 1

    cap.release()
    return observations, frame_w, frame_h


def pick_baselines(obs_list: list[dict], frame_w: int, frame_h: int):
    # Baseline: Random
    random_pick = random.choice(obs_list)

    # 대안 A: Highest Confidence
    conf_pick = max(obs_list, key=lambda o: o["conf"])

    # 대안 B: BestShot Score
    scored = []
    for o in obs_list:
        s = bestshot_score(o["conf"], o["crop"], o["bbox"], o["others"], frame_w, frame_h)
        scored.append((s["total"], s, o))
    scored.sort(key=lambda x: -x[0])
    best_total, best_scores, best_pick = scored[0]

    return random_pick, conf_pick, (best_pick, best_scores)


def main():
    out_dir = ROOT / "results/EXP-004"
    grid_dir = out_dir / "bestshot_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"

    print("Collecting track observations (this scans the video with tracking)...")
    observations, fw, fh = collect_track_observations(model, video_path, max_frames=300)
    print(f"Frame size: {fw}x{fh}, tracks found: {len(observations)}")

    # 관측치가 충분한(>=8 프레임) Track만 비교 대상으로 사용
    candidate_tracks = {tid: obs for tid, obs in observations.items() if len(obs) >= 8}
    print(f"Tracks with >=8 observations: {len(candidate_tracks)}")

    rows = []
    for tid, obs_list in candidate_tracks.items():
        random_pick, conf_pick, (best_pick, best_scores) = pick_baselines(obs_list, fw, fh)

        # 3장을 가로로 이어붙인 비교 grid 저장 (Baseline | Conf-only | BestShot Score)
        def resize_h(img, target_h=200):
            if img.size == 0:
                return np.zeros((target_h, 100, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (max(1, int(w * scale)), target_h))

        imgs = [resize_h(random_pick["crop"]), resize_h(conf_pick["crop"]), resize_h(best_pick["crop"])]
        labels = ["Baseline(Random)", "A(Confidence)", "B(BestShot Score)"]
        labeled = []
        for im, lbl in zip(imgs, labels):
            canvas = np.full((im.shape[0] + 20, im.shape[1], 3), 255, dtype=np.uint8)
            canvas[20:, :, :] = im
            cv2.putText(canvas, lbl, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            labeled.append(canvas)
        max_h = max(im.shape[0] for im in labeled)
        padded = []
        for im in labeled:
            if im.shape[0] < max_h:
                pad = np.full((max_h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
                im = np.vstack([im, pad])
            padded.append(im)
        grid = np.hstack(padded)
        cv2.imwrite(str(grid_dir / f"track_{tid}.jpg"), grid)

        rows.append(
            {
                "track_id": tid,
                "num_observations": len(obs_list),
                "baseline_frame": random_pick["frame_idx"],
                "confidence_only_frame": conf_pick["frame_idx"],
                "confidence_only_conf": round(conf_pick["conf"], 3),
                "bestshot_frame": best_pick["frame_idx"],
                "bestshot_total": round(best_scores["total"], 3),
                "bestshot_conf": round(best_scores["confidence"], 3),
                "bestshot_sharpness": round(best_scores["sharpness"], 3),
                "bestshot_size": round(best_scores["size"], 3),
                "bestshot_occlusion": round(best_scores["occlusion"], 3),
                "bestshot_position": round(best_scores["position"], 3),
            }
        )

    import csv

    if rows:
        with open(out_dir / "bestshot_selection.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSaved: {out_dir}/bestshot_selection.csv, grids in {grid_dir}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
