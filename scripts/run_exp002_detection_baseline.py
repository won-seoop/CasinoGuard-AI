"""EXP-002: Person Detection Baseline (YOLO11n).

1) coco128(person class=0)로 confidence threshold sweep -> Precision/Recall/F1
2) 실제 영상(pedestrian_crossing, crowded_intersection)에서 FPS/Latency 측정
3) 결과를 results/EXP-002/ 에 저장
"""

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from detection.eval_utils import match_image, yolo_to_xyxy  # noqa: E402

COCO_DIR = ROOT / "data/raw/coco128"
IMG_DIR = COCO_DIR / "images/train2017"
LBL_DIR = COCO_DIR / "labels/train2017"
PERSON_CLASS_ID = 0

CONF_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
LOW_CONF_FOR_COLLECT = 0.05  # sweep을 위해 낮은 conf로 한 번만 추론
IMG_SIZE_SWEEP = [640, 960, 1280]


def load_gt_person_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        if int(cls) != PERSON_CLASS_ID:
            continue
        boxes.append(yolo_to_xyxy(float(cx), float(cy), float(w), float(h), img_w, img_h))
    return boxes


def collect_predictions(model: YOLO, image_paths: list[Path]):
    """전체 이미지에 대해 낮은 conf로 한 번만 추론해서 (박스, 점수, GT) 캐시."""
    cache = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        result = model.predict(img, conf=LOW_CONF_FOR_COLLECT, classes=[PERSON_CLASS_ID], imgsz=640, verbose=False)[0]
        pred_boxes = [tuple(b) for b in result.boxes.xyxy.tolist()] if len(result.boxes) else []
        pred_scores = result.boxes.conf.tolist() if len(result.boxes) else []
        gt_boxes = load_gt_person_boxes(LBL_DIR / (img_path.stem + ".txt"), w, h)
        cache.append({"path": img_path.name, "gt": gt_boxes, "pred": pred_boxes, "score": pred_scores})
    return cache


def evaluate_at_threshold(cache, threshold: float):
    from detection.eval_utils import Counts

    total = Counts()
    for item in cache:
        keep_idx = [i for i, s in enumerate(item["score"]) if s >= threshold]
        pred = [item["pred"][i] for i in keep_idx]
        score = [item["score"][i] for i in keep_idx]
        c = match_image(item["gt"], pred, score, iou_threshold=0.5)
        total.tp += c.tp
        total.fp += c.fp
        total.fn += c.fn
    return total


def measure_video_fps(model: YOLO, video_path: Path, imgsz: int, conf: float, max_frames: int = 150, warmup: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    # Warm-up: 모델/백엔드 초기화(JIT, 메모리 할당 등) 비용이 첫 추론에 몰려서
    # 측정을 오염시키는 것을 막기 위해 워밍업 프레임은 타이밍에서 제외한다.
    # (EXP-002 1차 실행에서 실제로 관찰된 문제 — Analysis 참고)
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)

    latencies = []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)
        latencies.append(time.perf_counter() - t0)
        n += 1
    cap.release()
    if not latencies:
        return None
    import numpy as np

    arr = np.array(latencies) * 1000
    return {
        "frames": n,
        "fps": n / sum(latencies),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-coco", action="store_true", help="coco128 confidence sweep을 다시 돌리지 않음")
    args = parser.parse_args()

    out_dir = ROOT / "results/EXP-002"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")

    if not args.skip_coco:
        image_paths = sorted(IMG_DIR.glob("*.jpg"))
        print(f"coco128 images: {len(image_paths)}")

        t0 = time.perf_counter()
        cache = collect_predictions(model, image_paths)
        collect_time = time.perf_counter() - t0
        print(f"collected predictions for {len(cache)} images in {collect_time:.1f}s")

        n_with_person = sum(1 for c in cache if c["gt"])
        print(f"images with >=1 person GT: {n_with_person} / {len(cache)}")

        threshold_rows = []
        for th in CONF_SWEEP:
            counts = evaluate_at_threshold(cache, th)
            row = {
                "confidence_threshold": th,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "precision": round(counts.precision, 4),
                "recall": round(counts.recall, 4),
                "f1": round(counts.f1, 4),
            }
            threshold_rows.append(row)
            print(row)

        with open(out_dir / "confidence_sweep.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
            writer.writeheader()
            writer.writerows(threshold_rows)

    # 실제 영상 FPS/Latency 측정 (baseline conf=0.4, imgsz 스윕, warm-up 5프레임 제외)
    video_rows = []
    videos = {
        "normal_480p": ROOT / "data/raw/pedestrian_crossing_480p.ogg",
        "crowded_1080p": ROOT / "data/raw/crowded_intersection_1080p.webm",
    }
    for vname, vpath in videos.items():
        for imgsz in IMG_SIZE_SWEEP:
            res = measure_video_fps(model, vpath, imgsz=imgsz, conf=0.4, max_frames=100)
            if res is None:
                continue
            row = {"video": vname, "imgsz": imgsz, **res}
            video_rows.append(row)
            print(row)

    with open(out_dir / "video_latency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(video_rows)

    print(f"\nSaved: {out_dir}/confidence_sweep.csv, {out_dir}/video_latency.csv")


if __name__ == "__main__":
    main()
