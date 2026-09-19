"""EXP-009: Edge Optimization — PyTorch FP32 vs ONNX FP32 vs ONNX INT8(Dynamic).

지침 24번: MacBook Air M3 환경이므로 CUDA/TensorRT는 필수로 하지 않는다.
Accuracy(coco128 Precision/Recall/F1) vs Latency(FPS) vs Model Size Trade-off를 측정한다.
"""

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from detection.eval_utils import match_image, yolo_to_xyxy, Counts  # noqa: E402

PERSON_CLASS_ID = 0
COCO_DIR = ROOT / "data/raw/coco128"
IMG_DIR = COCO_DIR / "images/train2017"
LBL_DIR = COCO_DIR / "labels/train2017"

MODELS = {
    "pytorch_fp32": ROOT / "yolo11n.pt",
    "onnx_fp32": ROOT / "models/yolo11n.onnx",
    "onnx_int8_dynamic": ROOT / "models/yolo11n_int8.onnx",
}


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


def eval_accuracy(model: YOLO, image_paths: list[Path], conf: float = 0.4):
    total = Counts()
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        result = model.predict(img, conf=conf, classes=[PERSON_CLASS_ID], imgsz=640, verbose=False)[0]
        pred_boxes = [tuple(b) for b in result.boxes.xyxy.tolist()] if len(result.boxes) else []
        pred_scores = result.boxes.conf.tolist() if len(result.boxes) else []
        gt_boxes = load_gt_person_boxes(LBL_DIR / (img_path.stem + ".txt"), w, h)
        c = match_image(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5)
        total.tp += c.tp
        total.fp += c.fp
        total.fn += c.fn
    return total


def measure_fps(model: YOLO, video_path: Path, conf: float = 0.4, warmup: int = 5, max_frames: int = 100):
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], imgsz=640, verbose=False)

    latencies = []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], imgsz=640, verbose=False)
        latencies.append(time.perf_counter() - t0)
        n += 1
    cap.release()
    if not latencies:
        return None
    arr = np.array(latencies) * 1000
    return {
        "frames": n,
        "fps": n / sum(latencies),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def main():
    out_dir = ROOT / "results/EXP-009"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMG_DIR.glob("*.jpg"))
    video_path = ROOT / "data/raw/crowded_intersection_1080p.webm"

    rows = []
    for name, path in MODELS.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        print(f"\n=== {name} ({path.name}) ===")
        model = YOLO(str(path))
        size_mb = path.stat().st_size / 1e6

        acc = eval_accuracy(model, image_paths, conf=0.4)
        perf = measure_fps(model, video_path, conf=0.4)

        row = {
            "model": name,
            "size_mb": round(size_mb, 2),
            "precision": round(acc.precision, 4),
            "recall": round(acc.recall, 4),
            "f1": round(acc.f1, 4),
            "tp": acc.tp,
            "fp": acc.fp,
            "fn": acc.fn,
            "fps": round(perf["fps"], 2) if perf else None,
            "p50_ms": round(perf["p50_ms"], 2) if perf else None,
            "p95_ms": round(perf["p95_ms"], 2) if perf else None,
        }
        rows.append(row)
        print(row)

    with open(out_dir / "edge_optimization_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_dir}/edge_optimization_comparison.csv")


if __name__ == "__main__":
    main()
