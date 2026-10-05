"""EXP-028: Attribute Metadata 색상 분류 - Segmentation Mask 기반 영역 분리 (대안 C, 지침 19).

Roadmap(01. Architecture & Roadmap) 다음 Action: "FC-012 다음 후보는 Segmentation Mask 등
더 근본적인 영역 분리 방법 검토(단순 HSV 필터링은 n=21 결과로 한계가 드러남)". EXP-026/027은
FC-012(검정 옷의 strong_light/cool_cast 오분류)를 겨냥한 색 보정(White Balance)/확신도
신호를 검토했지만 둘 다 기각됐다. 이 실험은 FC-012가 아니라 method b가 이미 가진 "고정 비율
사각형으로 상/하 영역을 자르면 배경·소수 피부색 Bleed가 항상 섞인다"는 더 근본적인 한계를
겨냥한다 - person Instance Segmentation Mask로 배경을 픽셀 단위로 제외하면 Hue 다수결이
덜 오염될 것이라는 가설을 EXP-027과 동일한 n=21 Ground Truth로 검증한다.

방법: yolo11n.pt(기존 EXP-027과 동일 Detector, 동일 conf=0.4)로 person bbox를 검출해
crop 이름(GT key)을 그대로 재현하고, yolo11n-seg.pt(Instance Segmentation)로 같은 이미지의
Segmentation Mask를 얻어 IoU로 Detector bbox와 매칭한다. 매칭된 Mask를 crop 좌표로 잘라
classify_person_attributes_with_mask()(대안 C)에 전달하고, method baseline/a/b/b_wb와
동일 Ground Truth·동일 5종 합성 조명으로 Accuracy를 비교한다. 추가로 대안 C의 결과가
FC-012의 표적 실패(cool_cast 흑백->blue)에서는 b_wb보다 못할 가능성을 바로 확인하기 위해,
Mask(대안 C)와 White Balance(EXP-026 대안 B)를 함께 적용하는 c_wb도 같이 측정한다.

실행: python scripts/run_exp028_attribute_segmentation_mask.py
출력: results/EXP-028/{masks/, predictions.csv, summary.json}
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attributes.color import (  # noqa: E402
    classify_person_attributes,
    classify_person_attributes_with_mask,
    classify_person_attributes_with_mask_and_wb,
)

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-028")
MASKS_DIR = RESULTS_DIR / "masks"
MIN_BOX_AREA = 120 * 250
SEG_IOU_MATCH_THRESHOLD = 0.5

METHODS = ["baseline", "a", "b", "b_wb", "c", "c_wb"]
LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

# EXP-027과 동일한 Ground Truth (동일 Detector/conf/coco128 -> 동일 crop 이름 재현 확인됨).
GROUND_TRUTH = {
    "000000000086_0": {"upper": "black", "lower": "black"},
    "000000000113_0": {"upper": "white", "lower": "black"},
    "000000000165_0": {"upper": "gray", "lower": "gray"},
    "000000000165_1": {"upper": "black", "lower": "black"},
    "000000000328_0": {"upper": "black", "lower": "black"},
    "000000000328_1": {"upper": "black", "lower": "black"},
    "000000000328_2": {"upper": "black", "lower": "black"},
    "000000000368_0": {"upper": "red", "lower": "black"},
    "000000000370_0": {"upper": "pink", "lower": "pink"},
    "000000000389_0": {"upper": "white", "lower": "white"},
    "000000000395_0": {"upper": "orange", "lower": "orange"},
    "000000000415_0": {"upper": "blue", "lower": "white"},
    "000000000431_0": {"upper": "white", "lower": "white"},
    "000000000446_0": {"upper": "gray", "lower": "black"},
    "000000000459_0": {"upper": "black", "lower": "black"},
    "000000000536_0": {"upper": "black", "lower": "black"},
    "000000000564_0": {"upper": "white", "lower": "white"},
    "000000000572_0": {"upper": "black", "lower": "black"},
    "000000000572_1": {"upper": "black", "lower": "blue"},
    "000000000623_0": {"upper": "gray", "lower": "pink"},
    "000000000634_0": {"upper": "black", "lower": "black"},
}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def detect_crops_and_masks() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """GT 이름별로 (crop, local mask)를 만든다. local mask는 crop과 동일 HxW의 bool
    배열이며, 매칭되는 Segmentation Instance가 없으면(IoU<threshold) 전부 False
    (classify_person_attributes_with_mask가 알아서 method b 폴백으로 처리한다)."""
    detector = YOLO("yolo11n.pt")
    segmenter = YOLO("yolo11n-seg.pt")
    crops: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    unmatched: list[str] = []

    image_paths = sorted(COCO128_IMAGES.glob("*.jpg"))
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        det_res = detector.predict(img, classes=[0], conf=0.4, verbose=False)[0]
        if len(det_res.boxes) == 0:
            continue
        seg_res = segmenter.predict(img, classes=[0], conf=0.4, verbose=False)[0]
        seg_boxes = seg_res.boxes.xyxy.cpu().numpy() if len(seg_res.boxes) else np.zeros((0, 4))
        seg_masks_full = None
        if seg_res.masks is not None and len(seg_res.masks.data) > 0:
            raw = seg_res.masks.data.cpu().numpy()  # (n, mh, mw) at model mask resolution
            seg_masks_full = np.stack(
                [cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) for m in raw]
            ).astype(bool)

        stem = img_path.stem
        for i, box in enumerate(det_res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_BOX_AREA:
                continue
            name = f"{stem}_{i}"
            crops[name] = img[y1:y2, x1:x2]

            local_mask = np.zeros((h, w), dtype=bool)
            if seg_masks_full is not None and len(seg_boxes) > 0:
                ious = [_box_iou((x1, y1, x2, y2), sb) for sb in seg_boxes]
                best_idx = int(np.argmax(ious))
                if ious[best_idx] >= SEG_IOU_MATCH_THRESHOLD:
                    local_mask = seg_masks_full[best_idx][y1:y2, x1:x2]
                elif name in GROUND_TRUTH:
                    unmatched.append(name)
            masks[name] = local_mask
    if unmatched:
        print(f"WARNING: {len(unmatched)} GT crop(s) had no matching segmentation instance "
              f"(IoU<{SEG_IOU_MATCH_THRESHOLD}), method c will fall back to method b for them: {unmatched}")
    return crops, masks


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027과 동일한 합성 조명 변형(Before/After 비교가 유효하려면 동일해야 함)."""
    img = img_bgr.astype(np.float32)
    if condition == "normal":
        out = img
    elif condition == "low_light":
        out = img * 0.35
    elif condition == "strong_light":
        out = img * 1.9 + 25
    elif condition == "warm_cast":
        out = img.copy()
        out[:, :, 2] *= 1.35
        out[:, :, 0] *= 0.75
    elif condition == "cool_cast":
        out = img.copy()
        out[:, :, 0] *= 1.35
        out[:, :, 2] *= 0.75
    else:
        raise ValueError(condition)
    return np.clip(out, 0, 255).astype(np.uint8)


def predict(method: str, lit_crop: np.ndarray, mask: np.ndarray) -> dict[str, str]:
    if method == "c":
        return classify_person_attributes_with_mask(lit_crop, mask)
    if method == "c_wb":
        return classify_person_attributes_with_mask_and_wb(lit_crop, mask)
    return classify_person_attributes(lit_crop, method=method)


def measure_inference_overhead(n_images: int = 20) -> dict:
    """대안 C는 Detector 외에 Segmentation 모델을 추가로 돌려야 한다 - Edge 환경(지침 24)
    에서 이 추가 비용이 얼마인지 같은 이미지 n장으로 직접 측정해 기록한다."""
    detector = YOLO("yolo11n.pt")
    segmenter = YOLO("yolo11n-seg.pt")
    image_paths = sorted(COCO128_IMAGES.glob("*.jpg"))[:n_images]
    imgs = [cv2.imread(str(p)) for p in image_paths]

    for img in imgs[:3]:
        detector.predict(img, classes=[0], conf=0.4, verbose=False)
        segmenter.predict(img, classes=[0], conf=0.4, verbose=False)

    t0 = time.perf_counter()
    for img in imgs:
        detector.predict(img, classes=[0], conf=0.4, verbose=False)
    det_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for img in imgs:
        segmenter.predict(img, classes=[0], conf=0.4, verbose=False)
    seg_elapsed = time.perf_counter() - t0

    det_pt = Path("yolo11n.pt")
    seg_pt = Path("yolo11n-seg.pt")
    return {
        "n_images": len(imgs),
        "detector_ms_per_image": round(det_elapsed / len(imgs) * 1000, 2),
        "segmenter_ms_per_image": round(seg_elapsed / len(imgs) * 1000, 2),
        "segmenter_overhead_ratio": round(seg_elapsed / det_elapsed, 2),
        "detector_model_size_mb": round(det_pt.stat().st_size / 1e6, 2) if det_pt.exists() else None,
        "segmenter_model_size_mb": round(seg_pt.stat().st_size / 1e6, 2) if seg_pt.exists() else None,
    }


def main() -> None:
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    crops, masks = detect_crops_and_masks()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")

    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (detector mismatch?): {missing}")

    n_mask_matched = sum(1 for n in GROUND_TRUTH if masks[n].sum() > 0)
    print(f"segmentation mask matched for {n_mask_matched}/{len(GROUND_TRUTH)} GT crops")

    for name in GROUND_TRUTH:
        mask_vis = (masks[name].astype(np.uint8) * 255)
        cv2.imwrite(str(MASKS_DIR / f"{name}_mask.png"), mask_vis)

    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        mask = masks[name]
        for condition in LIGHTING_CONDITIONS:
            lit = apply_lighting(crop, condition)
            for method in METHODS:
                pred = predict(method, lit, mask)
                rows.append(
                    {
                        "name": name,
                        "condition": condition,
                        "method": method,
                        "mask_matched": int(mask.sum() > 0),
                        "gt_upper": gt["upper"],
                        "gt_lower": gt["lower"],
                        "pred_upper": pred["upper"],
                        "pred_lower": pred["lower"],
                        "upper_correct": int(pred["upper"] == gt["upper"]),
                        "lower_correct": int(pred["lower"] == gt["lower"]),
                    }
                )

    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {
        "n_ground_truth_crops": len(GROUND_TRUTH),
        "n_segmentation_mask_matched": n_mask_matched,
        "methods": {},
    }
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        total = len(method_rows) * 2
        correct = sum(r["upper_correct"] + r["lower_correct"] for r in method_rows)

        normal_rows = [r for r in method_rows if r["condition"] == "normal"]
        normal_total = len(normal_rows) * 2
        normal_correct = sum(r["upper_correct"] + r["lower_correct"] for r in normal_rows)

        black_rows = [r for r in method_rows if r["gt_upper"] == "black" or r["gt_lower"] == "black"]
        black_cells = []
        for r in black_rows:
            if r["gt_upper"] == "black":
                black_cells.append(r["upper_correct"])
            if r["gt_lower"] == "black":
                black_cells.append(r["lower_correct"])
        black_total = len(black_cells)
        black_correct = sum(black_cells)

        # FC-012가 원래 겨냥한 실패 모드: GT=black 셀이 strong_light/cool_cast에서
        # "blue"로 오분류되는 비율 (EXP-026/027과 동일 분석 패턴).
        black_to_blue_by_condition = {}
        for condition in ("strong_light", "cool_cast"):
            cond_black_cells = []
            for r in black_rows:
                if r["condition"] != condition:
                    continue
                if r["gt_upper"] == "black":
                    cond_black_cells.append(r["pred_upper"])
                if r["gt_lower"] == "black":
                    cond_black_cells.append(r["pred_lower"])
            if cond_black_cells:
                black_to_blue_by_condition[condition] = round(
                    sum(1 for p in cond_black_cells if p == "blue") / len(cond_black_cells), 4
                )

        summary["methods"][method] = {
            "overall_accuracy": round(correct / total, 4),
            "overall_correct": correct,
            "overall_total": total,
            "normal_lighting_accuracy": round(normal_correct / normal_total, 4),
            "gt_black_accuracy": round(black_correct / black_total, 4) if black_total else None,
            "gt_black_total_cells": black_total,
            "gt_black_misclassified_as_blue_rate_by_condition": black_to_blue_by_condition,
        }
        by_condition = {}
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in method_rows if r["condition"] == condition]
            cond_total = len(cond_rows) * 2
            cond_correct = sum(r["upper_correct"] + r["lower_correct"] for r in cond_rows)
            by_condition[condition] = round(cond_correct / cond_total, 4)
        summary["methods"][method]["accuracy_by_condition"] = by_condition

    # method c/c_wb: 마스크가 실제로 매칭된 crop(mask_matched=1)만 슬라이싱한 Accuracy.
    # 폴백(매칭 실패 -> method b와 동일 로직)이 섞이면 "마스크 자체의 효과"를 가릴 수 있다.
    for mask_method in ("c", "c_wb"):
        matched_rows = [r for r in rows if r["method"] == mask_method and r["mask_matched"] == 1]
        if matched_rows:
            m_total = len(matched_rows) * 2
            m_correct = sum(r["upper_correct"] + r["lower_correct"] for r in matched_rows)
            summary["methods"][mask_method]["mask_matched_only_accuracy"] = round(m_correct / m_total, 4)
            summary["methods"][mask_method]["mask_matched_only_total"] = m_total

    print("measuring detector-only vs detector+segmenter inference overhead...")
    summary["inference_overhead"] = measure_inference_overhead()

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
