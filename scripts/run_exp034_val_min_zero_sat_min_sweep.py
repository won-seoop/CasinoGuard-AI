"""EXP-034: val_min=0 직접 검증 + sat_min Sweep (지침 19, FC-012/EXP-033 후속,
Roadmap 다음 Action 6번 - 최우선).

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021), EXP-031(PAR-022), EXP-032(PAR-023), EXP-033(PAR-024)

## 배경
EXP-033은 region_dominant_color의 그림자 하한(val_min)을 35->10으로 완화한 대안 A를
production(c_wb) 후보로 채택했지만, "production에서 실제로 쓰는 Mask+WB 경로
(c_wb_relaxed)로 val_min을 더 낮춰도 괜찮은가"는 직접 측정하지 않았다 - EXP-033의
val_min Sweep(섹션 5)은 사후 분석으로 Mask/WB 없이 region_dominant_color만 단독
(method "b")으로 돌린 근사치였다. 이 실험은 그 근사가 실제 c_wb_relaxed 전체
파이프라인에서도 유지되는지 직접 검증하고, 지금까지 전혀 건드리지 않은 sat_min
(채도 하한)도 같은 방식으로 Sweep해 "그림자 제거 필터" 전체의 가치를 재평가한다.

## Hypothesis
1. EXP-033 섹션 5의 단조 추세(val_min을 낮출수록 Accuracy가 계속 오름)가 Mask+WB를
   포함한 실제 c_wb_relaxed 파이프라인에서도 유지되고, val_min=0이 Sweep 범위 내
   최고일 것이다(method "b" 근사와 동일한 메커니즘이므로).
2. sat_min도 val_min과 같은 메커니즘(검정 옷이 필터에 걸려 사라짐)으로 낮출수록
   Accuracy가 개선될 것이다.

## 측정
val_min in {35(baseline), 30, 25, 20, 15, 10(EXP-033 대안 A), 5, 0}을 c_wb_relaxed
로 직접 측정(Mask+WB 포함) -> 최적 val_min 확정.
그 val_min을 고정한 뒤 sat_min in {25(baseline), 20, 15, 10, 5, 0}을 Sweep해
val_min과 독립적인 sat_min의 기여를 분리.
Overall/GT=black/비검정 GT(Accuracy가 실제 색상 옷에도 부작용 없는지) 모두 측정.

Dataset: EXP-027/028/030/031/032/033과 완전히 동일한 coco128 n=21 GT(EXP-032/033의
GT 라벨 교정 2건 반영) + 동일 5종 합성 조명.

실행: python scripts/run_exp034_val_min_zero_sat_min_sweep.py
출력: results/EXP-034/{predictions.csv, summary.json}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attributes.color import classify_person_attributes_with_mask_and_wb_relaxed  # noqa: E402

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-034")
MIN_BOX_AREA = 120 * 250
SEG_IOU_MATCH_THRESHOLD = 0.5

LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

VAL_MIN_SWEEP = [35, 30, 25, 20, 15, 10, 5, 0]
SAT_MIN_SWEEP = [25, 20, 15, 10, 5, 0]

# EXP-027/028/030/031/032/033과 완전히 동일한 Ground Truth(EXP-032/033의 GT 교정 반영).
GROUND_TRUTH = {
    "000000000086_0": {"upper": "white", "lower": "black"},
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
    "000000000634_0": {"upper": "gray", "lower": "black"},
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
    """EXP-028/032/033과 동일한 Detector+Segmenter IoU 매칭(동일 GT 이름 재현용)."""
    detector = YOLO("yolo11n.pt")
    segmenter = YOLO("yolo11n-seg.pt")
    crops: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

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
            raw = seg_res.masks.data.cpu().numpy()
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
            masks[name] = local_mask
    return crops, masks


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027/028/032/033과 동일한 합성 조명 변형."""
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


def _score(rows: list[dict], val_min: int, sat_min: int, crops, masks, lit_cache) -> dict:
    """주어진 (val_min, sat_min)으로 c_wb_relaxed를 전수 측정해 row별 결과를 채운다."""
    correct, total = 0, 0
    black_correct, black_total = 0, 0
    nonblack_correct, nonblack_total = 0, 0
    by_condition: dict[str, dict] = {c: {"correct": 0, "total": 0, "black_correct": 0, "black_total": 0} for c in LIGHTING_CONDITIONS}

    for name, gt in GROUND_TRUTH.items():
        mask = masks[name]
        for condition in LIGHTING_CONDITIONS:
            lit = lit_cache[(name, condition)]
            pred = classify_person_attributes_with_mask_and_wb_relaxed(lit, mask, val_min=val_min, sat_min=sat_min)
            for region in ("upper", "lower"):
                gt_color = gt[region]
                is_correct = int(pred[region] == gt_color)
                total += 1
                correct += is_correct
                by_condition[condition]["total"] += 1
                by_condition[condition]["correct"] += is_correct
                if gt_color == "black":
                    black_total += 1
                    black_correct += is_correct
                    by_condition[condition]["black_total"] += 1
                    by_condition[condition]["black_correct"] += is_correct
                else:
                    nonblack_total += 1
                    nonblack_correct += is_correct

    return {
        "val_min": val_min,
        "sat_min": sat_min,
        "overall_accuracy": round(correct / total, 4),
        "gt_black_accuracy": round(black_correct / black_total, 4) if black_total else None,
        "nonblack_accuracy": round(nonblack_correct / nonblack_total, 4) if nonblack_total else None,
        "accuracy_by_condition": {
            c: {
                "overall_accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None,
                "gt_black_accuracy": round(v["black_correct"] / v["black_total"], 4) if v["black_total"] else None,
            }
            for c, v in by_condition.items()
        },
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crops, masks = detect_crops_and_masks()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")
    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (detector mismatch?): {missing}")

    # 조명 적용은 (val_min, sat_min) Sweep 전체에서 동일하므로 한 번만 캐시한다.
    lit_cache = {
        (name, condition): apply_lighting(crops[name], condition)
        for name in GROUND_TRUTH
        for condition in LIGHTING_CONDITIONS
    }

    # 1) val_min Sweep (sat_min=25 고정, EXP-033 baseline과 동일 조건)
    val_min_results = [_score(None, v, 25, crops, masks, lit_cache) for v in VAL_MIN_SWEEP]
    best_val_min_entry = max(val_min_results, key=lambda r: r["overall_accuracy"])
    best_val_min = best_val_min_entry["val_min"]

    # 2) sat_min Sweep (최적 val_min 고정)
    sat_min_results = [_score(None, best_val_min, s, crops, masks, lit_cache) for s in SAT_MIN_SWEEP]

    summary = {
        "n_ground_truth_crops": len(GROUND_TRUTH),
        "n_cells_total": len(GROUND_TRUTH) * len(LIGHTING_CONDITIONS) * 2,
        "val_min_sweep_sat_min_fixed_at": 25,
        "val_min_sweep": val_min_results,
        "best_val_min_by_overall_accuracy": best_val_min,
        "sat_min_sweep_val_min_fixed_at": best_val_min,
        "sat_min_sweep": sat_min_results,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # predictions.csv: val_min Sweep 끝(0)과 sat_min Sweep 끝(0)의 실제 cell 단위
    # 예측값도 남겨 둔다(회귀/추가 분석용).
    rows = []
    for name, gt in GROUND_TRUTH.items():
        mask = masks[name]
        for condition in LIGHTING_CONDITIONS:
            lit = lit_cache[(name, condition)]
            pred_baseline = classify_person_attributes_with_mask_and_wb_relaxed(lit, mask, val_min=35, sat_min=25)
            pred_val0 = classify_person_attributes_with_mask_and_wb_relaxed(lit, mask, val_min=0, sat_min=25)
            pred_best = classify_person_attributes_with_mask_and_wb_relaxed(
                lit, mask, val_min=best_val_min, sat_min=SAT_MIN_SWEEP[-1]
            )
            for region in ("upper", "lower"):
                gt_color = gt[region]
                rows.append(
                    {
                        "name": name,
                        "region": region,
                        "condition": condition,
                        "gt_color": gt_color,
                        "is_gt_black": int(gt_color == "black"),
                        "pred_val_min_35_sat_min_25": pred_baseline[region],
                        "correct_val_min_35_sat_min_25": int(pred_baseline[region] == gt_color),
                        "pred_val_min_0_sat_min_25": pred_val0[region],
                        "correct_val_min_0_sat_min_25": int(pred_val0[region] == gt_color),
                        f"pred_val_min_{best_val_min}_sat_min_{SAT_MIN_SWEEP[-1]}": pred_best[region],
                        f"correct_val_min_{best_val_min}_sat_min_{SAT_MIN_SWEEP[-1]}": int(
                            pred_best[region] == gt_color
                        ),
                    }
                )
    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "val_min_sweep": [
                    {"val_min": r["val_min"], "overall": r["overall_accuracy"], "gt_black": r["gt_black_accuracy"]}
                    for r in val_min_results
                ],
                "best_val_min": best_val_min,
                "sat_min_sweep": [
                    {"sat_min": r["sat_min"], "overall": r["overall_accuracy"], "gt_black": r["gt_black_accuracy"]}
                    for r in sat_min_results
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
