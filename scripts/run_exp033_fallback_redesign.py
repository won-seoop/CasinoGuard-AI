"""EXP-033: min_filtered_fraction 폴백 경로 재설계 (지침 19, FC-012/EXP-032 후속).

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021), EXP-031(PAR-022), EXP-032(PAR-023)

## 배경
EXP-032가 GT=black 110셀 오분류를 직접 진단해 "조명 보정으로는 고칠 수 없는 범위"
(normal 조명에서도 40%+ 오분류)와 "고칠 수 있는 범위" 하나를 구체적으로 특정했다 -
오분류의 35~40%가 `min_filtered_fraction` 폴백 경로(필터 통과 픽셀 <30% -> 필터를
완전히 버리고 원본 전체 픽셀로 복귀, skin exclusion까지 함께 사라짐)에서 발생한다.
EXP-032 Next Action 1번: "이분법적 폴백 대신 완화된 필터 기준 또는 부분 폴백 전략을
Alt A/B로 비교할 것."

## 대안
- 대안 A(b_relaxed/c_wb_relaxed): 가장 단순한 수정 - region_dominant_color의 그림자
  제거용 val_min을 35->10으로 그냥 낮춘다. 폴백 설계(이분법)는 그대로 둔다.
- 대안 B(b_graded/c_wb_graded, 채택 여부 검증 대상): min_filtered_fraction 폴백
  자체를 val_min 단계적 완화(35->15->0)로 교체한다(region_dominant_color_graded_
  fallback). sat_min/val_max/skin exclusion은 모든 단계에서 유지된다.

둘 다 EXP-025~032와 동일하게 method "b"(실험용, region_dominant_color 직접 사용)와
"c_wb"(production이 실제로 run_full_pipeline.py에 연결한 방식, PAR-020)에 각각
적용해 비교한다.

## 측정
1. Overall Accuracy(GT 색상 전체, EXP-027 방식과 동일 - 21 crop x 42 cell x 5 조명)
2. GT=black Accuracy만 슬라이싱(FC-012가 겨냥한 부분집합, EXP-026/027/032와 동일 정의)
3. 폴백(또는 완화 단계) 발동 빈도 - 대안이 실제로 폴백 경로를 얼마나 줄이는지 직접 확인

Dataset: EXP-027/028/030/031/032와 완전히 동일한 coco128 n=21 Ground Truth(EXP-032의
GT 라벨 교정 2건 반영) + 동일 5종 합성 조명.

실행: python scripts/run_exp033_fallback_redesign.py
출력: results/EXP-033/{predictions.csv, summary.json}
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
from src.attributes.color import (  # noqa: E402
    classify_person_attributes,
    classify_person_attributes_with_mask_and_wb,
    classify_person_attributes_with_mask_and_wb_graded,
    classify_person_attributes_with_mask_and_wb_relaxed,
    region_dominant_color_diagnostic,
    region_dominant_color_graded_fallback_diagnostic,
    split_upper_lower,
)

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-033")
MIN_BOX_AREA = 120 * 250
SEG_IOU_MATCH_THRESHOLD = 0.5

LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

# method "b" 변형 3종(region_dominant_color 직접 사용, 실험용) + method "c_wb" 변형
# 3종(production이 실제로 쓰는 Mask+WB 경로). "b"/"c_wb"는 EXP-025~032와 완전히
# 동일한 기존(이분법) 함수 - 이 실험의 Baseline이다.
METHODS = ["b", "b_relaxed", "b_graded"]
MASK_METHODS = {
    "c_wb": classify_person_attributes_with_mask_and_wb,
    "c_wb_relaxed": classify_person_attributes_with_mask_and_wb_relaxed,
    "c_wb_graded": classify_person_attributes_with_mask_and_wb_graded,
}

# EXP-027/028/030/031/032와 완전히 동일한 Ground Truth.
GROUND_TRUTH = {
    # EXP-032가 발견/교정한 GT 라벨 오류 2건을 반영한다(지침 31, 교정 후 재측정).
    # 원래 주석은 "어두운 자켓"이었지만 B&W 사진 속 실제 상의는 밝은 크림색 셔츠.
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
    # 그레이스케일 스케이트보더 사진 - 실제 상의는 연회색 티셔츠(하의만 검정 반바지).
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
    """EXP-028/032와 동일한 Detector+Segmenter IoU 매칭(동일 GT 이름 재현용)."""
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
    """EXP-025/027/028/032와 동일한 합성 조명 변형."""
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


def region_fallback_flag(lit_crop: np.ndarray, region: str, graded: bool) -> bool:
    """method b 계열의 진단 함수로 이 region이 폴백(또는 완화 단계)을 썼는지 확인한다."""
    upper, lower = split_upper_lower(
        lit_crop, head_skip_ratio=0.22, upper_end_ratio=0.55, foot_skip_ratio=0.08, side_margin_ratio=0.12
    )
    target = upper if region == "upper" else lower
    diag_fn = region_dominant_color_graded_fallback_diagnostic if graded else region_dominant_color_diagnostic
    return bool(diag_fn(target)["used_fallback"])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crops, masks = detect_crops_and_masks()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")
    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (detector mismatch?): {missing}")

    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        mask = masks[name]
        for condition in LIGHTING_CONDITIONS:
            lit = apply_lighting(crop, condition)

            preds: dict[str, dict[str, str]] = {}
            for method in METHODS:
                preds[method] = classify_person_attributes(lit, method=method)
            for method, fn in MASK_METHODS.items():
                preds[method] = fn(lit, mask)

            for region in ("upper", "lower"):
                gt_color = gt[region]
                row = {
                    "name": name,
                    "region": region,
                    "condition": condition,
                    "gt_color": gt_color,
                    "is_gt_black": int(gt_color == "black"),
                    "b_used_fallback": int(region_fallback_flag(lit, region, graded=False)),
                    "b_graded_used_fallback": int(region_fallback_flag(lit, region, graded=True)),
                }
                for method in list(METHODS) + list(MASK_METHODS.keys()):
                    pred_color = preds[method][region]
                    row[f"pred_{method}"] = pred_color
                    row[f"correct_{method}"] = int(pred_color == gt_color)
                rows.append(row)

    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    all_methods = list(METHODS) + list(MASK_METHODS.keys())
    summary: dict = {
        "n_ground_truth_crops": len(GROUND_TRUTH),
        "n_cells_total": len(rows),
        "methods": {},
    }
    for method in all_methods:
        total = len(rows)
        correct = sum(r[f"correct_{method}"] for r in rows)
        black_rows = [r for r in rows if r["is_gt_black"]]
        black_total = len(black_rows)
        black_correct = sum(r[f"correct_{method}"] for r in black_rows)

        by_condition = {}
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in rows if r["condition"] == condition]
            cond_total = len(cond_rows)
            cond_correct = sum(r[f"correct_{method}"] for r in cond_rows)
            cond_black_rows = [r for r in cond_rows if r["is_gt_black"]]
            cond_black_total = len(cond_black_rows)
            cond_black_correct = sum(r[f"correct_{method}"] for r in cond_black_rows)
            by_condition[condition] = {
                "overall_accuracy": round(cond_correct / cond_total, 4) if cond_total else None,
                "gt_black_accuracy": (
                    round(cond_black_correct / cond_black_total, 4) if cond_black_total else None
                ),
            }

        summary["methods"][method] = {
            "overall_accuracy": round(correct / total, 4),
            "overall_correct": correct,
            "overall_total": total,
            "gt_black_accuracy": round(black_correct / black_total, 4) if black_total else None,
            "gt_black_total_cells": black_total,
            "accuracy_by_condition": by_condition,
        }

    n_fallback_b = sum(r["b_used_fallback"] for r in rows)
    n_fallback_b_graded = sum(r["b_graded_used_fallback"] for r in rows)
    summary["fallback_usage"] = {
        "n_cells_total": len(rows),
        "b_binary_fallback_rate": round(n_fallback_b / len(rows), 4),
        "b_graded_any_relaxation_rate": round(n_fallback_b_graded / len(rows), 4),
        "note": (
            "b_graded_any_relaxation_rate는 '첫 단계(val_min=35)로 충분하지 않아 어느 "
            "완화 단계든 사용한' 비율이다 - 완화된 단계를 쓰는 것 자체는 문제가 아니라 "
            "설계 의도이므로, 이 수치는 '폴백이 발동했는가'와 '정확도'를 분리해서 봐야 "
            "한다(완화 단계를 더 자주 쓰면서도 Accuracy가 올랐다면 설계가 의도대로 "
            "작동했다는 뜻)."
        ),
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "overall_accuracy": {m: summary["methods"][m]["overall_accuracy"] for m in all_methods},
                "gt_black_accuracy": {m: summary["methods"][m]["gt_black_accuracy"] for m in all_methods},
                "fallback_usage": summary["fallback_usage"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
