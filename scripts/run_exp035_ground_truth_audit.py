"""EXP-035: Attribute Color GT 라벨 21개 crop x 42개 영역 전체 재검증 (지침 19/31,
FC-012/EXP-032/EXP-033/EXP-034 후속, Roadmap 다음 Action 최우선 - "GT 라벨 전체
재검증은 이제 3회 이월됐다").

관련 실험: EXP-027(PAR-018, GT 최초 n=21 라벨링), EXP-032(PAR-023, 오분류 110셀
중 10셀만 육안 재확인해 라벨 오류 2건 발견), EXP-033(PAR-024), EXP-034(PAR-025,
현재 production 후보 val_min=0/sat_min=20 확정)

## 배경

EXP-032는 "검정→blue 오분류"로 분류된 110셀 중 normal 조건 10셀만 육안으로
재확인해 라벨 오류 2건(086_0 upper, 634_0 upper)을 발견했다. 이 교정은 그 2건이
"classifier가 아니라 GT가 틀렸다"는 것을 보여줬지만, 감사 범위가 "오분류로 보고된
셀의 일부"로 제한되어 있었다 - GT가 애초에 틀렸는데 classifier의 예측과 우연히
일치해 "정답"으로 집계된 셀은 전혀 검토되지 않았다.

이 실험은 Roadmap이 3번 이월한 작업 — results/EXP-027/crops/*.jpg 21개 crop의
upper/lower 42개 라벨 **전체**를 처음부터 다시 육안으로 재확인 — 을 실제로
수행한다. 또한 GROUND_TRUTH가 EXP-027/028/030/031/032/033/034 6개 스크립트에
그대로 복사-붙여넣기되어 있어서, EXP-032가 찾은 교정 2건이 EXP-033/034의
복사본에는 반영됐지만 EXP-027/028/030/031/032 자신의 원본 복사본에는 반영되지
않는 Drift가 이미 발생해 있었다는 것도 이번 감사 중 직접 확인했다(아래 Analysis
참고) - 이는 "일부만 교정하고 끝났다"로 착각하기 쉬운 구조적 위험이다.

## Hypothesis

1. 오분류로 보고되지 않은 나머지 32셀(42-10) 중에서도 추가 라벨 오류가 있을 것이다
   (EXP-032가 "classifier 예측과 우연히 일치한 틀린 GT"는 애초에 볼 수 없는 범위
   였으므로).
2. GROUND_TRUTH 복사-붙여넣기 중복 때문에 EXP-032 교정이 모든 스크립트에
   전파되지 않았을 것이다.

## 측정

results/EXP-027/crops/*.jpg 21개 전체를 3배 Upscale해 직접 육안으로 재확인하고,
GROUND_TRUTH 42개 라벨(상/하 x 21) 전부를 원본 주석과 대조한다. 발견한 오류는
evaluation/attribute_color/coco128_ground_truth.py(신규 단일 진실 공급원)에
근거와 함께 기록하고, 교정 전/후 GT로 EXP-034가 확정한 현재 최종 후보
(val_min=0, sat_min=20, c_wb_relaxed)를 동일 Mask+WB 파이프라인에서 재측정해
Before/After를 비교한다.

실행: python scripts/run_exp035_ground_truth_audit.py
출력: results/EXP-035/{predictions.csv, summary.json}
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
from evaluation.attribute_color.coco128_ground_truth import (  # noqa: E402
    CORRECTIONS,
    GROUND_TRUTH,
    assert_consistent,
)
from src.attributes.color import classify_person_attributes_with_mask_and_wb_relaxed  # noqa: E402

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-035")
MIN_BOX_AREA = 120 * 250
SEG_IOU_MATCH_THRESHOLD = 0.5

LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

# EXP-034가 확정한 현재 최종 후보 설정.
VAL_MIN = 0
SAT_MIN = 20

# EXP-035 이전까지 6개 스크립트에 흩어져 있던 "GT" 중 EXP-033/EXP-034가 실제로 쓴
# 버전(EXP-032의 2건 교정만 반영, 이번 EXP-035의 5건 신규 교정은 미반영) -
# Before 비교의 기준점. evaluation/ 모듈이 생기기 전 마지막 GT 상태를 그대로 보존한다.
GROUND_TRUTH_BEFORE = {
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
    """EXP-028/032/033/034와 동일한 Detector+Segmenter IoU 매칭(동일 GT 이름 재현용)."""
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
    """EXP-025/027/028/032/033/034와 동일한 합성 조명 변형."""
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


def _score(gt: dict[str, dict[str, str]], preds: dict[tuple[str, str, str], str]) -> dict:
    correct, total = 0, 0
    black_correct, black_total = 0, 0
    nonblack_correct, nonblack_total = 0, 0
    by_condition: dict[str, dict] = {
        c: {"correct": 0, "total": 0, "black_correct": 0, "black_total": 0} for c in LIGHTING_CONDITIONS
    }

    for name, regions in gt.items():
        for condition in LIGHTING_CONDITIONS:
            for region in ("upper", "lower"):
                gt_color = regions[region]
                pred_color = preds[(name, condition, region)]
                is_correct = int(pred_color == gt_color)
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
    assert_consistent()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    crops, masks = detect_crops_and_masks()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")
    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (detector mismatch?): {missing}")

    # 예측은 GT 교정과 무관하게 동일(예측 파이프라인은 손대지 않았다) -
    # (name, condition, region) -> pred_color 한 번만 계산해 Before/After에 공유한다.
    preds: dict[tuple[str, str, str], str] = {}
    pred_rows: list[dict] = []
    for name in GROUND_TRUTH:
        mask = masks[name]
        for condition in LIGHTING_CONDITIONS:
            lit = apply_lighting(crops[name], condition)
            pred = classify_person_attributes_with_mask_and_wb_relaxed(lit, mask, val_min=VAL_MIN, sat_min=SAT_MIN)
            for region in ("upper", "lower"):
                preds[(name, condition, region)] = pred[region]
                pred_rows.append(
                    {
                        "name": name,
                        "region": region,
                        "condition": condition,
                        "pred_color": pred[region],
                        "gt_before": GROUND_TRUTH_BEFORE[name][region],
                        "gt_after": GROUND_TRUTH[name][region],
                        "gt_changed": int(GROUND_TRUTH_BEFORE[name][region] != GROUND_TRUTH[name][region]),
                        "correct_before": int(pred[region] == GROUND_TRUTH_BEFORE[name][region]),
                        "correct_after": int(pred[region] == GROUND_TRUTH[name][region]),
                    }
                )

    before = _score(GROUND_TRUTH_BEFORE, preds)
    after = _score(GROUND_TRUTH, preds)

    changed_cells = [
        {"name": c["name"], "region": c["region"], "old": c["old"], "new": c["new"], "source": c["source"]}
        for c in CORRECTIONS
    ]

    summary = {
        "config": {"val_min": VAL_MIN, "sat_min": SAT_MIN, "method": "c_wb_relaxed"},
        "n_ground_truth_crops": len(GROUND_TRUTH),
        "n_cells_total": len(GROUND_TRUTH) * 2,
        "n_corrections_total": len(CORRECTIONS),
        "n_corrections_from_exp032_already_in_before": sum(1 for c in CORRECTIONS if c["source"] == "EXP-032"),
        "n_corrections_new_in_exp035": sum(1 for c in CORRECTIONS if c["source"] == "EXP-035"),
        "changed_cells": changed_cells,
        "before_gt_correction": before,
        "after_gt_correction": after,
        "delta": {
            "overall_accuracy": round(after["overall_accuracy"] - before["overall_accuracy"], 4),
            "gt_black_accuracy": round((after["gt_black_accuracy"] or 0) - (before["gt_black_accuracy"] or 0), 4),
            "nonblack_accuracy": round((after["nonblack_accuracy"] or 0) - (before["nonblack_accuracy"] or 0), 4),
        },
    }

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(RESULTS_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
