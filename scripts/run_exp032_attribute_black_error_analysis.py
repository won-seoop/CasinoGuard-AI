"""EXP-032: GT=black 오분류 Error Analysis - Hue 다수결 단위 직접 진단 (지침 19, FC-012 후속).

관련 실험: EXP-025(PAR-016), EXP-026(PAR-017), EXP-027(PAR-018), EXP-028(PAR-019),
EXP-030(PAR-021), EXP-031(PAR-022)

## 배경
FC-012(검정 옷이 strong_light/cool_cast 조명에서 blue로 오분류)를 겨냥한 pixel-level
보정 4종(White Balance/Segmentation Mask/Gamma-self/Gamma-background) + confidence
신호 후보 2종(achromatic 채도·BGR std/Frame 노출·Cast)이 전부 부분효과 또는 완전
반증으로 끝났다(EXP-031 Decision). EXP-031 Next Action은 "더 많은 신호를 추가하기
전에, method b/c_wb의 오분류 GT=black 셀을 직접 조건x Hue bin 단위로 Error Analysis해
무엇이 실제로 오분류를 일으키는지 먼저 진단할 것"을 명시했다. 이 실험은 그 진단이다 -
새로운 보정이나 신호를 설계하지 않고, 기존 로직(region_dominant_color)이 왜 틀렸는지를
Hue 다수결 중간값(진단 함수, src/attributes/color.py에 신규 추가)으로 직접 들여다본다.

## 대상
- method "b"(src 기준 실험용 변형, region_dominant_color 직접 사용)
- method "c_wb"(production이 실제로 run_full_pipeline.py에 연결한 방식, Segmentation
  Mask + White Balance)
두 방식 모두 내부적으로 동일한 region_dominant_color 로직을 쓰므로(c_wb는 입력 픽셀
집합만 마스크로 바뀜) 같은 진단 함수로 비교할 수 있다.

Dataset: EXP-027/028과 완전히 동일한 coco128 n=21 Ground Truth crop + 동일 5종
합성 조명. GT=black 110셀(22 base cell x 5 조건, EXP-031이 인용한 수치와 일치하는지도
재확인).

실행: python scripts/run_exp032_attribute_black_error_analysis.py
출력: results/EXP-032/{error_rows.csv, summary.json}
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attributes.color import (  # noqa: E402
    classify_person_attributes_with_mask_and_wb_diagnostic,
    region_dominant_color_diagnostic,
    split_upper_lower,
    white_balance_gray_world,
)

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-032")
MIN_BOX_AREA = 120 * 250
SEG_IOU_MATCH_THRESHOLD = 0.5

LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

# EXP-027/028과 완전히 동일한 Ground Truth (동일 Detector/conf=0.4 -> 동일 crop 이름 재현).
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
    """EXP-028과 동일한 Detector+Segmenter IoU 매칭(동일 GT 이름 재현 확인용)."""
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
    """EXP-025/027/028과 동일한 합성 조명 변형."""
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


def diagnose_method_b(lit_crop: np.ndarray, region: str) -> dict:
    upper, lower = split_upper_lower(
        lit_crop, head_skip_ratio=0.22, upper_end_ratio=0.55, foot_skip_ratio=0.08, side_margin_ratio=0.12
    )
    target = upper if region == "upper" else lower
    return region_dominant_color_diagnostic(target)


def diagnose_method_c_wb(lit_crop: np.ndarray, mask: np.ndarray, region: str) -> dict:
    diag = classify_person_attributes_with_mask_and_wb_diagnostic(lit_crop, mask)
    return diag[region]


# EXP-032에서 직접 발견한 GT 라벨 오류 (지침 31: GT를 지어내지 않고, 오류를 발견하면
# 투명하게 교정한다). results/EXP-032/crops/*.jpg를 육안으로 직접 확인해 확정했다 -
# EXP-027이 원래 GT를 만들 때 쓴 coco128 흑백/그레이스케일 원본 사진 2장에서, "어두운
# 자켓/바지"라는 주석과 달리 실제로는 밝은 셔츠를 입고 있었다(아래 주석 참고).
# 과거 EXP-027/028/030/031의 experiment.md/PAR에 이미 기록된 수치는 당시 GT 기준으로
# 유효했던 측정이므로 건드리지 않는다(지침 38, 과거 기록 보존) - 이 교정은 EXP-032
# 자신의 재측정에만 적용하고, "교정 전/후" 차이를 그대로 함께 보고한다.
GT_LABEL_CORRECTIONS = {
    ("000000000086_0", "upper"): {
        "old": "black",
        "new": "white",
        "reason": (
            "B&W 모터사이클 사진. 주석은 '어두운 자켓'이라 적었지만 results/EXP-032/"
            "crops/000000000086_0.jpg를 육안 확인하면 남성은 밝은 크림색 드레스셔츠를"
            " 입고 있다(검정은 하의뿐). 상의 GT를 white로 교정."
        ),
    },
    ("000000000634_0", "upper"): {
        "old": "black",
        "new": "gray",
        "reason": (
            "그레이스케일 스케이트보더 사진. 주석은 '그레이스케일 스케이트보더'(옷 색상"
            " 미언급)로만 적혀 상/하의 모두 black으로 라벨링됐지만, crops/"
            "000000000634_0.jpg를 육안 확인하면 상의는 연회색 티셔츠, 하의만 검정"
            " 반바지다. 상의 GT를 gray로 교정."
        ),
    },
}

# 교정하지 않고 '애매함'으로만 기록하는 경계 사례 (참고용, 과대 교정 방지):
# - 000000000113_0 lower(네이비 바지)/000000000572_1 upper(네이비 스웨터): 원 주석이
#   이미 "네이비"라고 적어두었지만, 실내 저조도 사진에서 육안으로도 거의 검정에
#   가깝게 보여 black/blue 경계가 실제로 모호하다 - 다수결 예측(blue)이 틀렸다고
#   단정할 근거가 부족해 교정하지 않았다.


def apply_gt_corrections(ground_truth: dict) -> dict:
    corrected = {name: dict(gt) for name, gt in ground_truth.items()}
    for (name, region), fix in GT_LABEL_CORRECTIONS.items():
        corrected[name][region] = fix["new"]
    return corrected


def build_rows_and_summary(ground_truth: dict, crops: dict, masks: dict) -> tuple[list[dict], dict]:
    # GT=black 셀만 대상으로 전수 진단 (upper/lower 각각, method b/c_wb 각각).
    rows = []
    for name, gt in ground_truth.items():
        crop = crops[name]
        mask = masks[name]
        for region in ("upper", "lower"):
            if gt[region] != "black":
                continue
            for condition in LIGHTING_CONDITIONS:
                lit = apply_lighting(crop, condition)
                diag_b = diagnose_method_b(lit, region)
                diag_c = diagnose_method_c_wb(lit, mask, region)
                for method, diag in (("b", diag_b), ("c_wb", diag_c)):
                    rows.append(
                        {
                            "name": name,
                            "region": region,
                            "condition": condition,
                            "method": method,
                            "correct": int(diag["color"] == "black"),
                            "pred_color": diag["color"],
                            "used_fallback": diag["used_fallback"],
                            "used_mask": diag.get("used_mask"),
                            "filtered_fraction": diag["filtered_fraction"],
                            "dominant_bin": diag["dominant_bin"],
                            "dominant_bin_count": diag["dominant_bin_count"],
                            "runner_up_bin": diag["runner_up_bin"],
                            "runner_up_bin_count": diag["runner_up_bin_count"],
                            "median_h": diag["median_h"],
                            "median_s": diag["median_s"],
                            "median_v": diag["median_v"],
                        }
                    )

    n_base_cells = sum(
        1 for gt in ground_truth.values() for region in ("upper", "lower") if gt[region] == "black"
    )

    summary: dict = {
        "n_base_black_cells": n_base_cells,
        "n_cells_per_method": n_base_cells * len(LIGHTING_CONDITIONS),
        "methods": {},
    }

    for method in ("b", "c_wb"):
        method_rows = [r for r in rows if r["method"] == method]
        total = len(method_rows)
        correct = sum(r["correct"] for r in method_rows)
        misclassified = [r for r in method_rows if not r["correct"]]

        # 1) 오분류가 필터 폴백(진짜 검정이라 필터에 다 걸려 원본으로 되돌아간) 경로에서
        #    발생하는 비율 vs 필터가 정상 작동했는데도 틀린 비율.
        n_fallback_misclassified = sum(1 for r in misclassified if r["used_fallback"])
        n_filtered_misclassified = len(misclassified) - n_fallback_misclassified

        # 2) 오분류된 셀의 예측 색상 분포 (FC-012가 가정한 "항상 blue"인지 실제 확인).
        pred_color_dist = dict(Counter(r["pred_color"] for r in misclassified))

        # 3) 오분류된 셀의 dominant_bin(최종 Hue 분류를 결정한 bin)과 runner_up_bin의
        #    득표 차이 - 차이가 작을수록(margin이 좁을수록) 다수결이 "거의 동전 던지기"
        #    였다는 뜻이다.
        margins = []
        for r in misclassified:
            if r["dominant_bin_count"] and r["runner_up_bin_count"] is not None:
                total_votes = r["dominant_bin_count"] + r["runner_up_bin_count"]
                if total_votes > 0:
                    margins.append(r["dominant_bin_count"] / total_votes)
        narrow_margin_count = sum(1 for m in margins if m < 0.6)

        # 4) 조건별 오분류율 + 조건별 median_v(밝기)·median_s(채도) 평균 - "진짜 원인이
        #    V(밝기)가 VAL_BLACK(45) 문턱을 못 넘었는지, 아니면 S/H가 흔들렸는지" 구분.
        by_condition = {}
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in method_rows if r["condition"] == condition]
            cond_total = len(cond_rows)
            cond_correct = sum(r["correct"] for r in cond_rows)
            cond_wrong = [r for r in cond_rows if not r["correct"]]
            by_condition[condition] = {
                "accuracy": round(cond_correct / cond_total, 4) if cond_total else None,
                "n_misclassified": len(cond_wrong),
                "mean_median_v_when_wrong": (
                    round(float(np.mean([r["median_v"] for r in cond_wrong])), 1) if cond_wrong else None
                ),
                "mean_median_s_when_wrong": (
                    round(float(np.mean([r["median_s"] for r in cond_wrong])), 1) if cond_wrong else None
                ),
                "pred_color_dist_when_wrong": dict(Counter(r["pred_color"] for r in cond_wrong)),
            }

        summary["methods"][method] = {
            "accuracy": round(correct / total, 4),
            "total_cells": total,
            "n_misclassified": len(misclassified),
            "n_fallback_misclassified": n_fallback_misclassified,
            "n_filtered_misclassified": n_filtered_misclassified,
            "pred_color_dist_when_wrong": pred_color_dist,
            "n_narrow_margin_misclassified": narrow_margin_count,
            "narrow_margin_fraction_of_errors": (
                round(narrow_margin_count / len(margins), 4) if margins else None
            ),
            "accuracy_by_condition": by_condition,
        }

    return rows, summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crops, masks = detect_crops_and_masks()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")
    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (detector mismatch?): {missing}")

    rows, summary = build_rows_and_summary(GROUND_TRUTH, crops, masks)
    with open(RESULTS_DIR / "error_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[original GT] n_base_black_cells={summary['n_base_black_cells']}")

    # GT 라벨 오류 교정 재측정 (지침 31, 위 GT_LABEL_CORRECTIONS 참고) - 육안으로 확인한
    # 명백한 라벨 오류 2건만 고쳐서 같은 파이프라인을 다시 돌리고, 교정 전/후 Accuracy
    # 차이를 "라벨 오류가 측정치에 얼마나 영향을 줬는가"로 직접 보고한다.
    corrected_gt = apply_gt_corrections(GROUND_TRUTH)
    corrected_rows, corrected_summary = build_rows_and_summary(corrected_gt, crops, masks)
    with open(RESULTS_DIR / "error_rows_gt_corrected.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corrected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corrected_rows)
    print(f"[GT-corrected] n_base_black_cells={corrected_summary['n_base_black_cells']}")

    comparison = {
        "gt_corrections_applied": [
            {"name": name, "region": region, **fix} for (name, region), fix in GT_LABEL_CORRECTIONS.items()
        ],
        "n_cells_corrected": len(GT_LABEL_CORRECTIONS),
        "original": {
            m: {
                "accuracy": summary["methods"][m]["accuracy"],
                "normal_condition_accuracy": summary["methods"][m]["accuracy_by_condition"]["normal"]["accuracy"],
            }
            for m in ("b", "c_wb")
        },
        "gt_corrected": {
            m: {
                "accuracy": corrected_summary["methods"][m]["accuracy"],
                "normal_condition_accuracy": corrected_summary["methods"][m]["accuracy_by_condition"]["normal"][
                    "accuracy"
                ],
            }
            for m in ("b", "c_wb")
        },
    }

    full_summary = {
        "original": summary,
        "gt_corrected": corrected_summary,
        "gt_correction_impact": comparison,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
