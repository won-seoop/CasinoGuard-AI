"""EXP-030: FC-012 strong_light(노출 과다) 복원 - Adaptive Gamma Correction (지침 19).

Roadmap(01. Architecture & Roadmap) 다음 Action 3번: FC-012는 EXP-026(White Balance)/
EXP-028(Segmentation Mask)로도 해결되지 않았고, 다음 후보로 "Gamma 보정 등 노출 복원
기법 또는 '판단 불가/낮은 신뢰도' 표시로의 설계 전환"을 명시했다. EXP-026/028의 실측
(PAR-017/PAR-019)을 보면 GT=black 셀의 strong_light->blue 오분류율이 b(18.2%) ->
b_wb(22.7%) -> c_wb(27.3%)로 WB/Mask를 추가할수록 오히려 악화되고 있었다 - WB는 "채널
간 상대적 Cast"를 보정하는 기법인데 strong_light(EXP-025 `img*1.9+25`)는 모든 채널이
함께 밝아지는 전역 노출 문제라 WB로 해결될 이유가 없었다(원인 재분석). 이 실험은 전역
밝기 자체를 보정하는 Adaptive Gamma Correction(크롭의 실측 평균 밝기로 감마를 추정하는
표준 Auto-Exposure 기법, src/attributes/color.py::adaptive_gamma_correct)을 method b
앞에 추가해(b_gamma) strong_light를 개선하면서 다른 조건(특히 b_wb가 이미 개선한
cool_cast, 그리고 아직 손대지 않은 low_light)을 악화시키지 않는지 측정한다. Gamma와
WB를 함께 적용(b_wb_gamma)해 두 보정이 서로 다른 원인(노출 vs Cast)을 겨냥하므로
합쳐지는지도 확인한다.

중요: 이 실험은 EXP-025의 특정 합성 변형(`img*1.9+25`)의 역함수를 외우지 않는다 -
adaptive_gamma_correct는 crop의 실측 평균 밝기만으로 감마를 추정하므로, 실제로는
존재하지 않는 "answer key를 아는 치팅"이 되지 않도록 normal/low_light/warm_cast/
cool_cast에도 동일하게 (무조건) 적용해 회귀가 있는지 함께 확인한다.

실행: python scripts/run_exp030_attribute_gamma_correction.py
출력: results/EXP-030/{predictions.csv, summary.json}
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
    adaptive_gamma_correct_from_background,
    classify_person_attributes,
    region_dominant_color,
    split_upper_lower,
)

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-030")
MIN_BOX_AREA = 120 * 250
BACKGROUND_MIN_PIXELS = 5000  # 배경 샘플이 너무 작으면(꽉 찬 클로즈업) 노출 추정이 불안정

METHODS = ["b", "b_wb", "b_gamma", "b_wb_gamma", "b_gamma_bg"]
LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

# EXP-027/028과 완전히 동일한 Ground Truth (동일 Detector yolo11n.pt, conf=0.4, coco128
# train2017 -> 동일 crop 이름 재현 확인됨).
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


def detect_person_crops_and_bboxes() -> tuple[dict[str, np.ndarray], dict[str, tuple[Path, tuple[int, int, int, int]]]]:
    """EXP-030 대안 B(배경 기반 노출 추정)를 위해, crop 자체뿐 아니라 원본 이미지 경로와
    bbox 좌표도 함께 반환한다 - 배경(person bbox 밖 영역)은 person 영역이 조명 변형된
    "뒤"에 같은 이미지에서 다시 만들어야 하므로 원본 전체 이미지가 필요하다."""
    model = YOLO("yolo11n.pt")
    crops: dict[str, np.ndarray] = {}
    refs: dict[str, tuple[Path, tuple[int, int, int, int]]] = {}
    image_paths = sorted(COCO128_IMAGES.glob("*.jpg"))
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = model.predict(img, classes=[0], conf=0.4, verbose=False)[0]
        stem = img_path.stem
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_BOX_AREA:
                continue
            name = f"{stem}_{i}"
            crops[name] = img[y1:y2, x1:x2]
            refs[name] = (img_path, (x1, y1, x2, y2))
    return crops, refs


def background_sample(full_img_lit: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """person bbox를 제외한 Frame 나머지 영역의 픽셀만 (N,1,3) 형태로 반환한다
    (estimate_gamma_from_reference는 평균만 쓰므로 위치 정보는 필요 없다)."""
    h, w = full_img_lit.shape[:2]
    x1, y1, x2, y2 = bbox
    mask = np.ones((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = False
    pixels = full_img_lit[mask]
    return pixels.reshape(-1, 1, 3)


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027/028과 완전히 동일한 합성 조명 변형."""
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


def predict_b_gamma_bg(lit_crop: np.ndarray, lit_background: np.ndarray) -> dict[str, str]:
    """method b와 동일한 상/하 분리·필터링·Hue 다수결을 쓰되, 감마를 crop 자신이 아니라
    같은 Frame의 배경에서 추정한다(EXP-030 대안 B)."""
    corrected = adaptive_gamma_correct_from_background(lit_crop, lit_background)
    upper, lower = split_upper_lower(
        corrected, head_skip_ratio=0.22, upper_end_ratio=0.55, foot_skip_ratio=0.08, side_margin_ratio=0.12
    )
    return {"upper": region_dominant_color(upper), "lower": region_dominant_color(lower)}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crops, refs = detect_person_crops_and_bboxes()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")

    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (re-run detection?): {missing}")

    # 원본 전체 이미지는 이름(GT key)이 여러 crop(같은 이미지의 여러 사람)을 공유할 수
    # 있으므로 이미지 경로별로 한 번만 읽어 캐시한다.
    full_images: dict[Path, np.ndarray] = {}
    for img_path, _ in refs.values():
        if img_path not in full_images:
            full_images[img_path] = cv2.imread(str(img_path))

    small_background_skipped: list[str] = []
    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        img_path, bbox = refs[name]
        for condition in LIGHTING_CONDITIONS:
            lit = apply_lighting(crop, condition)
            lit_full = apply_lighting(full_images[img_path], condition)
            bg_pixels = background_sample(lit_full, bbox)
            if bg_pixels.shape[0] < BACKGROUND_MIN_PIXELS and name not in small_background_skipped:
                small_background_skipped.append(name)
            for method in METHODS:
                if method == "b_gamma_bg":
                    pred = predict_b_gamma_bg(lit, bg_pixels)
                else:
                    pred = classify_person_attributes(lit, method=method)
                rows.append(
                    {
                        "name": name,
                        "condition": condition,
                        "method": method,
                        "gt_upper": gt["upper"],
                        "gt_lower": gt["lower"],
                        "pred_upper": pred["upper"],
                        "pred_lower": pred["lower"],
                        "upper_correct": int(pred["upper"] == gt["upper"]),
                        "lower_correct": int(pred["lower"] == gt["lower"]),
                    }
                )
    if small_background_skipped:
        print(
            f"NOTE: {len(small_background_skipped)} crop(s) have a background sample smaller than "
            f"{BACKGROUND_MIN_PIXELS}px (person fills most of the frame): {small_background_skipped}"
        )

    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {"n_ground_truth_crops": len(GROUND_TRUTH), "methods": {}}
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

        # FC-012가 겨냥한 실패 모드: GT=black 셀이 strong_light/cool_cast에서 "blue"로
        # 오분류되는 비율 (EXP-026/027/028과 동일 분석 패턴, 직접 비교 가능하게 유지).
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

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
