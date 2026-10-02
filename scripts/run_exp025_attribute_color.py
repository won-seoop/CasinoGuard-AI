"""EXP-025: Attribute Metadata - 상의/하의 색상 분류 Baseline vs 대안 A vs 대안 B (지침 19).

실제 YOLO11n Person Detection 결과(ultralytics 번들 bus.jpg/zidane.jpg)에서 crop을 뽑고,
수동으로 라벨링한 상/하의 색상 Ground Truth와 비교한다. 핵심 질문은 "조명이 바뀌어도 같은
색으로 분류되는가"이므로, 정상 조명 crop에 합성 조명 변형(저조도/강한 조명·역광/난색·한색
Cast)을 적용해 각 방식의 Accuracy를 측정한다.

실행: python scripts/run_exp025_attribute_color.py
출력: results/EXP-025/{crops/, lighting_variants/, predictions.csv, summary.json, metrics.png 없음(plot 생략)}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attributes.color import classify_person_attributes  # noqa: E402

RESULTS_DIR = Path("results/EXP-025")
CROPS_DIR = RESULTS_DIR / "crops"
VARIANTS_DIR = RESULTS_DIR / "lighting_variants"

# 수동 라벨링한 Ground Truth. Read 도구로 각 crop을 직접 눈으로 확인하고 라벨링했다
# (EXP-025 experiment.md Dataset 절 참고). bus_3은 occlusion이 심해 사람 여부가
# 불확실해 평가 대상에서 제외한다 (지침 31: Ground Truth를 억지로 만들지 않는다).
GROUND_TRUTH = {
    "zidane_0": {"upper": "black", "lower": "black"},
    "zidane_1": {"upper": "black", "lower": "black"},
    "bus_0": {"upper": "black", "lower": "black"},
    "bus_1": {"upper": "white", "lower": "blue"},
    "bus_2": {"upper": "black", "lower": "blue"},
}


def detect_person_crops() -> dict[str, np.ndarray]:
    model = YOLO("yolo11n.pt")
    crops: dict[str, np.ndarray] = {}
    for img_name in ["zidane.jpg", "bus.jpg"]:
        path = str(ASSETS / img_name)
        img = cv2.imread(path)
        res = model.predict(img, classes=[0], conf=0.3, verbose=False)[0]
        stem = img_name.split(".")[0]
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = img[max(0, y1) : y2, max(0, x1) : x2]
            name = f"{stem}_{i}"
            crops[name] = crop
    return crops


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """합성 조명 변형. 실제 카메라에서 흔히 관찰되는 변화를 단순 모델로 근사한다."""
    img = img_bgr.astype(np.float32)
    if condition == "normal":
        out = img
    elif condition == "low_light":
        out = img * 0.35
    elif condition == "strong_light":
        # 강한 인공조명/역광으로 인한 과노출 + 대비 감소(washed-out)
        out = img * 1.9 + 25
    elif condition == "warm_cast":
        # 백열등 등 난색 조명 (R 증가, B 감소)
        out = img.copy()
        out[:, :, 2] *= 1.35  # R
        out[:, :, 0] *= 0.75  # B
    elif condition == "cool_cast":
        # 형광등/역광 등 한색 조명 (B 증가, R 감소)
        out = img.copy()
        out[:, :, 0] *= 1.35  # B
        out[:, :, 2] *= 0.75  # R
    else:
        raise ValueError(condition)
    return np.clip(out, 0, 255).astype(np.uint8)


LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]
METHODS = ["baseline", "a", "b"]


def main() -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    crops = detect_person_crops()
    for name, crop in crops.items():
        cv2.imwrite(str(CROPS_DIR / f"{name}.jpg"), crop)

    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        for condition in LIGHTING_CONDITIONS:
            variant = apply_lighting(crop, condition)
            cv2.imwrite(str(VARIANTS_DIR / f"{name}_{condition}.jpg"), variant)
            for method in METHODS:
                pred = classify_person_attributes(variant, method=method)
                upper_correct = pred["upper"] == gt["upper"]
                lower_correct = pred["lower"] == gt["lower"]
                rows.append(
                    {
                        "crop": name,
                        "condition": condition,
                        "method": method,
                        "gt_upper": gt["upper"],
                        "gt_lower": gt["lower"],
                        "pred_upper": pred["upper"],
                        "pred_lower": pred["lower"],
                        "upper_correct": int(upper_correct),
                        "lower_correct": int(lower_correct),
                        # baseline은 상/하 구분이 없으므로 "둘 중 하나라도 맞으면" 최선의 경우로도 집계
                        "either_correct": int(upper_correct or lower_correct),
                    }
                )

    csv_path = RESULTS_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 요약 집계: method x condition 별 Accuracy
    summary: dict = {"per_method_condition": {}, "per_method_overall": {}}
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in method_rows if r["condition"] == condition]
            n = len(cond_rows)
            if method == "baseline":
                acc = sum(r["either_correct"] for r in cond_rows) / n
            else:
                correct = sum(r["upper_correct"] for r in cond_rows) + sum(r["lower_correct"] for r in cond_rows)
                acc = correct / (n * 2)
            summary["per_method_condition"].setdefault(method, {})[condition] = round(acc, 4)

        if method == "baseline":
            overall = sum(r["either_correct"] for r in method_rows) / len(method_rows)
        else:
            correct = sum(r["upper_correct"] for r in method_rows) + sum(r["lower_correct"] for r in method_rows)
            overall = correct / (len(method_rows) * 2)
        summary["per_method_overall"][method] = round(overall, 4)

    # 조명 변형(저조도/강한조명/난색/한색)에 대한 "안정성": normal 대비 나머지 4개 조건 평균 Accuracy
    for method in METHODS:
        non_normal = [c for c in LIGHTING_CONDITIONS if c != "normal"]
        vals = [summary["per_method_condition"][method][c] for c in non_normal]
        summary.setdefault("lighting_stability_accuracy", {})[method] = round(sum(vals) / len(vals), 4)

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nrows: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
