"""EXP-027: Attribute Metadata 색상 분류 - coco128 기반 n 확장 재검증 (지침 19, FC-012 후속).

EXP-025/026은 ultralytics 번들 이미지(bus.jpg/zidane.jpg) 2장에서 나온 person crop 5개로만
Ground Truth를 만들어 검정 옷에 편중됐다 (Roadmap 01. Architecture & Roadmap 다음 Action:
"EXP-025 Attribute 색상 분류를 더 크고 색상 다양성 높은 실제 인물 crop으로 재검증"). 이 실험은
coco128(실제 COCO train2017 128장, PyPI/GitHub Release Asset 경로로 egress 제약 없이 접근
가능함을 확인)에서 person crop을 모으고, 상/하의가 모두 명확히 보이고 단일 색으로 판단 가능한
crop 21개를 수동으로 Ground Truth 라벨링해(지침 31) n을 5→21로 늘렸다.

측정 항목:
1. 기존 4가지 method(baseline/a/b/b_wb)를 더 큰 n·조명 조건(EXP-025와 동일 5종)에서 재측정.
2. FC-012가 겨냥하는 GT=black 셀만 슬라이싱한 조건별 Accuracy (EXP-026과 동일 분석 패턴).
3. Next Action 3번("무채색 신호 자체가 안정적인지 먼저 검증")을 위해, GT가 achromatic
   (black/white/gray)인 영역과 chromatic(그 외 색)인 영역의 raw HSV 채도(S) 분포가 실제로
   분리되는지 분석한다 — 분리가 나쁘면 "낮은 confidence는 unknown으로 표시" 설계 자체가
   성립하지 않는다.

실행: python scripts/run_exp027_attribute_coco128.py
출력: results/EXP-027/{crops/, predictions.csv, summary.json, achromatic_signal.csv}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attributes.color import (  # noqa: E402
    classify_person_attributes,
    region_color_signal,
    split_upper_lower,
)

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-027")
CROPS_DIR = RESULTS_DIR / "crops"
MIN_BOX_AREA = 120 * 250

METHODS = ["baseline", "a", "b", "b_wb"]
LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

ACHROMATIC_COLORS = {"black", "white", "gray"}

# 수동 Ground Truth (coco128 person crop 21개, Read 도구로 직접 눈으로 확인하고 라벨링,
# 그리드/개별 확대로 교차 검증함). 상/하의 중 하나라도 패턴이 다색이거나(꽃무늬/타이다이),
# 가려져 색을 판단할 수 없는 crop은 지침 31에 따라 제외했다(109개 추가 crop 중 21개만 채택).
GROUND_TRUTH = {
    "000000000086_0": {"upper": "black", "lower": "black"},  # B&W 오토바이 남성, 어두운 자켓/바지
    "000000000113_0": {"upper": "white", "lower": "black"},  # 크림 스트라이프 셔츠, 네이비 바지
    "000000000165_0": {"upper": "gray", "lower": "gray"},  # ACU 디지털 카모(그레이-탄 계열)
    "000000000165_1": {"upper": "black", "lower": "black"},  # 검정 정장+레드 타이
    "000000000328_0": {"upper": "black", "lower": "black"},  # B&W 해군 제복
    "000000000328_1": {"upper": "black", "lower": "black"},  # B&W 해군 제복(앉은 모습)
    "000000000328_2": {"upper": "black", "lower": "black"},  # B&W 해군 제복(모자 든 소년)
    "000000000368_0": {"upper": "red", "lower": "black"},  # 축구 유니폼(적색)+검정 반바지
    "000000000370_0": {"upper": "pink", "lower": "pink"},  # 분홍 원피스 여아
    "000000000389_0": {"upper": "white", "lower": "white"},  # 흰 드레스셔츠(바지 미노출, 셔츠 연장)
    "000000000395_0": {"upper": "orange", "lower": "orange"},  # 주황 재킷
    "000000000415_0": {"upper": "blue", "lower": "white"},  # 테니스 선수(블루 셔츠/화이트 반바지)
    "000000000431_0": {"upper": "white", "lower": "white"},  # 테니스 선수(올화이트)
    "000000000446_0": {"upper": "gray", "lower": "black"},  # 그레이 니트(나비무늬)+검정 바지
    "000000000459_0": {"upper": "black", "lower": "black"},  # 어두운 정장(거울 셀카)
    "000000000536_0": {"upper": "black", "lower": "black"},  # 검정 원피스
    "000000000564_0": {"upper": "white", "lower": "white"},  # 테니스 선수(올화이트)
    "000000000572_0": {"upper": "black", "lower": "black"},  # 어두운 니트+어두운 바지
    "000000000572_1": {"upper": "black", "lower": "blue"},  # 네이비 스웨터+청바지
    "000000000623_0": {"upper": "gray", "lower": "pink"},  # 그레이 가디건+핑크 바지
    "000000000634_0": {"upper": "black", "lower": "black"},  # 그레이스케일 스케이트보더
}


def detect_person_crops() -> dict[str, np.ndarray]:
    model = YOLO("yolo11n.pt")
    crops: dict[str, np.ndarray] = {}
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
            crop = img[y1:y2, x1:x2]
            crops[f"{stem}_{i}"] = crop
    return crops


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025와 동일한 합성 조명 변형(동일 Config로 재현해야 Before/After 비교가 유효함)."""
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


def achromatic_signal_rows(crops: dict[str, np.ndarray]) -> list[dict]:
    """Next Action 3: GT가 achromatic(black/white/gray)인 영역과 chromatic인 영역의
    raw HSV 채도(S) 분포가 실제로 분리되는지 분석하기 위한 영역별 평균 채도를 계산한다.
    상/하 영역을 분리해(method b와 동일 split) 영역 단위로 GT와 짝지을 수 있게 한다."""
    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        upper, lower = split_upper_lower(
            crop, head_skip_ratio=0.22, upper_end_ratio=0.55, foot_skip_ratio=0.08, side_margin_ratio=0.12
        )
        for region_name, region, gt_color in (("upper", upper, gt["upper"]), ("lower", lower, gt["lower"])):
            if region.size == 0:
                continue
            signal = region_color_signal(region)
            rows.append(
                {
                    "name": name,
                    "region": region_name,
                    "gt_color": gt_color,
                    "is_achromatic_gt": gt_color in ACHROMATIC_COLORS,
                    "mean_saturation": round(signal["mean_saturation"], 2),
                    "bgr_channel_std": round(signal["bgr_channel_std"], 2),
                }
            )
    return rows


def main() -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    crops = detect_person_crops()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")
    for name, crop in crops.items():
        cv2.imwrite(str(CROPS_DIR / f"{name}.jpg"), crop)

    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (re-run detection?): {missing}")

    # 1) method x lighting x GT 조합 전수 예측
    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        for condition in LIGHTING_CONDITIONS:
            lit = apply_lighting(crop, condition)
            for method in METHODS:
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

    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2) method별 전체 Accuracy + normal 조명만(EXP-025 "Overall Accuracy"와 동일 정의) +
    #    GT=black 셀만(FC-012/EXP-026과 동일 슬라이싱)
    summary: dict = {"n_ground_truth_crops": len(GROUND_TRUTH), "methods": {}}
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        total = len(method_rows) * 2  # upper+lower
        correct = sum(r["upper_correct"] + r["lower_correct"] for r in method_rows)

        normal_rows = [r for r in method_rows if r["condition"] == "normal"]
        normal_total = len(normal_rows) * 2
        normal_correct = sum(r["upper_correct"] + r["lower_correct"] for r in normal_rows)

        black_rows = [
            r for r in method_rows if r["gt_upper"] == "black" or r["gt_lower"] == "black"
        ]
        black_cells = []
        for r in black_rows:
            if r["gt_upper"] == "black":
                black_cells.append((r["upper_correct"], r["pred_upper"]))
            if r["gt_lower"] == "black":
                black_cells.append((r["lower_correct"], r["pred_lower"]))
        black_total = len(black_cells)
        black_correct = sum(c for c, _ in black_cells)
        black_to_blue = sum(1 for c, p in black_cells if not c and p == "blue")

        summary["methods"][method] = {
            "overall_accuracy": round(correct / total, 4),
            "overall_correct": correct,
            "overall_total": total,
            "normal_lighting_accuracy": round(normal_correct / normal_total, 4),
            "gt_black_accuracy": round(black_correct / black_total, 4) if black_total else None,
            "gt_black_total_cells": black_total,
            "gt_black_misclassified_as_blue": black_to_blue,
        }

        # 조명 조건별 Accuracy (EXP-025/026과 동일 형식)
        by_condition = {}
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in method_rows if r["condition"] == condition]
            cond_total = len(cond_rows) * 2
            cond_correct = sum(r["upper_correct"] + r["lower_correct"] for r in cond_rows)
            by_condition[condition] = round(cond_correct / cond_total, 4)
        summary["methods"][method]["accuracy_by_condition"] = by_condition

    # 3) achromatic 신호 분리도 분석 (Next Action 3)
    signal_rows = achromatic_signal_rows(crops)
    with open(RESULTS_DIR / "achromatic_signal.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(signal_rows[0].keys()))
        writer.writeheader()
        writer.writerows(signal_rows)

    achromatic_s = [r["mean_saturation"] for r in signal_rows if r["is_achromatic_gt"]]
    chromatic_s = [r["mean_saturation"] for r in signal_rows if not r["is_achromatic_gt"]]
    summary["achromatic_signal"] = {
        "n_achromatic_regions": len(achromatic_s),
        "n_chromatic_regions": len(chromatic_s),
        "achromatic_saturation_mean": round(float(np.mean(achromatic_s)), 2) if achromatic_s else None,
        "achromatic_saturation_p90": round(float(np.percentile(achromatic_s, 90)), 2) if achromatic_s else None,
        "achromatic_saturation_max": round(float(np.max(achromatic_s)), 2) if achromatic_s else None,
        "chromatic_saturation_mean": round(float(np.mean(chromatic_s)), 2) if chromatic_s else None,
        "chromatic_saturation_p10": round(float(np.percentile(chromatic_s, 10)), 2) if chromatic_s else None,
        "chromatic_saturation_min": round(float(np.min(chromatic_s)), 2) if chromatic_s else None,
        "overlap": (
            float(np.max(achromatic_s)) >= float(np.min(chromatic_s))
            if achromatic_s and chromatic_s
            else None
        ),
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
