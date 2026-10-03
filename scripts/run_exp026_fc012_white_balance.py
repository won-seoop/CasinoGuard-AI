"""EXP-026: FC-012 (Attribute 색상 분류, 검정 옷이 strong_light/cool_cast에서 blue로
오분류) 재현 + 대안 비교 (지침 19, 40).

EXP-025가 남긴 FC-012: 대안 B(region_dominant_color)로도 검정 옷이 strong_light/
cool_cast 합성 조명에서 반복적으로 blue로 오분류됐다. 이 실험은 그 원인을 실측하고
두 가지 대안을 비교한다.

- Baseline: EXP-025의 production 방식(method="b", White Balance 없음).
- 대안 A: Hue 분류 이전에 "무채색 여부"를 raw BGR 채널 spread(max-min)로 먼저
  판정해 무채색이면 Hue를 보지 않고 V만으로 black/gray/white를 정한다.
- 대안 B(채택): Gray World White Balance를 Hue 분류 이전에 적용해 채널 Cast 자체를
  줄인다 (`white_balance_gray_world`, 이미 `classify_person_attributes(method="b")`에
  production 기본값으로 반영됨).

새로 YOLO를 돌리지 않고 EXP-025가 이미 저장한 `results/EXP-025/lighting_variants/`의
동일 crop+조명 변형 이미지와 동일 Ground Truth를 재사용한다 (지침 30 Regression Test
취지: 동일 Dataset/환경에서 Before/After를 비교해야 한다).

실행: python scripts/run_exp026_fc012_white_balance.py
출력: results/EXP-026/{predictions.csv, summary.json}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attributes.color import (  # noqa: E402
    _is_skin_like,
    _to_hsv,
    classify_hsv_pixel,
    classify_person_attributes,
    region_dominant_color,
    split_upper_lower,
)

VARIANTS_DIR = Path("results/EXP-025/lighting_variants")
RESULTS_DIR = Path("results/EXP-026")

GROUND_TRUTH = {
    "zidane_0": {"upper": "black", "lower": "black"},
    "zidane_1": {"upper": "black", "lower": "black"},
    "bus_0": {"upper": "black", "lower": "black"},
    "bus_1": {"upper": "white", "lower": "blue"},
    "bus_2": {"upper": "black", "lower": "blue"},
}
LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

SPLIT_KWARGS = dict(head_skip_ratio=0.22, upper_end_ratio=0.55, foot_skip_ratio=0.08, side_margin_ratio=0.12)

# 대안 A에서 "최선"으로 측정된 threshold (grid search 8/12/16 모두 0.42로 동률,
# 가장 보수적인 값을 채택했다 - experiment.md Analysis 참고).
ALT_A_SPREAD_THRESHOLD = 12


def region_dominant_color_achromatic_shortcut(
    region_bgr: np.ndarray,
    spread_thresh: int = ALT_A_SPREAD_THRESHOLD,
    sat_min: int = 25,
    val_min: int = 35,
    val_max: int = 245,
    hue_bin_width: int = 10,
    min_filtered_fraction: float = 0.3,
) -> str:
    """대안 A: Hue를 보기 전에 raw BGR 채널 spread(max-min)로 무채색 여부를 먼저
    판정한다. spread가 작으면(색 채널 간 절대 차이가 작으면) 이미 무채색이라고
    보고 V만으로 black/gray/white를 정한다 (HSV S가 아니라 raw BGR spread를 쓰는
    이유: S는 V가 작을 때 절대 차이를 과장해 보여준다 - experiment.md 원인 분석).
    """
    if region_bgr.size == 0:
        return "gray"
    hsv = _to_hsv(region_bgr).reshape(-1, 3).astype(np.int32)
    bgr = region_bgr.reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mask = (s >= sat_min) & (v >= val_min) & (v <= val_max)
    mask = mask & ~_is_skin_like(h, s, v)
    filtered_hsv = hsv[mask]
    filtered_bgr = bgr[mask]
    if filtered_hsv.shape[0] < min_filtered_fraction * hsv.shape[0]:
        filtered_hsv = hsv
        filtered_bgr = bgr
    spread = filtered_bgr.max(axis=1) - filtered_bgr.min(axis=1)
    if np.median(spread) < spread_thresh:
        return classify_hsv_pixel(0.0, 0.0, float(np.median(filtered_bgr.max(axis=1))))
    bins = (filtered_hsv[:, 0] // hue_bin_width).astype(np.int32)
    values, counts = np.unique(bins, return_counts=True)
    dominant_bin = values[np.argmax(counts)]
    in_bin = filtered_hsv[bins == dominant_bin]
    med_h, med_s, med_v = np.median(in_bin, axis=0)
    return classify_hsv_pixel(float(med_h), float(med_s), float(med_v))


def predict_baseline(crop_bgr: np.ndarray) -> dict[str, str]:
    upper, lower = split_upper_lower(crop_bgr, **SPLIT_KWARGS)
    return {"upper": region_dominant_color(upper), "lower": region_dominant_color(lower)}


def predict_alt_a(crop_bgr: np.ndarray) -> dict[str, str]:
    upper, lower = split_upper_lower(crop_bgr, **SPLIT_KWARGS)
    return {
        "upper": region_dominant_color_achromatic_shortcut(upper),
        "lower": region_dominant_color_achromatic_shortcut(lower),
    }


def predict_alt_b(crop_bgr: np.ndarray) -> dict[str, str]:
    return classify_person_attributes(crop_bgr, method="b_wb")


METHODS = {"baseline": predict_baseline, "alt_a": predict_alt_a, "alt_b": predict_alt_b}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, gt in GROUND_TRUTH.items():
        for condition in LIGHTING_CONDITIONS:
            variant_path = VARIANTS_DIR / f"{name}_{condition}.jpg"
            variant = cv2.imread(str(variant_path))
            if variant is None:
                raise FileNotFoundError(
                    f"{variant_path} 없음 - 먼저 scripts/run_exp025_attribute_color.py를 실행해야 한다."
                )
            for method_name, predict_fn in METHODS.items():
                pred = predict_fn(variant)
                upper_correct = pred["upper"] == gt["upper"]
                lower_correct = pred["lower"] == gt["lower"]
                rows.append(
                    {
                        "crop": name,
                        "condition": condition,
                        "method": method_name,
                        "gt_upper": gt["upper"],
                        "gt_lower": gt["lower"],
                        "pred_upper": pred["upper"],
                        "pred_lower": pred["lower"],
                        "upper_correct": int(upper_correct),
                        "lower_correct": int(lower_correct),
                    }
                )

    csv_path = RESULTS_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict = {"per_method_condition": {}, "per_method_overall": {}}
    for method_name in METHODS:
        method_rows = [r for r in rows if r["method"] == method_name]
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in method_rows if r["condition"] == condition]
            correct = sum(r["upper_correct"] for r in cond_rows) + sum(r["lower_correct"] for r in cond_rows)
            acc = correct / (len(cond_rows) * 2)
            summary["per_method_condition"].setdefault(method_name, {})[condition] = round(acc, 4)
        correct = sum(r["upper_correct"] for r in method_rows) + sum(r["lower_correct"] for r in method_rows)
        summary["per_method_overall"][method_name] = round(correct / (len(method_rows) * 2), 4)

    for method_name in METHODS:
        non_normal = [c for c in LIGHTING_CONDITIONS if c != "normal"]
        vals = [summary["per_method_condition"][method_name][c] for c in non_normal]
        summary.setdefault("lighting_stability_accuracy", {})[method_name] = round(sum(vals) / len(vals), 4)

    # FC-012 직접 재현: GT가 black인 (crop, region) 셀만 모아 정확도와 blue 오분류율을
    # 전체 조건 + 조건별로 나눠서 본다. 전체 평균만 보면 "다른 색 셀이 좋아진 것"과
    # "이 실험이 겨냥한 검정 셀 자체가 좋아진 것"을 구분할 수 없다(지침 36).
    black_gt_cells = []
    for name, gt in GROUND_TRUTH.items():
        for region in ("upper", "lower"):
            if gt[region] == "black":
                black_gt_cells.append((name, region))
    for method_name in METHODS:
        method_rows = [r for r in rows if r["method"] == method_name]
        overall_correct = 0
        overall_blue = 0
        overall_total = 0
        per_condition: dict = {}
        for condition in LIGHTING_CONDITIONS:
            correct = 0
            blue = 0
            total = 0
            for r in method_rows:
                if r["condition"] != condition:
                    continue
                for region in ("upper", "lower"):
                    if (r["crop"], region) not in black_gt_cells:
                        continue
                    total += 1
                    if r[f"{region}_correct"] == 1:
                        correct += 1
                    if r[f"pred_{region}"] == "blue":
                        blue += 1
            per_condition[condition] = {
                "accuracy": round(correct / total, 4),
                "blue_rate": round(blue / total, 4),
            }
            overall_correct += correct
            overall_blue += blue
            overall_total += total
        summary.setdefault("fc012_black_gt_cells", {})[method_name] = {
            "overall_accuracy": round(overall_correct / overall_total, 4),
            "overall_blue_rate": round(overall_blue / overall_total, 4),
            "per_condition": per_condition,
        }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nrows: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
