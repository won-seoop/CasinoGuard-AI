"""EXP-031: FC-012 "구간 단위 낮은 신뢰도 표시"로의 설계 전환 (지침 19, 22).

Roadmap(01. Architecture & Roadmap) 다음 Action 3번: EXP-026(White Balance)/EXP-028
(Segmentation Mask)/EXP-030(Gamma Correction, self/background 둘 다) 네 가지
pixel-level 보정이 모두 FC-012(검정 옷이 strong_light/cool_cast에서 blue로 오분류)를
부분적으로만 개선했다. EXP-030 Decision은 다음 후보를 "구간 단위 낮은 신뢰도 표시로의
설계 전환"으로 명시했다 - 틀린 색상을 고치려 하지 말고, 색상을 신뢰할 수 없는 구간
자체를 표시하자는 것이다.

이 실험은 그 신호를 Frame 전체(배경+person 전부 포함, crop이 아님)의 HSV 평균 밝기로
설계한다. EXP-030이 기각한 crop 자체 밝기(adaptive_gamma_correct)의 confound는
"crop 밝기 = 옷 색상(albedo) + 조명 노출이 섞인 신호"였다 - Frame 전체의 평균 밝기는
한 사람의 옷 색상에 좌우되지 않으므로 이 confound가 생기지 않는다(이 실험으로 직접
검증).

비교 대상:
- Alt A: 일반적인 Auto-Exposure 휴리스틱(이 Dataset을 보지 않고 흔히 쓰는 값, V<85
  "underexposed"/V>170 "overexposed" - 사진 Zone System에서 중간 회색 128을 중심으로
  대략 ±1.5 Zone 폭).
- Alt B: 이 Dataset의 실측 분포로 Threshold를 정한다(지침 22 Crowd Analysis와 동일
  원칙 - "Threshold는 임의로 정하지 않고, 실제 Dataset의 Distribution을 확인한 뒤
  정의한다"). normal 조건에서 측정한 Frame 평균 밝기의 P10/P90을 Threshold로 쓴다.
- Alt C: Alt B(노출 편차)만으로는 FC-012의 또 다른 축(warm_cast/cool_cast, 채널
  Cast)을 전혀 겨냥하지 못하므로, frame_channel_cast_deviation(채널 평균이 중립
  회색에서 벗어난 정도, normal 조건 분포의 P90으로 역시 Dataset 기반 Threshold)을
  추가해 "노출 편차 OR 채널 Cast 편차"로 Flag 범위를 넓힌다.

측정: 세 Threshold 세트 각각에 대해 flagged_rate(얼마나 많은 구간을 저신뢰로
표시하는가)와 unflagged 부분집합의 Accuracy(Overall, GT=black)를 Baseline(플래그
없음, 전부 신뢰)과 비교한다. 조건별 flagged_rate도 측정해 "노출 문제(low_light/
strong_light)만 겨냥하고 Cast 문제(warm_cast/cool_cast)는 건드리지 않는다"는 설계
의도가 실제로 지켜지는지 확인한다.

실행: python scripts/run_exp031_attribute_confidence_flag.py
출력: results/EXP-031/{predictions.csv, summary.json}
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
    classify_exposure_level,
    classify_person_attributes,
    frame_channel_cast_deviation,
    frame_mean_brightness,
)

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
RESULTS_DIR = Path("results/EXP-031")
MIN_BOX_AREA = 120 * 250
LIGHTING_CONDITIONS = ["normal", "low_light", "strong_light", "warm_cast", "cool_cast"]

# EXP-025/026/027/028/030과 완전히 동일한 coco128 n=21 Ground Truth.
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

# EXP-030과 동일한 Auto-Exposure 공식 기반 heuristic. 사진 Zone System은 중간 회색을
# Zone V(=128/255≈0.5)에 두고 보통 ±1.5 Zone(각 Zone은 2배/0.5배 밝기 차)을 "정상 노출
# 범위"로 본다 - 128*0.5^1.5≈45, 128*2^1.5≈362(>255이므로 clip)에 가깝지만, 이 실험은
# 더 널리 쓰이는 단순 값(OpenCV 튜토리얼/실무에서 "어둡다/밝다" 판정에 흔히 쓰는 절대
# V 값)을 그대로 사용해 "이 Dataset을 보지 않고 고른 값"이라는 Alt A의 성격을 유지한다.
ALT_A_LOW, ALT_A_HIGH = 85.0, 170.0


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027/028/030과 완전히 동일한 합성 조명 변형."""
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


def detect_person_crops_and_full_images() -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    """crop(색상 분류용)과 원본 전체 이미지 경로(Frame 전체 밝기 측정용)를 함께 반환한다.
    여러 crop이 같은 원본 이미지를 공유할 수 있다(예: 000000000328_0/1/2)."""
    model = YOLO("yolo11n.pt")
    crops: dict[str, np.ndarray] = {}
    full_image_paths: dict[str, Path] = {}
    for img_path in sorted(COCO128_IMAGES.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = model.predict(img, classes=[0], conf=0.4, verbose=False)[0]
        stem = img_path.stem
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue
            name = f"{stem}_{i}"
            crops[name] = img[y1:y2, x1:x2]
            full_image_paths[name] = img_path
    return crops, full_image_paths


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    crops, full_image_paths = detect_person_crops_and_full_images()
    print(f"detected {len(crops)} person crops (area >= {MIN_BOX_AREA}px) from coco128")

    missing = [n for n in GROUND_TRUTH if n not in crops]
    if missing:
        raise RuntimeError(f"GT names missing from detected crops (re-run detection?): {missing}")

    full_images: dict[Path, np.ndarray] = {}
    for path in full_image_paths.values():
        if path not in full_images:
            full_images[path] = cv2.imread(str(path))

    # 1단계: normal 조건에서 Frame 전체 평균 밝기 분포를 측정해 Alt B(Dataset 기반)
    # Threshold를 정한다. GT 21개가 가리키는 "고유 이미지" 기준(같은 이미지를 중복
    # 세지 않음 - 사람이 여러 명인 이미지가 그 이미지의 밝기를 여러 번 세면 안 된다).
    unique_images = sorted(set(full_image_paths[n] for n in GROUND_TRUTH))
    normal_brightness = [frame_mean_brightness(full_images[p]) for p in unique_images]
    normal_cast = [frame_channel_cast_deviation(full_images[p]) for p in unique_images]
    p10, p90 = np.percentile(normal_brightness, [10, 90])
    alt_b_low, alt_b_high = round(float(p10), 2), round(float(p90), 2)
    cast_p90 = round(float(np.percentile(normal_cast, 90)), 4)
    print(
        f"normal-condition frame brightness (n={len(unique_images)} unique images): "
        f"mean={np.mean(normal_brightness):.2f} std={np.std(normal_brightness):.2f} "
        f"min={np.min(normal_brightness):.2f} max={np.max(normal_brightness):.2f} "
        f"P10={alt_b_low} P90={alt_b_high}"
    )
    print(
        f"normal-condition frame cast deviation: mean={np.mean(normal_cast):.4f} "
        f"std={np.std(normal_cast):.4f} P90={cast_p90}"
    )

    threshold_sets = {
        "alt_a_fixed_heuristic": (ALT_A_LOW, ALT_A_HIGH),
        "alt_b_dataset_calibrated": (alt_b_low, alt_b_high),
    }

    rows = []
    for name, gt in GROUND_TRUTH.items():
        crop = crops[name]
        full_img = full_images[full_image_paths[name]]
        for condition in LIGHTING_CONDITIONS:
            lit_crop = apply_lighting(crop, condition)
            lit_full = apply_lighting(full_img, condition)
            mean_v = frame_mean_brightness(lit_full)
            cast_dev = frame_channel_cast_deviation(lit_full)
            pred = classify_person_attributes(lit_crop, method="b")
            row = {
                "name": name,
                "condition": condition,
                "frame_mean_v": round(mean_v, 2),
                "frame_cast_deviation": round(cast_dev, 4),
                "gt_upper": gt["upper"],
                "gt_lower": gt["lower"],
                "pred_upper": pred["upper"],
                "pred_lower": pred["lower"],
                "upper_correct": int(pred["upper"] == gt["upper"]),
                "lower_correct": int(pred["lower"] == gt["lower"]),
            }
            for set_name, (low, high) in threshold_sets.items():
                level = classify_exposure_level(mean_v, low, high)
                row[f"{set_name}_level"] = level
                row[f"{set_name}_flagged"] = int(level != "normal")
            # Alt C: Alt B(노출)의 Flag 범위에 Cast 편차(P90 초과)를 OR로 추가.
            row["alt_c_combined_flagged"] = int(
                row["alt_b_dataset_calibrated_flagged"] or (cast_dev > cast_p90)
            )
            rows.append(row)

    with open(RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def accuracy(subset: list[dict]) -> dict:
        total = len(subset) * 2
        if total == 0:
            return {"accuracy": None, "total_cells": 0}
        correct = sum(r["upper_correct"] + r["lower_correct"] for r in subset)
        black_cells = []
        for r in subset:
            if r["gt_upper"] == "black":
                black_cells.append(r["upper_correct"])
            if r["gt_lower"] == "black":
                black_cells.append(r["lower_correct"])
        return {
            "accuracy": round(correct / total, 4),
            "total_cells": total,
            "gt_black_accuracy": round(sum(black_cells) / len(black_cells), 4) if black_cells else None,
            "gt_black_total_cells": len(black_cells),
        }

    summary: dict = {
        "n_ground_truth_crops": len(GROUND_TRUTH),
        "n_unique_images": len(unique_images),
        "normal_condition_frame_brightness": {
            "mean": round(float(np.mean(normal_brightness)), 2),
            "std": round(float(np.std(normal_brightness)), 2),
            "min": round(float(np.min(normal_brightness)), 2),
            "max": round(float(np.max(normal_brightness)), 2),
            "p10": alt_b_low,
            "p90": alt_b_high,
        },
        "thresholds": {name: {"low": low, "high": high} for name, (low, high) in threshold_sets.items()},
        "cast_threshold_p90": cast_p90,
        "baseline_no_flagging": accuracy(rows),
    }

    flag_variants = ["alt_a_fixed_heuristic", "alt_b_dataset_calibrated", "alt_c_combined"]
    for variant in flag_variants:
        flagged_key = f"{variant}_flagged"
        flagged = [r for r in rows if r[flagged_key]]
        unflagged = [r for r in rows if not r[flagged_key]]
        by_condition_flagged_rate = {}
        by_condition_unflagged_composition = {}
        for condition in LIGHTING_CONDITIONS:
            cond_rows = [r for r in rows if r["condition"] == condition]
            by_condition_flagged_rate[condition] = round(
                sum(r[flagged_key] for r in cond_rows) / len(cond_rows), 4
            )
            by_condition_unflagged_composition[condition] = sum(
                1 for r in unflagged if r["condition"] == condition
            )
        entry = {
            "flagged_rate": round(len(flagged) / len(rows), 4),
            "flagged_rate_by_condition": by_condition_flagged_rate,
            "unflagged_subset": accuracy(unflagged),
            "flagged_subset": accuracy(flagged),
            "unflagged_subset_condition_composition_cells": by_condition_unflagged_composition,
        }
        if variant != "alt_c_combined":
            level_key = f"{variant}_level"
            entry["level_counts"] = {
                lvl: sum(1 for r in rows if r[level_key] == lvl) for lvl in ("normal", "low_light", "strong_light")
            }
        summary[variant] = entry

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
