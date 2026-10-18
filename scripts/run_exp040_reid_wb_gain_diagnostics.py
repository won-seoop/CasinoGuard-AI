"""EXP-040: Person Re-ID FC-013(a) - Gamma-Background+WB가 Baseline(Color Histogram)의
low_light/strong_light를 추가로 악화시킨 원인을 Gray World 게인 값 직접 로깅으로 규명
(지침 18).

Roadmap(01. Architecture & Roadmap) EXP-037 Next Action 7-(a''), EXP-038/039 Next Action
1번에 이월된 과제: EXP-038에서 Gamma-Background+WB를 적용한 뒤 Baseline(Color Histogram)
의 low_light Top-1이 0.3556->0.2222(-13.33pp), strong_light Top-1이 0.2889->0.2222
(-6.67pp)로 추가 악화됐다. EXP-039(Mask 기반 WB)도 이 둘을 고치지 못했다(low_light
0.2222->0.2667만 소폭, strong_light는 0.2222로 그대로) - FC-014(Occlusion)와는 다른
메커니즘임을 시사한다. 지금까지 이 악화의 원인은 "crop이 거의 단색에 가까운 상태에서
Gray World 게인이 극단적으로 작동할 수 있다"는 "추정"으로만 기록되고 실측 확인되지
않았다(EXP-038 Analysis, EXP-039 Next Action 1).

이 실험은 white_balance_gray_world_diagnostic()(신규, src/attributes/color.py)으로
Gamma-Background 보정 직후(WB 직전) 각 variant의 실제 채널 평균/게인/clip 발동 여부를
로깅해 그 추정을 확인/반증한다. 원인이 확인되면, 그 진단 신호를 직접 이용하는 최소 두
가지 대안(Alt A: 더 보수적인 Clip 범위, Alt B: 게인이 불안정한 영역에서 WB를 건너뛰는
Confidence Gate)을 실제로 비교해 다음 production 후보를 정한다.

실행: python scripts/run_exp040_reid_wb_gain_diagnostics.py
출력: results/EXP-040/{gain_diagnostics.csv, gain_diagnostics_summary.json,
      alternative_comparison.json}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attributes.color import (  # noqa: E402
    adaptive_gamma_correct_from_background,
    apply_gamma,
    estimate_gamma_from_reference,
    white_balance_gray_world,
    white_balance_gray_world_diagnostic,
)
from reid.embedding import YoloBackboneEmbedder, color_histogram_embedding  # noqa: E402
from reid.matching import average_precision, evaluate_identification, rank_gallery_fused, topk_hit  # noqa: E402

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
CROPS_DIR = ROOT / "results" / "EXP-027" / "crops"
RESULTS_DIR = ROOT / "results" / "EXP-040"
RNG_SEED = 42
MIN_BOX_AREA = 120 * 250
PERSON_CLASS_ID = 0

LIGHTING_CONDITIONS = ["low_light", "strong_light", "warm_cast", "cool_cast"]
ALL_VARIANTS = [*LIGHTING_CONDITIONS, "occlusion", "bbox_jitter", "flip"]

# Alt A: 더 보수적인(좁은) Clip 범위 - EXP-026/037/038이 쓰던 기본값(0.3~3.0)보다
# 극단적 게인을 더 강하게 억제한다.
TIGHT_GAIN_MIN, TIGHT_GAIN_MAX = 0.5, 2.0

# Alt B: Confidence Gate - 감마 보정 직후 overall_mean이 이 범위를 벗어나면(거의
# 흑/백에 가까워 Gray World 가정이 깨지는 영역) WB를 건너뛰고 Gamma만 적용한다.
GATE_MEAN_MIN, GATE_MEAN_MAX = 40.0, 215.0


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027/030/036~039와 완전히 동일한 합성 조명 변형(재현성을 위해 유지)."""
    img = img_bgr.astype(np.float32)
    if condition == "low_light":
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


def apply_occlusion(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    ow = int(w * rng.uniform(0.4, 0.6))
    oh = int(h * rng.uniform(0.3, 0.5))
    x0 = int(rng.uniform(0, max(1, w - ow)))
    y0 = int(rng.uniform(0, max(1, h - oh)))
    out[y0 : y0 + oh, x0 : x0 + ow] = (120, 120, 120)
    return out


def apply_bbox_jitter(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    top = int(h * rng.uniform(0.0, 0.15))
    bottom = int(h * rng.uniform(0.0, 0.15))
    left = int(w * rng.uniform(0.0, 0.15))
    right = int(w * rng.uniform(0.0, 0.15))
    cropped = img_bgr[top : h - bottom, left : w - right]
    if cropped.size == 0:
        return img_bgr
    return cv2.resize(cropped, (w, h))


def apply_flip(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(img_bgr, 1)


def background_sample(full_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = full_img.shape[:2]
    x1, y1, x2, y2 = bbox
    mask = np.ones((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = False
    pixels = full_img[mask]
    return pixels.reshape(-1, 1, 3)


def detect_person_crops_and_bboxes() -> tuple[dict[str, np.ndarray], dict[str, tuple[Path, tuple[int, int, int, int]]]]:
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


def build_queries(gallery: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, dict]:
    """EXP-036~039와 완전히 동일한 Query 구성(같은 RNG 시드 42로 동일 호출 순서 유지)."""
    queries: dict[str, dict] = {}
    for gallery_id, img in gallery.items():
        for condition in LIGHTING_CONDITIONS:
            qid = f"{gallery_id}__{condition}"
            queries[qid] = {"image": apply_lighting(img, condition), "true_id": gallery_id, "variant": condition}
        queries[f"{gallery_id}__occlusion"] = {
            "image": apply_occlusion(img, rng),
            "true_id": gallery_id,
            "variant": "occlusion",
        }
        queries[f"{gallery_id}__bbox_jitter"] = {
            "image": apply_bbox_jitter(img, rng),
            "true_id": gallery_id,
            "variant": "bbox_jitter",
        }
        queries[f"{gallery_id}__flip"] = {
            "image": apply_flip(img),
            "true_id": gallery_id,
            "variant": "flip",
        }
    return queries


# ---------------------------------------------------------------------------
# Part 1: Gray World 게인 진단 로깅
# ---------------------------------------------------------------------------


def part1_gain_diagnostics(
    gallery: dict[str, np.ndarray],
    refs: dict[str, tuple[Path, tuple[int, int, int, int]]],
    full_images: dict[Path, np.ndarray],
) -> list[dict]:
    """각 gallery crop에 대해 (변형 없음 + lighting 4종) x Gamma-Background 보정 후,
    WB 직전 Gray World 게인을 로깅한다(EXP-038 unmasked 파이프라인과 동일한 감마
    보정 - gamma_background_then_white_balance의 첫 단계를 그대로 재현)."""
    rows: list[dict] = []
    for gid, img in gallery.items():
        img_path, bbox = refs[gid]
        full_img = full_images[img_path]
        groups = {"gallery": img}
        for condition in LIGHTING_CONDITIONS:
            groups[condition] = apply_lighting(img, condition)

        for variant, variant_img in groups.items():
            lit_full = apply_lighting(full_img, variant) if variant in LIGHTING_CONDITIONS else full_img
            bg_pixels = background_sample(lit_full, bbox)
            gamma = estimate_gamma_from_reference(bg_pixels)
            gamma_corrected = apply_gamma(variant_img, gamma)
            diag = white_balance_gray_world_diagnostic(gamma_corrected)
            gains = diag["gains_after_clip"]
            rows.append(
                {
                    "gallery_id": gid,
                    "variant": variant,
                    "gamma": gamma,
                    "post_gamma_overall_mean": diag["overall_mean"],
                    "gain_b": gains[0],
                    "gain_g": gains[1],
                    "gain_r": gains[2],
                    "max_abs_gain_deviation": max(abs(g - 1.0) for g in gains),
                    "n_channels_clipped": diag["n_channels_clipped"],
                }
            )
    return rows


def summarize_part1(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    variants = sorted({r["variant"] for r in rows})
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        n = len(subset)
        n_clipped_any = sum(1 for r in subset if r["n_channels_clipped"] > 0)
        summary[variant] = {
            "n": n,
            "mean_post_gamma_overall_mean": float(np.mean([r["post_gamma_overall_mean"] for r in subset])),
            "mean_max_abs_gain_deviation": float(np.mean([r["max_abs_gain_deviation"] for r in subset])),
            "pct_crops_with_any_clipped_channel": n_clipped_any / n,
        }
    return summary


# ---------------------------------------------------------------------------
# Part 2: 대안 비교 (Alt A: 보수적 Clip 범위 / Alt B: Confidence Gate)
# ---------------------------------------------------------------------------


def gamma_background_then_white_balance_tight_clip(crop_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    """Alt A: EXP-038의 gamma_background_then_white_balance와 동일하지만 WB의
    gain_min/gain_max를 (0.3,3.0)에서 (0.5,2.0)으로 좁힌다 - Part 1 진단에서 발견된
    극단적 게인 자체를 더 강하게 억제하려는 가장 단순한 대안."""
    gamma_corrected = adaptive_gamma_correct_from_background(crop_bgr, background_bgr)
    return white_balance_gray_world(gamma_corrected, gain_min=TIGHT_GAIN_MIN, gain_max=TIGHT_GAIN_MAX)


def gamma_background_then_white_balance_confidence_gate(crop_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    """Alt B: Part 1 진단 신호(감마 보정 직후 overall_mean)를 직접 Gate로 사용한다 -
    Gray World 가정이 신뢰할 수 없어지는 영역(거의 흑/백에 가까운 crop)에서는 WB를
    건너뛰고 Gamma만 적용, 그 범위를 벗어나지 않을 때만 WB를 추가한다."""
    gamma_corrected = adaptive_gamma_correct_from_background(crop_bgr, background_bgr)
    diag = white_balance_gray_world_diagnostic(gamma_corrected)
    if diag["overall_mean"] < GATE_MEAN_MIN or diag["overall_mean"] > GATE_MEAN_MAX:
        return gamma_corrected
    return white_balance_gray_world(gamma_corrected)


def correct_with_pipeline(
    pipeline_fn,
    img_bgr: np.ndarray,
    gallery_id: str,
    variant: str | None,
    refs: dict[str, tuple[Path, tuple[int, int, int, int]]],
    full_images: dict[Path, np.ndarray],
) -> np.ndarray:
    img_path, bbox = refs[gallery_id]
    full_img = full_images[img_path]
    lit_full = apply_lighting(full_img, variant) if variant in LIGHTING_CONDITIONS else full_img
    bg_pixels = background_sample(lit_full, bbox)
    return pipeline_fn(img_bgr, bg_pixels)


def evaluate_method(
    method_name: str,
    embed_fn,
    gallery_embeddings_input: dict[str, np.ndarray],
    query_embeddings_input: dict[str, np.ndarray],
    query_true_ids: dict[str, str],
) -> dict:
    gallery_ids = list(gallery_embeddings_input.keys())
    gallery_embeddings = np.stack([embed_fn(gallery_embeddings_input[gid]) for gid in gallery_ids])
    query_embeddings = {qid: embed_fn(img) for qid, img in query_embeddings_input.items()}
    overall = evaluate_identification(query_embeddings, query_true_ids, gallery_embeddings, gallery_ids)
    by_variant: dict[str, dict] = {}
    variants = sorted({v for v in (qid.split("__", 1)[1] for qid in query_embeddings)})
    for variant in variants:
        subset_ids = {qid: emb for qid, emb in query_embeddings.items() if qid.endswith(f"__{variant}")}
        subset_true = {qid: query_true_ids[qid] for qid in subset_ids}
        by_variant[variant] = evaluate_identification(subset_ids, subset_true, gallery_embeddings, gallery_ids)
    return {
        "method": method_name,
        "overall": {k: v for k, v in overall.items() if k != "per_query"},
        "by_variant": {v: {k: val for k, val in r.items() if k != "per_query"} for v, r in by_variant.items()},
    }


def evaluate_fusion_images(
    gallery_images: dict[str, np.ndarray],
    query_images: dict[str, dict],
    baseline_embed_fn,
    candidate_embed_fn,
    weight_baseline: float = 0.5,
    ks: tuple[int, ...] = (1, 5),
) -> dict:
    gallery_ids = list(gallery_images.keys())
    baseline_gallery = np.stack([baseline_embed_fn(gallery_images[gid]) for gid in gallery_ids])
    candidate_gallery = np.stack([candidate_embed_fn(gallery_images[gid]) for gid in gallery_ids])

    topk_counts = dict.fromkeys(ks, 0)
    ap_sum = 0.0
    by_variant_counts: dict[str, dict] = {}
    for qid, q in query_images.items():
        variant = q["variant"]
        true_id = q["true_id"]
        q_baseline = baseline_embed_fn(q["image"])
        q_candidate = candidate_embed_fn(q["image"])
        ranked = rank_gallery_fused(
            [q_baseline, q_candidate],
            [baseline_gallery, candidate_gallery],
            [weight_baseline, 1.0 - weight_baseline],
            gallery_ids,
        )
        ranked_ids = [gid for gid, _ in ranked]
        ap = average_precision(ranked_ids, {true_id})
        ap_sum += ap
        bucket = by_variant_counts.setdefault(variant, {"topk": dict.fromkeys(ks, 0), "ap_sum": 0.0, "n": 0})
        bucket["ap_sum"] += ap
        bucket["n"] += 1
        for k in ks:
            if topk_hit(ranked_ids, true_id, k):
                topk_counts[k] += 1
                bucket["topk"][k] += 1

    n = len(query_images)
    overall = {"n_queries": n, "mAP": ap_sum / n, **{f"top{k}": topk_counts[k] / n for k in ks}}
    by_variant = {
        v: {"n_queries": b["n"], "mAP": b["ap_sum"] / b["n"], **{f"top{k}": b["topk"][k] / b["n"] for k in ks}}
        for v, b in by_variant_counts.items()
    }
    return {"overall": overall, "by_variant": by_variant}


def part2_alternative_comparison(
    gallery: dict[str, np.ndarray],
    queries: dict[str, dict],
    refs: dict[str, tuple[Path, tuple[int, int, int, int]]],
    full_images: dict[Path, np.ndarray],
) -> dict:
    query_true_ids = {qid: q["true_id"] for qid, q in queries.items()}
    embedder = YoloBackboneEmbedder(weights_path="yolo11n-cls.pt")

    pipelines = {
        "tight_clip": gamma_background_then_white_balance_tight_clip,
        "confidence_gate": gamma_background_then_white_balance_confidence_gate,
    }

    results: dict[str, dict] = {}
    for label, pipeline_fn in pipelines.items():
        gallery_corrected = {gid: correct_with_pipeline(pipeline_fn, img, gid, None, refs, full_images) for gid, img in gallery.items()}
        query_images_corrected = {
            qid: correct_with_pipeline(pipeline_fn, q["image"], q["true_id"], q["variant"], refs, full_images)
            for qid, q in queries.items()
        }
        queries_corrected = {
            qid: {"image": query_images_corrected[qid], "true_id": q["true_id"], "variant": q["variant"]}
            for qid, q in queries.items()
        }

        baseline_result = evaluate_method(
            f"baseline_color_histogram_{label}", color_histogram_embedding, gallery_corrected, query_images_corrected, query_true_ids
        )
        candidate_result = evaluate_method(
            f"candidate_yolo_backbone_{label}", embedder.embed, gallery_corrected, query_images_corrected, query_true_ids
        )
        fusion_result = evaluate_fusion_images(gallery_corrected, queries_corrected, color_histogram_embedding, embedder.embed)

        print(
            f"[{label}] Baseline Top-1={baseline_result['overall']['top1']:.4f} "
            f"low_light={baseline_result['by_variant']['low_light']['top1']:.4f} "
            f"strong_light={baseline_result['by_variant']['strong_light']['top1']:.4f} "
            f"warm_cast={baseline_result['by_variant']['warm_cast']['top1']:.4f} "
            f"cool_cast={baseline_result['by_variant']['cool_cast']['top1']:.4f} "
            f"occlusion={baseline_result['by_variant']['occlusion']['top1']:.4f}"
        )
        print(
            f"[{label}] Candidate Top-1={candidate_result['overall']['top1']:.4f} "
            f"low_light={candidate_result['by_variant']['low_light']['top1']:.4f} "
            f"strong_light={candidate_result['by_variant']['strong_light']['top1']:.4f}"
        )
        print(f"[{label}] Fusion Top-1={fusion_result['overall']['top1']:.4f}")

        results[label] = {
            "baseline_color_histogram": baseline_result,
            "candidate_yolo11n_cls_backbone": candidate_result,
            "fusion_0.5_0.5": fusion_result,
        }
    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng_for_queries = np.random.default_rng(RNG_SEED)

    print("coco128 재탐지 중 (배경 추출용 bbox 확보)...")
    _, refs = detect_person_crops_and_bboxes()

    gallery: dict[str, np.ndarray] = {}
    for path in sorted(CROPS_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            continue
        gallery[path.stem] = img

    missing = set(gallery) - set(refs)
    if missing:
        raise RuntimeError(f"재탐지 결과가 EXP-027/036~039 Gallery와 불일치(배경 bbox 없음): {missing}")
    print(f"Gallery: {len(gallery)}명, 재탐지로 bbox 확보 완료 (EXP-027/036~039와 동일 집합 확인됨)")

    full_images: dict[Path, np.ndarray] = {}
    for img_path, _ in refs.values():
        if img_path not in full_images:
            full_images[img_path] = cv2.imread(str(img_path))

    print("\n=== Part 1: Gray World 게인 진단 로깅 ===")
    rows = part1_gain_diagnostics(gallery, refs, full_images)
    with (RESULTS_DIR / "gain_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    part1_summary = summarize_part1(rows)
    for variant, s in part1_summary.items():
        print(
            f"[{variant}] n={s['n']} post_gamma_mean={s['mean_post_gamma_overall_mean']:.1f} "
            f"mean|gain-1|={s['mean_max_abs_gain_deviation']:.3f} "
            f"clip발동비율={s['pct_crops_with_any_clipped_channel']:.1%}"
        )
    (RESULTS_DIR / "gain_diagnostics_summary.json").write_text(json.dumps(part1_summary, indent=2, ensure_ascii=False))
    print(f"저장: {RESULTS_DIR / 'gain_diagnostics_summary.json'}")

    print("\n=== Part 2: 대안 비교 (Alt A: Tight Clip vs Alt B: Confidence Gate) ===")
    queries = build_queries(gallery, rng_for_queries)
    part2_results = part2_alternative_comparison(gallery, queries, refs, full_images)
    (RESULTS_DIR / "alternative_comparison.json").write_text(json.dumps(part2_results, indent=2, ensure_ascii=False))
    print(f"저장: {RESULTS_DIR / 'alternative_comparison.json'}")


if __name__ == "__main__":
    main()
