"""EXP-038: FC-013(b) Person Re-ID warm_cast/cool_cast 저하 - Gamma-Background+WB 결합 검증 (지침 18).

Roadmap(01. Architecture & Roadmap) EXP-037 Next Action 7-(a)(최우선): EXP-037에서
배경 기반 Gamma 보정(대안 B)이 FC-013의 저노출/과다노출(low_light/strong_light) 붕괴를
크게 완화했지만(strong_light Top-1 YOLO Backbone 0.0%->66.7%), warm_cast/cool_cast
(채널 Cast)는 오히려 소폭(2.2~8.9pp) 악화된 채로 남아 있었다. Attribute 분류(FC-012)에서
White Balance(EXP-026 대안 B, white_balance_gray_world)가 겨냥하는 원인이 바로 이
"채널 간 상대적 Cast"이므로, 이 실험은 EXP-037이 채택한 Gamma-Background에 White
Balance를 EXP-030의 b_wb_gamma와 동일한 순서(감마 먼저 -> WB 나중)로 추가 적용하면
warm_cast/cool_cast Top-1이 개선되는지 "적용 전에 먼저" 측정한다.

Hypothesis: Gamma(전역 노출)와 WB(채널 간 상대적 Cast)는 서로 다른 원인을 겨냥하므로
합쳐질 것이다 - warm_cast/cool_cast Top-1이 Gamma-Background 단독보다 개선되고,
이미 Gamma-Background가 크게 개선한 low_light/strong_light에는 새 Regression이
생기지 않을 것이다.

중요(EXP-031의 교훈 반복 방지 - "한 Task에서 검증된 보정이 다른 Task에서도 그대로
통할 것"이라는 가정 자체가 EXP-037에서 이미 한 번 부분적으로 반증됐다. Self-Gamma가
Attribute에서는 confound로 완전히 기각됐지만 Re-ID에서는 오히려 저노출에 더 나았다):
"평균 개선"만 보고 끝내지 않는다 - warm_cast/cool_cast를 직접 슬라이싱하고, Gamma 단독이
이미 개선한 low_light/strong_light와 기하 변형(occlusion/bbox_jitter/flip)의 회귀 여부도
반드시 함께 측정한다. 검증 전까지 production(src/reid, run_full_pipeline.py)에는 연결하지
않는다.

Gallery/Query Protocol: EXP-036/037과 완전히 동일(coco128 crop 45명 Gallery, RNG seed
42, 동일 합성 변형 순서) - 세 실험이 직접 비교 가능하도록 유지한다.

실행: python scripts/run_exp038_reid_wb_gamma_background.py
출력: results/EXP-038/{summary.json, per_query_predictions.csv}
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

from attributes.color import adaptive_gamma_correct_from_background, gamma_background_then_white_balance  # noqa: E402
from reid.embedding import YoloBackboneEmbedder, color_histogram_embedding  # noqa: E402
from reid.matching import average_precision, evaluate_identification, rank_gallery_fused, topk_hit  # noqa: E402

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
CROPS_DIR = ROOT / "results" / "EXP-027" / "crops"
RESULTS_DIR = ROOT / "results" / "EXP-038"
RNG_SEED = 42
MIN_BOX_AREA = 120 * 250

LIGHTING_CONDITIONS = ["low_light", "strong_light", "warm_cast", "cool_cast"]
ALL_VARIANTS = [*LIGHTING_CONDITIONS, "occlusion", "bbox_jitter", "flip"]


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027/030/036/037과 완전히 동일한 합성 조명 변형(재현성을 위해 동일 Config 유지)."""
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
    """EXP-036/037과 동일 - bbox 폭 40~60%, 높이 30~50%의 사각형을 중립 회색으로 덮는다."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    ow = int(w * rng.uniform(0.4, 0.6))
    oh = int(h * rng.uniform(0.3, 0.5))
    x0 = int(rng.uniform(0, max(1, w - ow)))
    y0 = int(rng.uniform(0, max(1, h - oh)))
    out[y0 : y0 + oh, x0 : x0 + ow] = (120, 120, 120)
    return out


def apply_bbox_jitter(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """EXP-036/037과 동일 - 가장자리를 5~15% 범위에서 비대칭으로 잘라낸다."""
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
    """EXP-036/037과 동일 - 수평 Flip."""
    return cv2.flip(img_bgr, 1)


def background_sample(full_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """EXP-030/037과 동일 - person bbox를 제외한 Frame 나머지 영역 픽셀을 (N,1,3)으로 반환."""
    h, w = full_img.shape[:2]
    x1, y1, x2, y2 = bbox
    mask = np.ones((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = False
    pixels = full_img[mask]
    return pixels.reshape(-1, 1, 3)


def detect_person_crops_and_bboxes() -> tuple[dict[str, np.ndarray], dict[str, tuple[Path, tuple[int, int, int, int]]]]:
    """EXP-030/037과 동일한 탐지 설정(yolo11n.pt, conf=0.4, MIN_BOX_AREA)으로 coco128을
    재탐지해 각 crop의 원본 이미지 경로+bbox를 함께 확보한다(배경 추출에 필요)."""
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
    """EXP-036/037과 동일한 Query 구성(같은 RNG 시드 42로 동일 호출 순서 유지 -> 세 실험이
    동일한 합성 왜곡에서 직접 비교 가능하도록 보장)."""
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


def correct_image(
    img_bgr: np.ndarray,
    gallery_id: str,
    variant: str | None,
    refs: dict[str, tuple[Path, tuple[int, int, int, int]]],
    full_images: dict[Path, np.ndarray],
    correction_fn,
) -> np.ndarray:
    """img_bgr(Gallery 원본 또는 Query 변형 이미지)에 배경 기반 보정(Gamma 단독 또는
    Gamma+WB)을 적용한다. EXP-037과 동일한 배경 추출 로직(variant가 조명 조건이면 배경도
    같은 조명 변형을 거친 뒤 샘플링, 기하 변형/Gallery는 원본 배경 그대로)."""
    img_path, bbox = refs[gallery_id]
    full_img = full_images[img_path]
    if variant in LIGHTING_CONDITIONS:
        lit_full = apply_lighting(full_img, variant)
    else:
        lit_full = full_img
    bg_pixels = background_sample(lit_full, bbox)
    return correction_fn(img_bgr, bg_pixels)


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
        "per_query": overall["per_query"],
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
    return {"method": "fusion_0.5_0.5", "overall": overall, "by_variant": by_variant}


def strip_per_query(result: dict) -> dict:
    return {"overall": result["overall"], "by_variant": result["by_variant"]}


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
        raise RuntimeError(f"재탐지 결과가 EXP-027/036/037 Gallery와 불일치(배경 bbox 없음): {missing}")
    print(f"Gallery: {len(gallery)}명, 재탐지로 bbox 확보 완료 (EXP-027/036/037과 동일 집합 확인됨)")

    full_images: dict[Path, np.ndarray] = {}
    for img_path, _ in refs.values():
        if img_path not in full_images:
            full_images[img_path] = cv2.imread(str(img_path))

    queries = build_queries(gallery, rng_for_queries)
    query_true_ids = {qid: q["true_id"] for qid, q in queries.items()}
    print(f"Query: {len(queries)}개 ({len(gallery)}명 x variant {len(queries) // len(gallery)}종)")

    embedder = YoloBackboneEmbedder(weights_path="yolo11n-cls.pt")

    pipelines = {
        "gamma_background_alt_b": adaptive_gamma_correct_from_background,
        "gamma_background_plus_wb": gamma_background_then_white_balance,
    }

    summary: dict = {"n_gallery": len(gallery), "n_queries": len(queries), "variants": ALL_VARIANTS}

    for label, correction_fn in pipelines.items():
        gallery_corrected = {
            gid: correct_image(img, gid, None, refs, full_images, correction_fn) for gid, img in gallery.items()
        }
        query_images_corrected = {
            qid: correct_image(q["image"], q["true_id"], q["variant"], refs, full_images, correction_fn)
            for qid, q in queries.items()
        }
        queries_corrected = {
            qid: {"image": query_images_corrected[qid], "true_id": q["true_id"], "variant": q["variant"]}
            for qid, q in queries.items()
        }

        baseline_result = evaluate_method(
            f"baseline_color_histogram_{label}",
            color_histogram_embedding,
            gallery_corrected,
            query_images_corrected,
            query_true_ids,
        )
        candidate_result = evaluate_method(
            f"candidate_yolo_backbone_{label}", embedder.embed, gallery_corrected, query_images_corrected, query_true_ids
        )
        fusion_result = evaluate_fusion_images(
            gallery_corrected, queries_corrected, color_histogram_embedding, embedder.embed
        )

        print(
            f"[{label}] Baseline Top-1={baseline_result['overall']['top1']:.4f} "
            f"warm_cast={baseline_result['by_variant']['warm_cast']['top1']:.4f} "
            f"cool_cast={baseline_result['by_variant']['cool_cast']['top1']:.4f}"
        )
        print(
            f"[{label}] Candidate Top-1={candidate_result['overall']['top1']:.4f} "
            f"warm_cast={candidate_result['by_variant']['warm_cast']['top1']:.4f} "
            f"cool_cast={candidate_result['by_variant']['cool_cast']['top1']:.4f} "
            f"low_light={candidate_result['by_variant']['low_light']['top1']:.4f} "
            f"strong_light={candidate_result['by_variant']['strong_light']['top1']:.4f}"
        )

        summary[label] = {
            "baseline_color_histogram": strip_per_query(baseline_result),
            "candidate_yolo11n_cls_backbone": strip_per_query(candidate_result),
            "fusion_0.5_0.5": fusion_result,
        }

        with (RESULTS_DIR / f"per_query_predictions_{label}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "query_id", "true_id", "top1_pred", "rank_of_true", "ap"])
            for result in (baseline_result, candidate_result):
                for row in result["per_query"]:
                    writer.writerow(
                        [
                            result["method"],
                            row["query_id"],
                            row["true_id"],
                            row["top1_pred"],
                            row["rank_of_true"],
                            f"{row['ap']:.4f}",
                        ]
                    )

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"저장: {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
