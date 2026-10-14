"""EXP-036 Person Re-ID Baseline (지침 18).

Goal: coco128 실제 person crop(EXP-027이 이미 저장한 results/EXP-027/crops/*.jpg
45장, 서로 다른 실제 인물 45명)을 Gallery로 쓰고, 각 인물마다 합성 Query
변형(조명 변화/Occlusion/수평 Flip/Bbox Jitter)을 만들어 "Query가 Gallery
45명 중 진짜 자신을 찾아내는가"를 Baseline(Color Histogram)과 대안 B
(YOLO11n-cls Backbone 임베딩)로 비교한다.

Dataset 한계(정직하게 명시): Market-1501/MSMT17 같은 진짜 Multi-Camera
Re-ID Dataset은 이 세션의 egress 정책상 접근 불가(대부분 Google Drive/전용
서버 호스팅, drive.google.com 403 실측 확인 - 아래 Decision 참고). coco128은
인물별로 사진이 1장뿐이라 "같은 사람 다른 조명/Occlusion"을 합성으로
만들어야 한다 - 이는 진짜 다른 각도/카메라 변화를 대체하지 못하는 근사치임을
결과 해석 시 반드시 함께 읽어야 한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reid.embedding import YoloBackboneEmbedder, color_histogram_embedding  # noqa: E402
from reid.matching import average_precision, evaluate_identification, rank_gallery_fused, topk_hit  # noqa: E402

CROPS_DIR = ROOT / "results" / "EXP-027" / "crops"
RESULTS_DIR = ROOT / "results" / "EXP-036"
RNG_SEED = 42

LIGHTING_CONDITIONS = ["low_light", "strong_light", "warm_cast", "cool_cast"]


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027과 동일한 합성 조명 변형(재현성을 위해 동일 Config 유지)."""
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
    """실제 CCTV에서 흔한 부분 Occlusion(다른 사람/기둥에 가려짐)을 모사한다.
    bbox 폭 40~60%, 높이 30~50%의 사각형을 임의 위치에 중립 회색으로 덮는다."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    ow = int(w * rng.uniform(0.4, 0.6))
    oh = int(h * rng.uniform(0.3, 0.5))
    x0 = int(rng.uniform(0, max(1, w - ow)))
    y0 = int(rng.uniform(0, max(1, h - oh)))
    out[y0 : y0 + oh, x0 : x0 + ow] = (120, 120, 120)
    return out


def apply_bbox_jitter(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Detector bbox가 Frame마다 조금씩 흔들리는 상황(동일 Track 내 반복 검출)을
    모사한다 - 가장자리를 5~15% 범위에서 비대칭으로 잘라낸다."""
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
    """다른 각도(카메라 반대쪽에서 바라본 모습)의 매우 거친 근사치로 수평 Flip을
    쓴다 - 실제 시점 변화(앞/뒤/옆모습)를 대체하지 못하는 한계가 있음을
    experiment.md Decision에 명시한다."""
    return cv2.flip(img_bgr, 1)


def load_gallery(crops_dir: Path) -> dict[str, np.ndarray]:
    gallery = {}
    for path in sorted(crops_dir.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            continue
        gallery[path.stem] = img
    return gallery


def build_queries(gallery: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, dict]:
    """각 Gallery 인물마다 variant별 Query 이미지를 만든다.

    반환: {query_id: {"image": np.ndarray, "true_id": gallery_id, "variant": str}}
    """
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


def evaluate_method(
    method_name: str,
    embed_fn,
    gallery: dict[str, np.ndarray],
    queries: dict[str, dict],
) -> dict:
    gallery_ids = list(gallery.keys())
    gallery_embeddings = np.stack([embed_fn(gallery[gid]) for gid in gallery_ids])
    query_embeddings = {qid: embed_fn(q["image"]) for qid, q in queries.items()}
    query_true_ids = {qid: q["true_id"] for qid, q in queries.items()}

    overall = evaluate_identification(query_embeddings, query_true_ids, gallery_embeddings, gallery_ids)

    by_variant: dict[str, dict] = {}
    variants = sorted({q["variant"] for q in queries.values()})
    for variant in variants:
        subset_ids = {qid: emb for qid, emb in query_embeddings.items() if queries[qid]["variant"] == variant}
        subset_true = {qid: query_true_ids[qid] for qid in subset_ids}
        by_variant[variant] = evaluate_identification(subset_ids, subset_true, gallery_embeddings, gallery_ids)

    return {
        "method": method_name,
        "overall": {k: v for k, v in overall.items() if k != "per_query"},
        "by_variant": {v: {k: val for k, val in r.items() if k != "per_query"} for v, r in by_variant.items()},
        "per_query": overall["per_query"],
    }


def evaluate_fusion(
    gallery: dict[str, np.ndarray],
    queries: dict[str, dict],
    baseline_embed_fn,
    candidate_embed_fn,
    weight_baseline: float = 0.5,
    ks: tuple[int, ...] = (1, 5),
) -> dict:
    """Baseline(Color Histogram)과 대안 B(YOLO Backbone) Cosine Similarity를
    Score-level로 가중 평균해(기본 50:50) 두 방법의 상호 보완적 실패 조건을
    하나의 랭킹으로 완화할 수 있는지 측정한다."""
    gallery_ids = list(gallery.keys())
    baseline_gallery = np.stack([baseline_embed_fn(gallery[gid]) for gid in gallery_ids])
    candidate_gallery = np.stack([candidate_embed_fn(gallery[gid]) for gid in gallery_ids])

    topk_counts = dict.fromkeys(ks, 0)
    ap_sum = 0.0
    by_variant_counts: dict[str, dict] = {}
    for qid, q in queries.items():
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
            hit = topk_hit(ranked_ids, true_id, k)
            if hit:
                topk_counts[k] += 1
                bucket["topk"][k] += 1

    n = len(queries)
    overall = {"n_queries": n, "mAP": ap_sum / n, **{f"top{k}": topk_counts[k] / n for k in ks}}
    by_variant = {
        v: {"n_queries": b["n"], "mAP": b["ap_sum"] / b["n"], **{f"top{k}": b["topk"][k] / b["n"] for k in ks}}
        for v, b in by_variant_counts.items()
    }
    return {"method": "fusion_0.5_0.5", "overall": overall, "by_variant": by_variant}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    gallery = load_gallery(CROPS_DIR)
    print(f"Gallery: {len(gallery)}명 (coco128 실제 crop, results/EXP-027/crops)")
    queries = build_queries(gallery, rng)
    print(f"Query: {len(queries)}개 ({len(gallery)}명 x variant {len(queries) // len(gallery)}종)")

    baseline_result = evaluate_method("baseline_color_histogram", color_histogram_embedding, gallery, queries)
    print(
        f"[Baseline] Top-1={baseline_result['overall']['top1']:.4f} "
        f"Top-5={baseline_result['overall']['top5']:.4f} mAP={baseline_result['overall']['mAP']:.4f}"
    )

    embedder = YoloBackboneEmbedder(weights_path="yolo11n-cls.pt")
    candidate_result = evaluate_method("candidate_yolo11n_cls_backbone", embedder.embed, gallery, queries)
    print(
        f"[Candidate] Top-1={candidate_result['overall']['top1']:.4f} "
        f"Top-5={candidate_result['overall']['top5']:.4f} mAP={candidate_result['overall']['mAP']:.4f}"
    )

    fusion_result = evaluate_fusion(gallery, queries, color_histogram_embedding, embedder.embed)
    print(
        f"[Fusion 0.5/0.5] Top-1={fusion_result['overall']['top1']:.4f} "
        f"Top-5={fusion_result['overall']['top5']:.4f} mAP={fusion_result['overall']['mAP']:.4f}"
    )

    summary = {
        "n_gallery": len(gallery),
        "n_queries": len(queries),
        "variants": sorted({q["variant"] for q in queries.values()}),
        "baseline_color_histogram": baseline_result["overall"],
        "baseline_by_variant": baseline_result["by_variant"],
        "candidate_yolo11n_cls_backbone": candidate_result["overall"],
        "candidate_by_variant": candidate_result["by_variant"],
        "fusion_0.5_0.5": fusion_result["overall"],
        "fusion_by_variant": fusion_result["by_variant"],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    import csv

    with (RESULTS_DIR / "per_query_predictions.csv").open("w", newline="") as f:
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

    print(f"저장: {RESULTS_DIR / 'summary.json'}, {RESULTS_DIR / 'per_query_predictions.csv'}")


if __name__ == "__main__":
    main()
