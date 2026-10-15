"""EXP-037: FC-013(Person Re-ID 저노출/과다노출 붕괴) - 배경 기반 Gamma 보정 검증 (지침 18).

Roadmap(01. Architecture & Roadmap) EXP-036 Next Action 7번: FC-013은 Color Histogram과
YOLO11n-cls Backbone 임베딩 둘 다 low_light/strong_light 합성 변형에서 Top-1이 거의
무작위 수준으로 붕괴하는 문제였다(baseline low_light 33.3%/strong_light 2.2%, 후보 B
low_light 2.2%/strong_light 0.0%). 이 실험은 FC-012(Attribute 색상 분류)에서 이미
부분 효과가 검증된 배경 기반 Gamma 보정(EXP-030 대안 B, strong_light GT=black->blue
오분류율 18.2%->4.6%로 개선)을 Re-ID 임베딩 추출 "직전"에 적용하면 같은 효과가 재현되는지
측정한다.

Hypothesis: low_light(`img*0.35`)와 strong_light(`img*1.9+25`)는 전역 노출 변화이므로,
같은 Frame의 배경(person bbox 밖)에서 추정한 Gamma를 Query crop에 역으로 적용해 밝기를
정상 범위로 되돌리면, 두 임베딩 방식(Color Histogram의 HSV Hue/Saturation 양자화 오차,
YOLO Backbone의 ImageNet 분포 이탈) 모두 완화되어 low_light/strong_light Top-1이
개선될 것이다. 조명 Cast(warm_cast/cool_cast, 채널 간 상대적 편향)와 기하학적 변형
(occlusion/bbox_jitter/flip)에는 Gamma 보정이 겨냥하는 원인(전역 노출)이 없으므로 거의
불변이거나 소폭의 부작용만 있을 것으로 예상한다 - 검증 없이 "항상 적용"으로 가정하지
않고 전체 Variant에 대해 측정한다(EXP-030/031이 반복한 "가설이 반증될 수 있다"는 교훈).

중요(EXP-030/031의 교훈 반복 방지): "평균 개선"만 보고 끝내지 않는다 - FC-013이 겨냥한
low_light/strong_light Variant를 직접 슬라이싱해 확인하고, 다른 Variant의 회귀 여부도
반드시 함께 측정한다.

Gallery: EXP-027/036과 동일한 coco128 실제 person crop 45명(`results/EXP-027/crops`).
배경 추출을 위해 원본 이미지+bbox가 필요해 EXP-030과 동일한 방식(yolo11n.pt, conf=0.4,
MIN_BOX_AREA 동일)으로 coco128을 재탐지한다 - 아래에서 재탐지 결과가 EXP-027/036
Gallery 45명과 정확히 일치함을 실행 시 검증한다(불일치 시 RuntimeError).

실행: python scripts/run_exp037_reid_gamma_correction.py
출력: results/EXP-037/{summary.json, per_query_predictions.csv}
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

from attributes.color import adaptive_gamma_correct, adaptive_gamma_correct_from_background  # noqa: E402
from reid.embedding import YoloBackboneEmbedder, color_histogram_embedding  # noqa: E402
from reid.matching import average_precision, evaluate_identification, rank_gallery_fused, topk_hit  # noqa: E402

COCO128_IMAGES = Path("/tmp/coco128_extract/coco128/images/train2017")
CROPS_DIR = ROOT / "results" / "EXP-027" / "crops"
RESULTS_DIR = ROOT / "results" / "EXP-037"
RNG_SEED = 42
MIN_BOX_AREA = 120 * 250
BACKGROUND_MIN_PIXELS = 5000

LIGHTING_CONDITIONS = ["low_light", "strong_light", "warm_cast", "cool_cast"]
ALL_VARIANTS = [*LIGHTING_CONDITIONS, "occlusion", "bbox_jitter", "flip"]


def apply_lighting(img_bgr: np.ndarray, condition: str) -> np.ndarray:
    """EXP-025/027/030/036과 완전히 동일한 합성 조명 변형(재현성을 위해 동일 Config 유지)."""
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
    """EXP-036과 동일 - bbox 폭 40~60%, 높이 30~50%의 사각형을 중립 회색으로 덮는다."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    ow = int(w * rng.uniform(0.4, 0.6))
    oh = int(h * rng.uniform(0.3, 0.5))
    x0 = int(rng.uniform(0, max(1, w - ow)))
    y0 = int(rng.uniform(0, max(1, h - oh)))
    out[y0 : y0 + oh, x0 : x0 + ow] = (120, 120, 120)
    return out


def apply_bbox_jitter(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """EXP-036과 동일 - 가장자리를 5~15% 범위에서 비대칭으로 잘라낸다."""
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
    """EXP-036과 동일 - 수평 Flip."""
    return cv2.flip(img_bgr, 1)


def background_sample(full_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """EXP-030과 동일 - person bbox를 제외한 Frame 나머지 영역 픽셀을 (N,1,3)으로 반환."""
    h, w = full_img.shape[:2]
    x1, y1, x2, y2 = bbox
    mask = np.ones((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = False
    pixels = full_img[mask]
    return pixels.reshape(-1, 1, 3)


def detect_person_crops_and_bboxes() -> tuple[dict[str, np.ndarray], dict[str, tuple[Path, tuple[int, int, int, int]]]]:
    """EXP-030과 동일한 탐지 설정(yolo11n.pt, conf=0.4, MIN_BOX_AREA)으로 coco128을
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
    """EXP-036과 동일한 Query 구성(같은 RNG 시드 42로 동일 호출 순서 유지 -> 동일 난수
    소비, Baseline/Gamma 비교가 같은 합성 왜곡에서 이뤄지도록 보장)."""
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


def gamma_correct_image(
    img_bgr: np.ndarray,
    gallery_id: str,
    variant: str | None,
    refs: dict[str, tuple[Path, tuple[int, int, int, int]]],
    full_images: dict[Path, np.ndarray],
) -> np.ndarray:
    """img_bgr(Gallery 원본 또는 Query 변형 이미지)에 배경 기반 Gamma 보정을 적용한다.

    variant가 조명 조건(low_light/strong_light/warm_cast/cool_cast)이면 배경도 같은
    조명 변형을 거친 뒤 샘플링한다(EXP-030과 동일 - "같은 Frame이 함께 조명 변화를
    겪었다"는 가정). variant가 None(Gallery)이거나 기하 변형(occlusion/bbox_jitter/flip)
    이면 배경은 원본 그대로(조명 변화 없음)를 사용한다 - "답을 알고 치팅"하지 않고
    정상 조명 상황에서도 동일 파이프라인을 무조건 통과시켜 부작용을 함께 측정한다.
    """
    img_path, bbox = refs[gallery_id]
    full_img = full_images[img_path]
    if variant in LIGHTING_CONDITIONS:
        lit_full = apply_lighting(full_img, variant)
    else:
        lit_full = full_img
    bg_pixels = background_sample(lit_full, bbox)
    return adaptive_gamma_correct_from_background(img_bgr, bg_pixels)


def gamma_correct_image_self(img_bgr: np.ndarray) -> np.ndarray:
    """대안 A(검토, 비교용) - crop 자신의 평균 밝기로 감마를 추정해 자신에게 적용한다
    (EXP-030의 adaptive_gamma_correct(), Attribute 분류에서는 confound로 기각됨).
    Re-ID 임베딩은 "옷이 검정인가"가 아니라 crop 전체의 외형 패턴을 구분하는 것이
    목적이라 같은 confound가 얼마나 다르게(또는 비슷하게) 작용하는지 대안 B(배경 기반)와
    직접 비교한다."""
    return adaptive_gamma_correct(img_bgr)


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
        raise RuntimeError(f"재탐지 결과가 EXP-027/036 Gallery와 불일치(배경 bbox 없음): {missing}")
    print(f"Gallery: {len(gallery)}명, 재탐지로 bbox 확보 완료 (EXP-027/036과 동일 집합 확인됨)")

    full_images: dict[Path, np.ndarray] = {}
    for img_path, _ in refs.values():
        if img_path not in full_images:
            full_images[img_path] = cv2.imread(str(img_path))

    queries = build_queries(gallery, rng_for_queries)
    query_true_ids = {qid: q["true_id"] for qid, q in queries.items()}
    print(f"Query: {len(queries)}개 ({len(gallery)}명 x variant {len(queries) // len(gallery)}종)")

    # --- No-Gamma (EXP-036 재현, 동일 코드/시드) ---
    query_images_plain = {qid: q["image"] for qid, q in queries.items()}

    embedder = YoloBackboneEmbedder(weights_path="yolo11n-cls.pt")

    baseline_plain = evaluate_method(
        "baseline_color_histogram_no_gamma", color_histogram_embedding, gallery, query_images_plain, query_true_ids
    )
    candidate_plain = evaluate_method(
        "candidate_yolo_backbone_no_gamma", embedder.embed, gallery, query_images_plain, query_true_ids
    )
    fusion_plain = evaluate_fusion_images(gallery, queries, color_histogram_embedding, embedder.embed)

    print(
        f"[No-Gamma Baseline] Top-1={baseline_plain['overall']['top1']:.4f} "
        f"low_light={baseline_plain['by_variant']['low_light']['top1']:.4f} "
        f"strong_light={baseline_plain['by_variant']['strong_light']['top1']:.4f}"
    )
    print(
        f"[No-Gamma Candidate] Top-1={candidate_plain['overall']['top1']:.4f} "
        f"low_light={candidate_plain['by_variant']['low_light']['top1']:.4f} "
        f"strong_light={candidate_plain['by_variant']['strong_light']['top1']:.4f}"
    )

    # --- Gamma-Corrected (배경 기반 Gamma를 Gallery+Query 양쪽에 동일 파이프라인으로 적용) ---
    gallery_gamma = {
        gid: gamma_correct_image(img, gid, None, refs, full_images) for gid, img in gallery.items()
    }
    query_images_gamma = {
        qid: gamma_correct_image(q["image"], q["true_id"], q["variant"], refs, full_images)
        for qid, q in queries.items()
    }
    queries_gamma = {
        qid: {"image": query_images_gamma[qid], "true_id": q["true_id"], "variant": q["variant"]}
        for qid, q in queries.items()
    }

    baseline_gamma = evaluate_method(
        "baseline_color_histogram_gamma", color_histogram_embedding, gallery_gamma, query_images_gamma, query_true_ids
    )
    candidate_gamma = evaluate_method(
        "candidate_yolo_backbone_gamma", embedder.embed, gallery_gamma, query_images_gamma, query_true_ids
    )
    fusion_gamma = evaluate_fusion_images(gallery_gamma, queries_gamma, color_histogram_embedding, embedder.embed)

    print(
        f"[Gamma Baseline] Top-1={baseline_gamma['overall']['top1']:.4f} "
        f"low_light={baseline_gamma['by_variant']['low_light']['top1']:.4f} "
        f"strong_light={baseline_gamma['by_variant']['strong_light']['top1']:.4f}"
    )
    print(
        f"[Gamma Candidate] Top-1={candidate_gamma['overall']['top1']:.4f} "
        f"low_light={candidate_gamma['by_variant']['low_light']['top1']:.4f} "
        f"strong_light={candidate_gamma['by_variant']['strong_light']['top1']:.4f}"
    )

    # --- 대안 A: Gamma-Self (crop 자신의 밝기로 추정, 배경 불필요) ---
    gallery_gamma_self = {gid: gamma_correct_image_self(img) for gid, img in gallery.items()}
    query_images_gamma_self = {qid: gamma_correct_image_self(q["image"]) for qid, q in queries.items()}
    queries_gamma_self = {
        qid: {"image": query_images_gamma_self[qid], "true_id": q["true_id"], "variant": q["variant"]}
        for qid, q in queries.items()
    }

    baseline_gamma_self = evaluate_method(
        "baseline_color_histogram_gamma_self",
        color_histogram_embedding,
        gallery_gamma_self,
        query_images_gamma_self,
        query_true_ids,
    )
    candidate_gamma_self = evaluate_method(
        "candidate_yolo_backbone_gamma_self", embedder.embed, gallery_gamma_self, query_images_gamma_self, query_true_ids
    )
    fusion_gamma_self = evaluate_fusion_images(
        gallery_gamma_self, queries_gamma_self, color_histogram_embedding, embedder.embed
    )

    print(
        f"[Gamma-Self(A) Baseline] Top-1={baseline_gamma_self['overall']['top1']:.4f} "
        f"low_light={baseline_gamma_self['by_variant']['low_light']['top1']:.4f} "
        f"strong_light={baseline_gamma_self['by_variant']['strong_light']['top1']:.4f}"
    )
    print(
        f"[Gamma-Self(A) Candidate] Top-1={candidate_gamma_self['overall']['top1']:.4f} "
        f"low_light={candidate_gamma_self['by_variant']['low_light']['top1']:.4f} "
        f"strong_light={candidate_gamma_self['by_variant']['strong_light']['top1']:.4f}"
    )

    def strip_per_query(result: dict) -> dict:
        return {"overall": result["overall"], "by_variant": result["by_variant"]}

    summary = {
        "n_gallery": len(gallery),
        "n_queries": len(queries),
        "variants": ALL_VARIANTS,
        "no_gamma": {
            "baseline_color_histogram": strip_per_query(baseline_plain),
            "candidate_yolo11n_cls_backbone": strip_per_query(candidate_plain),
            "fusion_0.5_0.5": fusion_plain,
        },
        "gamma_self_alt_a": {
            "baseline_color_histogram": strip_per_query(baseline_gamma_self),
            "candidate_yolo11n_cls_backbone": strip_per_query(candidate_gamma_self),
            "fusion_0.5_0.5": fusion_gamma_self,
        },
        "gamma_background_alt_b": {
            "baseline_color_histogram": strip_per_query(baseline_gamma),
            "candidate_yolo11n_cls_backbone": strip_per_query(candidate_gamma),
            "fusion_0.5_0.5": fusion_gamma,
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with (RESULTS_DIR / "per_query_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "method", "query_id", "true_id", "top1_pred", "rank_of_true", "ap"])
        for gamma_label, result in (
            ("no_gamma", baseline_plain),
            ("no_gamma", candidate_plain),
            ("gamma_self_alt_a", baseline_gamma_self),
            ("gamma_self_alt_a", candidate_gamma_self),
            ("gamma_background_alt_b", baseline_gamma),
            ("gamma_background_alt_b", candidate_gamma),
        ):
            for row in result["per_query"]:
                writer.writerow(
                    [
                        gamma_label,
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
