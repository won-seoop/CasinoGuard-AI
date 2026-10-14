"""Stretch: Person Re-ID - Embedding -> Cosine Similarity -> Similar Person Search (지침 18).

이 모듈은 임베딩을 어떻게 뽑는지(cv2/torch 의존)와 완전히 분리된, 순수
numpy 기반의 랭킹/평가 로직만 담는다. 임베딩 추출 자체는 src/reid/embedding.py가
맡고, 이 모듈은 "임베딩 벡터가 주어졌을 때 가장 비슷한 사람을 어떻게 찾고
그 품질을 어떻게 측정하는가"만 다룬다 - 지침 32(직접 작성한 Business Logic
우선 테스트)에 따라 외부 모델 의존 없이 결정론적으로 단위 테스트할 수 있게
분리했다.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 임베딩 벡터의 Cosine Similarity. 둘 중 하나라도 영벡터면 0.0(정의되지
    않은 방향이므로 "닮지 않음"으로 취급)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_gallery(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_ids: list[str],
) -> list[tuple[str, float]]:
    """Gallery 전체를 query와의 Cosine Similarity 내림차순으로 정렬해 반환한다.

    gallery_embeddings: (N, D) 배열, gallery_ids[i]가 gallery_embeddings[i]의 ID.
    동일 Similarity일 때 순서는 안정적이도록(np.argsort는 stable) gallery_ids
    원래 순서를 그대로 유지한다.
    """
    if len(gallery_ids) != gallery_embeddings.shape[0]:
        raise ValueError("gallery_embeddings와 gallery_ids 길이가 다르다")
    sims = [cosine_similarity(query_embedding, gallery_embeddings[i]) for i in range(len(gallery_ids))]
    order = np.argsort(sims, kind="stable")[::-1]
    return [(gallery_ids[i], sims[i]) for i in order]


def topk_hit(ranked_ids: list[str], true_id: str, k: int) -> bool:
    """랭킹 상위 k개 안에 정답 ID가 있으면 True (Top-1, Top-5 등 지침 18 Metric)."""
    return true_id in ranked_ids[:k]


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """표준 Average Precision. relevant_ids가 비어 있으면 0.0(정의 불가)을 반환한다.

    이 프로젝트의 평가 Protocol(Baseline)은 Query 1개당 정답 Gallery ID가 정확히
    1개뿐이므로(단일 진짜 신원 + 나머지는 전부 다른 사람), 그 경우 AP는 단순히
    1/rank_of_correct_match로 축약되지만, 여러 개의 정답(예: 같은 사람의 Gallery
    등록이 여러 장인 경우)에도 그대로 동작하도록 일반형으로 구현한다.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, gid in enumerate(ranked_ids, start=1):
        if gid in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def rank_gallery_fused(
    query_embeddings: list[np.ndarray],
    gallery_embeddings_list: list[np.ndarray],
    weights: list[float],
    gallery_ids: list[str],
) -> list[tuple[str, float]]:
    """여러 임베딩 공간(예: Baseline Color Histogram + 대안 B Deep 임베딩)의 Cosine
    Similarity를 가중 평균해 하나의 랭킹으로 합친다(Score-level Fusion).

    서로 다른 임베딩은 차원·분포가 다르므로 Vector를 이어붙이는 대신, 공간별로
    따로 계산한 [-1, 1] 범위의 Cosine Similarity를 가중 평균하는 쪽이 한쪽
    임베딩의 Scale이 다른 쪽을 압도하지 않는다. weights는 합이 1이 아니어도
    되며 내부적으로 정규화한다(둘 다 0이면 ValueError).
    """
    if not (len(query_embeddings) == len(gallery_embeddings_list) == len(weights)):
        raise ValueError("query_embeddings/gallery_embeddings_list/weights 길이가 다르다")
    total_weight = sum(weights)
    if abs(total_weight) < 1e-12:
        raise ValueError("weights 합이 0이면 융합할 수 없다")
    normalized_weights = [w / total_weight for w in weights]

    n = len(gallery_ids)
    fused = np.zeros(n, dtype=np.float64)
    for q_emb, g_embs, w in zip(query_embeddings, gallery_embeddings_list, normalized_weights):
        if g_embs.shape[0] != n:
            raise ValueError("gallery_embeddings_list의 각 배열은 gallery_ids와 길이가 같아야 한다")
        sims = np.array([cosine_similarity(q_emb, g_embs[i]) for i in range(n)])
        fused += w * sims

    order = np.argsort(fused, kind="stable")[::-1]
    return [(gallery_ids[i], float(fused[i])) for i in order]


def evaluate_identification(
    query_embeddings: dict[str, np.ndarray],
    query_true_ids: dict[str, str],
    gallery_embeddings: np.ndarray,
    gallery_ids: list[str],
    ks: tuple[int, ...] = (1, 5),
) -> dict:
    """Query 집합 전체에 대해 Top-K Accuracy와 mAP를 계산한다.

    query_embeddings: {query_id: embedding}. query_true_ids: {query_id: 정답
    gallery_id}. 각 Query마다 rank_gallery로 전체 Gallery를 정렬하고, topk_hit과
    average_precision을 누적해 평균을 낸다.
    """
    if not query_embeddings:
        return {"n_queries": 0, **{f"top{k}": 0.0 for k in ks}, "mAP": 0.0}
    topk_counts = dict.fromkeys(ks, 0)
    ap_sum = 0.0
    per_query: list[dict] = []
    for qid, qemb in query_embeddings.items():
        true_id = query_true_ids[qid]
        ranked = rank_gallery(qemb, gallery_embeddings, gallery_ids)
        ranked_ids = [gid for gid, _ in ranked]
        rank_of_true = ranked_ids.index(true_id) + 1 if true_id in ranked_ids else None
        ap = average_precision(ranked_ids, {true_id})
        ap_sum += ap
        for k in ks:
            if topk_hit(ranked_ids, true_id, k):
                topk_counts[k] += 1
        per_query.append(
            {
                "query_id": qid,
                "true_id": true_id,
                "top1_pred": ranked_ids[0] if ranked_ids else None,
                "rank_of_true": rank_of_true,
                "ap": ap,
            }
        )
    n = len(query_embeddings)
    result = {"n_queries": n, "mAP": ap_sum / n, "per_query": per_query}
    for k in ks:
        result[f"top{k}"] = topk_counts[k] / n
    return result
