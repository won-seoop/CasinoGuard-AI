import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reid.matching import (  # noqa: E402
    average_precision,
    cosine_similarity,
    evaluate_identification,
    rank_gallery,
    rank_gallery_fused,
    topk_hit,
)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_rank_gallery_orders_by_descending_similarity():
    query = np.array([1.0, 0.0])
    gallery_embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    gallery_ids = ["far", "exact", "mid"]
    ranked = rank_gallery(query, gallery_embeddings, gallery_ids)
    assert [gid for gid, _ in ranked] == ["exact", "mid", "far"]


def test_rank_gallery_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rank_gallery(np.array([1.0, 0.0]), np.zeros((2, 2)), ["a", "b", "c"])


def test_topk_hit_true_within_k():
    assert topk_hit(["b", "a", "c"], "a", k=2) is True


def test_topk_hit_false_outside_k():
    assert topk_hit(["b", "c", "a"], "a", k=2) is False


def test_topk_hit_k_zero_never_hits():
    assert topk_hit(["a", "b"], "a", k=0) is False


def test_average_precision_correct_match_at_rank_one_is_one():
    assert average_precision(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_average_precision_correct_match_at_rank_three_is_one_third():
    assert average_precision(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)


def test_average_precision_no_relevant_in_ranking_is_zero():
    assert average_precision(["x", "y", "z"], {"a"}) == 0.0


def test_average_precision_empty_relevant_set_is_zero():
    assert average_precision(["x", "y"], set()) == 0.0


def test_average_precision_multiple_relevant_standard_formula():
    # relevant = {a, c}; ranked = [a, b, c] -> precision@1=1/1, precision@3=2/3
    # AP = (1 + 2/3) / 2
    ap = average_precision(["a", "b", "c"], {"a", "c"})
    assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_evaluate_identification_perfect_matches():
    gallery_ids = ["p1", "p2", "p3"]
    gallery_embeddings = np.eye(3)
    query_embeddings = {"q1": np.array([1.0, 0.0, 0.0]), "q2": np.array([0.0, 1.0, 0.0])}
    query_true_ids = {"q1": "p1", "q2": "p2"}
    result = evaluate_identification(query_embeddings, query_true_ids, gallery_embeddings, gallery_ids, ks=(1,))
    assert result["top1"] == pytest.approx(1.0)
    assert result["mAP"] == pytest.approx(1.0)
    assert result["n_queries"] == 2


def test_evaluate_identification_empty_queries():
    result = evaluate_identification({}, {}, np.eye(2), ["p1", "p2"])
    assert result["n_queries"] == 0
    assert result["mAP"] == 0.0


def test_evaluate_identification_tracks_rank_of_true_when_wrong():
    gallery_ids = ["p1", "p2", "p3"]
    gallery_embeddings = np.eye(3)
    # query is closest to p2, but true identity is p3 (rank 2: p1 is orthogonal, scores lowest)
    query_embeddings = {"q1": np.array([0.0, 1.0, 0.01])}
    query_true_ids = {"q1": "p3"}
    result = evaluate_identification(query_embeddings, query_true_ids, gallery_embeddings, gallery_ids, ks=(1, 2))
    row = result["per_query"][0]
    assert row["top1_pred"] == "p2"
    assert row["rank_of_true"] == 2
    assert result["top1"] == pytest.approx(0.0)
    assert result["top2"] == pytest.approx(1.0)


def test_rank_gallery_fused_weighted_average_matches_manual_computation():
    gallery_ids = ["a", "b"]
    q1, q2 = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    g1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # method 1: "a" is a perfect match
    g2 = np.array([[0.0, 1.0], [1.0, 0.0]])  # method 2: "b" is a perfect match
    # equal weights -> both candidates tie at 0.5 fused score each
    ranked = rank_gallery_fused([q1, q2], [g1, g2], [1.0, 1.0], gallery_ids)
    scores = dict(ranked)
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)


def test_rank_gallery_fused_weight_zero_ignores_that_method():
    gallery_ids = ["a", "b"]
    q1, q2 = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    g1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    g2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    ranked = rank_gallery_fused([q1, q2], [g1, g2], [1.0, 0.0], gallery_ids)
    assert ranked[0][0] == "a"
    assert ranked[0][1] == pytest.approx(1.0)


def test_rank_gallery_fused_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rank_gallery_fused([np.zeros(2)], [np.zeros((2, 2)), np.zeros((2, 2))], [1.0, 1.0], ["a", "b"])


def test_rank_gallery_fused_rejects_zero_total_weight():
    gallery_ids = ["a", "b"]
    g = np.eye(2)
    with pytest.raises(ValueError):
        rank_gallery_fused([np.zeros(2), np.zeros(2)], [g, g], [1.0, -1.0], gallery_ids)
