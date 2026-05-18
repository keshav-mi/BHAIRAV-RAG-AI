"""
Retrieval metrics for labeled (query, gold_chunk_id) evaluation.
Supports single or multiple gold chunk IDs per query.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set, Union

GoldIds = Union[str, Sequence[str]]


def _normalize_gold(gold: GoldIds) -> Set[str]:
    if isinstance(gold, str):
        return {gold}
    return {g for g in gold if g}


def reciprocal_rank(retrieved_ids: Sequence[str], gold: GoldIds) -> float:
    """MRR contribution for one query: 1/rank of first hit, else 0."""
    gold_set = _normalize_gold(gold)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved_ids: Sequence[str], gold: GoldIds, k: int) -> float:
    """Fraction of gold IDs found in top-k (set-based recall)."""
    gold_set = _normalize_gold(gold)
    if not gold_set:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(gold_set & top) / len(gold_set)


def precision_at_k(retrieved_ids: Sequence[str], gold: GoldIds, k: int) -> float:
    """Relevant items in top-k / k."""
    gold_set = _normalize_gold(gold)
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in gold_set)
    return hits / k


def ndcg_at_k(retrieved_ids: Sequence[str], gold: GoldIds, k: int) -> float:
    """Binary nDCG@k for one or more relevant documents."""
    gold_set = _normalize_gold(gold)
    if not gold_set or k <= 0:
        return 0.0

    def dcg(ids: Sequence[str]) -> float:
        s = 0.0
        for i, cid in enumerate(ids[:k], start=1):
            rel = 1.0 if cid in gold_set else 0.0
            if rel:
                s += rel / math.log2(i + 1)
        return s

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg(retrieved_ids) / idcg


def aggregate_metrics(
    per_query: List[dict],
    k_values: Iterable[int] = (1, 5, 10),
) -> dict:
    """Summarize list of per-query metric dicts."""
    if not per_query:
        return {}

    out: dict = {
        "num_queries": len(per_query),
        "MRR": sum(q["mrr"] for q in per_query) / len(per_query),
    }
    for k in k_values:
        out[f"Recall@{k}"] = sum(q[f"recall@{k}"] for q in per_query) / len(per_query)
        out[f"Precision@{k}"] = sum(q[f"precision@{k}"] for q in per_query) / len(per_query)
        out[f"nDCG@{k}"] = sum(q[f"ndcg@{k}"] for q in per_query) / len(per_query)
    return out


def score_one_query(
    retrieved_ids: Sequence[str],
    gold: GoldIds,
    k_values: Iterable[int] = (1, 5, 10),
) -> dict:
    row = {"mrr": reciprocal_rank(retrieved_ids, gold)}
    for k in k_values:
        row[f"recall@{k}"] = recall_at_k(retrieved_ids, gold, k)
        row[f"precision@{k}"] = precision_at_k(retrieved_ids, gold, k)
        row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, gold, k)
    return row
