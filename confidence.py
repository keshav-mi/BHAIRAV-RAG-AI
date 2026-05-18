# FAISS inner-product confidence bands for adaptive reranking.

from __future__ import annotations

from typing import List, Sequence, Tuple

from config import (
    NEIGHBOR_MIN_CONFIDENCE,
    RERANK_GATE_ENABLED,
    RERANK_GATE_MARGIN_THR,
    RERANK_GATE_SCORE_HIGH,
    RERANK_GATE_SCORE_MED,
    RERANK_TOPN_HIGH,
    RERANK_TOPN_LOW,
    RERANK_TOPN_MEDIUM,
)

_BAND_ORDER = {"high": 0, "medium": 1, "low": 2}


def neighbors_allowed(confidence_band: str) -> bool:
    need = _BAND_ORDER.get(NEIGHBOR_MIN_CONFIDENCE, 1)
    have = _BAND_ORDER.get(confidence_band, 2)
    return have >= need

FaissHit = Tuple[str, float]


def best_faiss_hits(all_hits: Sequence[FaissHit]) -> List[FaissHit]:
    """Merge multi-variant FAISS lists; keep max score per chunk id."""
    best: dict[str, float] = {}
    for cid, score in all_hits:
        best[cid] = max(best.get(cid, -1.0), score)
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def confidence_from_faiss(
    all_faiss_hits: Sequence[FaissHit],
) -> dict:
    """
    Returns band, scores, and rerank policy from FAISS IP (L2-normalized = cosine).
    """
    ranked = best_faiss_hits(all_faiss_hits)
    if not ranked:
        return {
            "confidence_band": "low",
            "faiss_top1_score": 0.0,
            "faiss_top2_score": 0.0,
            "faiss_margin": 0.0,
            "rerank_policy": "top15",
            "rerank_top_n": RERANK_TOPN_LOW,
            "skip_rerank": False,
        }

    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1 - top2

    if not RERANK_GATE_ENABLED:
        band = "medium"
        policy = "top15"
        top_n = RERANK_TOPN_LOW
        skip = False
    elif top1 >= RERANK_GATE_SCORE_HIGH and margin >= RERANK_GATE_MARGIN_THR:
        band = "high"
        policy = "skip"
        top_n = RERANK_TOPN_HIGH
        skip = True
    elif top1 >= RERANK_GATE_SCORE_MED:
        band = "medium"
        policy = "top8"
        top_n = RERANK_TOPN_MEDIUM
        skip = False
    else:
        band = "low"
        policy = "top15"
        top_n = RERANK_TOPN_LOW
        skip = False

    return {
        "confidence_band": band,
        "faiss_top1_score": round(top1, 4),
        "faiss_top2_score": round(top2, 4),
        "faiss_margin": round(margin, 4),
        "rerank_policy": policy,
        "rerank_top_n": top_n,
        "skip_rerank": skip,
    }
