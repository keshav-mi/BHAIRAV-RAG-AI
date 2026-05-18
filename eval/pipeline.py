"""
Run Bhairav retrieval at different pipeline stages (for metric comparison).
Primary KPI stage: after_rerank. after_neighbors is diagnostic only.
"""

from __future__ import annotations

import time
from typing import List, Literal, Tuple

from config import NEIGHBOR_WINDOW, RERANK_TOP_N
from confidence import neighbors_allowed

Stage = Literal["after_retrieve", "after_rerank", "after_neighbors"]

_retriever = None
_reranker = None


def _script(query: str) -> str:
    return (
        "devanagari"
        if any("\u0900" <= c <= "\u097f" for c in query)
        else "english"
    )


def load_components():
    global _retriever, _reranker
    if _retriever is None:
        from retriever import Retriever
        from reranker import Reranker

        print("Loading Retriever + Reranker (one-time)...")
        _retriever = Retriever()
        _reranker = Reranker()
    return _retriever, _reranker


def retrieve_ids(
    query: str,
    stage: Stage = "after_retrieve",
    top_k: int = 40,
    rerank_top_n: int = 15,
) -> Tuple[List[str], dict]:
    """
    Returns ordered chunk IDs at the chosen pipeline stage + instrumentation meta.
    """
    retriever, reranker = load_components()

    t0 = time.time()
    candidates, faiss_q, bm25_q, domains, retrieval_meta = retriever.retrieve(
        query, top_k=top_k
    )
    latency_retrieve_ms = int((time.time() - t0) * 1000)

    meta = {
        "faiss_query": faiss_q,
        "bm25_query": bm25_q,
        "domains": domains,
        "latency_retrieve_ms": latency_retrieve_ms,
        "confidence_band": retrieval_meta.get("confidence_band"),
        "faiss_top1_score": retrieval_meta.get("faiss_top1_score"),
        "faiss_margin": retrieval_meta.get("faiss_margin"),
        "rerank_policy": retrieval_meta.get("rerank_policy"),
        "intent": retrieval_meta.get("intent"),
        "status_flags": retrieval_meta.get("status_flags", {}),
        "rerank_used": False,
        "neighbors_used": False,
        "latency_rerank_ms": 0,
    }

    if stage == "after_retrieve":
        return [c["id"] for c in candidates], meta

    script = _script(query)
    t0 = time.time()
    top_chunks = reranker.rerank(
        query,
        candidates,
        top_n=rerank_top_n,
        script=script,
        retrieval_meta=retrieval_meta,
    )
    meta["latency_rerank_ms"] = int((time.time() - t0) * 1000)
    meta["rerank_used"] = not retrieval_meta.get("skip_rerank", False)

    if stage == "after_rerank":
        return [c["id"] for c in top_chunks], meta

    # after_neighbors — diagnostic only; append neighbors after reranked hits
    if neighbors_allowed(retrieval_meta.get("confidence_band", "low")):
        expanded = retriever.append_neighbor_chunks(top_chunks, window=NEIGHBOR_WINDOW)
        meta["neighbors_used"] = True
    else:
        expanded = top_chunks

    ids = []
    seen = set()
    for c in expanded:
        cid = c["id"]
        if cid not in seen:
            ids.append(cid)
            seen.add(cid)

    return ids, meta


def unload():
    global _retriever, _reranker
    if _retriever and hasattr(_retriever, "normalizer"):
        _retriever.normalizer.close()
    _retriever = None
    _reranker = None
