"""
Run Bhairav retrieval at different pipeline stages (for metric comparison).
"""

from __future__ import annotations

from typing import List, Literal, Tuple

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
    Returns ordered chunk IDs at the chosen pipeline stage.
    """
    retriever, reranker = load_components()

    candidates, faiss_q, bm25_q, domains = retriever.retrieve(query, top_k=top_k)
    meta = {"faiss_query": faiss_q, "bm25_query": bm25_q, "domains": domains}

    if stage == "after_retrieve":
        return [c["id"] for c in candidates], meta

    script = _script(query)
    top_chunks = reranker.rerank(
        query,
        candidates,
        top_n=rerank_top_n,
        script=script,
    )

    if stage == "after_rerank":
        return [c["id"] for c in top_chunks], meta

    # after_neighbors — mirrors main.py
    reranked_ids = [(c["id"], c.get("rerank_score", 1.0)) for c in top_chunks]
    expanded = retriever.get_neighbor_chunks(reranked_ids, window=1)

    seen = set()
    ids = []
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
