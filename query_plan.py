# Query intent routing and per-query plan (wraps multi_query when enabled).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import INTENT_ROUTER_ENABLED
from multi_query import detect_intent, generate_query_variants


@dataclass
class QueryPlan:
    query: str
    script: str = "roman"
    language: str = "en"
    intent: str = "descriptive"
    confidence_band: str = "medium"
    retrievers: List[str] = field(default_factory=lambda: ["faiss", "bm25"])
    rerank_policy: str = "top15"
    rerank_top_n: int = 15
    neighbor_policy: str = "none"
    generation_mode: str = "synthesis"
    status_flags: dict = field(default_factory=dict)
    faiss_top_k: int = 40
    bm25_top_k: int = 40


def _detect_script(query: str) -> str:
    has_deva = any("\u0900" <= c <= "\u097F" for c in query)
    has_latin = any(c.isascii() and c.isalpha() for c in query)
    if has_deva and has_latin:
        return "mixed"
    if has_deva:
        return "devanagari"
    return "roman"


def build_query_plan(
    query: str,
    confidence_band: str = "medium",
    rerank_policy: str = "top15",
    rerank_top_n: int = 15,
    status_flags: dict | None = None,
) -> QueryPlan:
    intent = detect_intent(query)
    script = _detect_script(query)

    if not INTENT_ROUTER_ENABLED:
        return QueryPlan(
            query=query,
            script=script,
            intent=intent,
            confidence_band=confidence_band,
            rerank_policy=rerank_policy,
            rerank_top_n=rerank_top_n,
            neighbor_policy="post_rerank",
            status_flags=status_flags or {},
        )

    if intent == "factoid":
        faiss_k, bm25_k, neighbors = 20, 20, "none"
        gen_mode = "strict"
    elif intent == "causal":
        faiss_k, bm25_k, neighbors = 30, 30, "post_rerank"
        gen_mode = "synthesis"
    elif intent == "philosophical":
        faiss_k, bm25_k, neighbors = 40, 40, "post_rerank"
        gen_mode = "synthesis"
    else:
        faiss_k, bm25_k, neighbors = 30, 30, "post_rerank"
        gen_mode = "synthesis"

    if confidence_band == "high":
        neighbors = "none"

    return QueryPlan(
        query=query,
        script=script,
        intent=intent,
        confidence_band=confidence_band,
        rerank_policy=rerank_policy,
        rerank_top_n=rerank_top_n,
        neighbor_policy=neighbors,
        generation_mode=gen_mode,
        faiss_top_k=faiss_k,
        bm25_top_k=bm25_k,
        status_flags=status_flags or {},
    )


def faiss_variants_for_plan(plan: QueryPlan) -> List[str]:
    """Delegate to multi_query; intent router does not duplicate variant strings."""
    return generate_query_variants(plan.query)
