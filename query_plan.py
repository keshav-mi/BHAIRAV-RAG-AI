# Query intent routing and per-query plan (Sprint 1).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import INTENT_ROUTER_ENABLED
from multi_query import generate_query_variants
from query_classifier import QueryClassification, classify_query

# Retrieval policies from BHAIRAV_PLAN_V3 §1.3
_INTENT_POLICIES = {
    "chitchat": dict(faiss_k=0, bm25_k=0, rerank="skip", neighbors="none", gen="redirect"),
    "factoid": dict(faiss_k=20, bm25_k=20, rerank="gate", neighbors="none", gen="strict"),
    "causal": dict(faiss_k=30, bm25_k=30, rerank="top15", neighbors="medium", gen="synthesis"),
    "philosophical": dict(faiss_k=40, bm25_k=40, rerank="top15", neighbors="post_rerank", gen="synthesis"),
    "narrative": dict(faiss_k=30, bm25_k=30, rerank="gate", neighbors="post_rerank", gen="synthesis"),
    "lexical": dict(faiss_k=15, bm25_k=15, rerank="skip", neighbors="none", gen="strict"),
    "descriptive": dict(faiss_k=30, bm25_k=30, rerank="gate", neighbors="post_rerank", gen="synthesis"),
}


@dataclass
class QueryPlan:
    query: str
    script: str = "english"
    language: str = "en"
    intent: str = "descriptive"
    chitchat_subtype: str | None = None
    confidence_band: str = "medium"
    faiss_k: int = 40
    bm25_k: int = 40
    rerank_policy: str = "top15"  # skip | gate | top8 | top15
    rerank_top_n: int = 15
    neighbor_policy: str = "post_rerank"  # none | post_rerank | medium
    generation_mode: str = "synthesis"  # redirect | strict | synthesis | conservative
    status_flags: dict = field(default_factory=dict)
    skip_pipeline: bool = False


def build_query_plan(
    query: str,
    confidence_band: str = "medium",
    rerank_policy: str | None = None,
    rerank_top_n: int = 15,
    status_flags: dict | None = None,
) -> QueryPlan:
    clf = classify_query(query)
    flags = dict(status_flags or {})
    flags["language"] = clf.language
    flags["script_detected"] = clf.script

    policy = _INTENT_POLICIES.get(clf.intent, _INTENT_POLICIES["descriptive"])

    if clf.intent == "chitchat":
        return QueryPlan(
            query=query,
            script=clf.script,
            language=clf.language,
            intent="chitchat",
            chitchat_subtype=clf.chitchat_subtype,
            confidence_band=confidence_band,
            rerank_policy="skip",
            rerank_top_n=0,
            neighbor_policy="none",
            generation_mode="redirect",
            status_flags=flags,
            skip_pipeline=True,
        )

    faiss_k = policy["faiss_k"]
    bm25_k = policy["bm25_k"]
    rerank_pol = rerank_policy or policy["rerank"]
    neighbors = policy["neighbors"]

    if confidence_band == "high" and neighbors == "post_rerank":
        neighbors = "none"

    if not INTENT_ROUTER_ENABLED:
        faiss_k = bm25_k = 40
        rerank_pol = "gate"
        neighbors = "post_rerank"

    return QueryPlan(
        query=query,
        script=clf.script,
        language=clf.language,
        intent=clf.intent,
        confidence_band=confidence_band,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        rerank_policy=rerank_pol,
        rerank_top_n=rerank_top_n,
        neighbor_policy=neighbors,
        generation_mode=policy["gen"],
        status_flags=flags,
    )


def apply_confidence_to_plan(plan: QueryPlan, retrieval_meta: dict) -> QueryPlan:
    """Merge FAISS confidence gate into plan after retrieval."""
    band = retrieval_meta.get("confidence_band", plan.confidence_band)
    plan.confidence_band = band
    plan.rerank_top_n = retrieval_meta.get("rerank_top_n", plan.rerank_top_n)

    if plan.rerank_policy == "gate":
        if retrieval_meta.get("skip_rerank"):
            plan.rerank_policy = "skip"
        elif band == "medium":
            plan.rerank_policy = "top8"
        else:
            plan.rerank_policy = "top15"

    if plan.intent == "lexical":
        plan.rerank_policy = "skip"

    if band == "high" and plan.neighbor_policy == "post_rerank":
        plan.neighbor_policy = "none"

    if band == "low" and plan.intent == "causal":
        plan.neighbor_policy = "post_rerank"

    return plan


def faiss_variants_for_plan(plan: QueryPlan) -> List[str]:
    if plan.intent == "lexical":
        return [plan.query]
    return generate_query_variants(plan.query, plan.intent)


def neighbors_enabled(plan: QueryPlan, confidence_band: str) -> bool:
    if plan.neighbor_policy == "none":
        return False
    if plan.neighbor_policy == "medium":
        return confidence_band in ("medium", "low")
    return confidence_band in ("medium", "low", "high")
