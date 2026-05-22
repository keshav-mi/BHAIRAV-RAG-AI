# Query intent routing — policies from data/routing_policies.json

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from bhairav_data import get_routing_policies
from config import INTENT_ROUTER_ENABLED
from multi_query import generate_query_variants
from query_classifier import classify_query


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
    rerank_policy: str = "top15"
    rerank_top_n: int = 15
    neighbor_policy: str = "post_rerank"
    generation_mode: str = "synthesis"
    status_flags: dict = field(default_factory=dict)
    skip_pipeline: bool = False


def _policy(intent: str) -> dict:
    policies = get_routing_policies()
    return policies.get(intent, policies.get("descriptive", {}))


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

    pol = _policy(clf.intent)
    faiss_k = pol.get("faiss_k", 40)
    bm25_k = pol.get("bm25_k", 40)
    rerank_pol = rerank_policy or pol.get("rerank", "gate")
    neighbors = pol.get("neighbors", "post_rerank")
    gen_mode = pol.get("generation_mode", "synthesis")

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
        generation_mode=gen_mode,
        status_flags=flags,
    )


def apply_confidence_to_plan(plan: QueryPlan, retrieval_meta: dict) -> QueryPlan:
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
