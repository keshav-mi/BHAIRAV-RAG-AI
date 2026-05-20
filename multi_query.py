"""Intent-aware FAISS query variants (Sprint 1)."""

from __future__ import annotations

from typing import List

# Legacy triggers kept for backward-compatible detect_intent()
CAUSAL_TRIGGERS = {
    "why", "kyon", "cause", "reason", "karan", "how", "kaise",
    "why did", "kyun", "kya wajah", "kya karan",
}
FACTUAL_TRIGGERS = {
    "who", "kaun", "what", "kya", "when", "kab", "where", "kahan",
    "which", "kon", "kis", "whom", "kisne",
}


def detect_intent(query: str) -> str:
    """Delegate to query_classifier for consistent labels."""
    from query_classifier import classify_query

    clf = classify_query(query)
    if clf.intent == "chitchat":
        return "chitchat"
    if clf.intent in ("factoid", "lexical"):
        return "factual"
    if clf.intent == "causal":
        return "causal"
    return "descriptive"


def generate_query_variants(base_query: str, intent: str | None = None) -> list[str]:
    """
    Semantic variants by intent (plan §1.3 / legacy multi_query).
    """
    q = base_query.strip()
    if intent is None:
        intent = detect_intent(q)

    if intent in ("causal",):
        return [
            q,
            q + " cause reason wajah karan motivation",
            q + " history background event backstory shraap",
        ]

    if intent in ("factual", "factoid", "lexical"):
        return [q, q + " description detail story role"]

    if intent in ("philosophical", "narrative"):
        return [q, q + " essence meaning context pramana"]

    return [q]
