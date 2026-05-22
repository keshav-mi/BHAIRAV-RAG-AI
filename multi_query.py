"""Intent-aware FAISS query variants — templates from data/query_variants.json."""
from __future__ import annotations
from typing import List
from bhairav_data import get_query_variants
from query_classifier import classify_query


def generate_query_variants(base_query: str, intent: str | None = None) -> List[str]:
    q = base_query.strip()
    if intent is None:
        intent = classify_query(q).intent

    cfg = get_query_variants()
    block = cfg.get(intent) or cfg.get("descriptive")
    if not block:
        return [q]

    templates = block.get("templates", ["{query}"])
    seen = []
    for t in templates:
        variant = t.replace("{query}", q)
        if variant not in seen:
            seen.append(variant)
    return seen
