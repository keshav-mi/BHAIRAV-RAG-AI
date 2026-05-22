# ============================================================
# BHAIRAV AI — QUERY EXPANDER v5
# No hardcoded entity/synonym dicts — data/*.json + entity_resolver + transliteration
# ============================================================

from __future__ import annotations

import re
from typing import List, Tuple

from groq import Groq
from rapidfuzz import fuzz, process

from bhairav_data import get_domain_signals, get_stopwords, load_prompt_optional
from config import GROQ_EXPAND_ENABLED, GROQ_MODEL
from entity_resolver import get_resolver
from transliteration import transliterate_query_tokens


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


def transliterate_query(query: str) -> Tuple[str, List[str]]:
    """Entity map → HK → ai4bharat → API (see transliteration.py)."""
    stops = get_stopwords()
    devanagari_found, _ = transliterate_query_tokens(query, stops)
    if devanagari_found:
        return query + " " + " ".join(devanagari_found), devanagari_found
    return query, []


def expand_entity_synonyms(tokens: List[str]) -> List[str]:
    return get_resolver().expand_synonyms(tokens)


def groq_expand(query: str, client: Groq) -> str:
    if has_devanagari(query):
        return query

    prompt = load_prompt_optional(
        "query_expand",
        "Query: {query}\nOutput:",
        query=query,
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        expanded = resp.choices[0].message.content.strip()
        if len(expanded.split()) > len(query.split()) * 5:
            return query
        return expanded
    except Exception:
        return query


def detect_domains(query: str, tokens: List[str]) -> List[str]:
    combined = (query + " " + " ".join(tokens)).lower()
    matched: List[str] = []
    for source, signals in get_domain_signals().items():
        for signal in signals:
            if signal.lower() in combined:
                matched.append(source)
                break
    return matched


def expand_query(query: str, client: Groq) -> Tuple[str, str, List[str]]:
    query = query.strip()

    faiss_expanded, deva_tokens = transliterate_query(query)

    synonyms = expand_entity_synonyms(deva_tokens)
    if synonyms:
        faiss_expanded = faiss_expanded + " " + " ".join(synonyms)

    if GROQ_EXPAND_ENABLED:
        faiss_expanded = groq_expand(faiss_expanded, client)

    bm25_query = query
    if deva_tokens:
        bm25_query = query + " " + " ".join(deva_tokens)

    domains = detect_domains(query, deva_tokens)
    return faiss_expanded, bm25_query, domains
