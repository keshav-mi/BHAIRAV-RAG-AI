"""
Multi-dimensional query classifier — patterns from data/intent_patterns.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from bhairav_data import get_intent_patterns
from chitchat import classify_chitchat_subtype

_COMPILED: dict[str, re.Pattern] = {}


def _pat(name: str) -> Optional[re.Pattern]:
    if name not in _COMPILED:
        raw = get_intent_patterns().get(name)
        if raw:
            _COMPILED[name] = re.compile(raw, re.I)
        else:
            return None
    return _COMPILED.get(name)


@dataclass
class QueryClassification:
    intent: str
    chitchat_subtype: Optional[str] = None
    script: str = "roman"
    language: str = "en"


def detect_script(query: str) -> str:
    has_deva = any("\u0900" <= c <= "\u097F" for c in query)
    has_latin = any(c.isascii() and c.isalpha() for c in query)
    sm = _pat("sanskrit_markers")
    if sm and sm.search(query) and has_deva:
        return "sanskrit"
    if has_deva and has_latin:
        return "mixed"
    if has_deva:
        return "devanagari"
    markers = set(get_intent_patterns().get("hinglish_markers", []))
    if has_latin and set(query.lower().split()) & markers:
        return "roman_hindi"
    return "english"


def detect_language(query: str, script: str) -> str:
    if script in ("devanagari", "sanskrit"):
        return "sa" if script == "sanskrit" else "hi"
    if script in ("roman_hindi", "mixed"):
        return "hinglish"
    return "en"


def classify_query(query: str) -> QueryClassification:
    q = query.strip()
    script = detect_script(q)
    language = detect_language(q, script)

    chitchat_sub = classify_chitchat_subtype(q)
    co = _pat("chitchat_only")
    if chitchat_sub or (co and co.match(q)):
        return QueryClassification(
            intent="chitchat",
            chitchat_subtype=chitchat_sub or "greeting",
            script=script,
            language=language,
        )

    if _pat("lexical") and _pat("lexical").search(q):
        intent = "lexical"
    elif _pat("philosophical") and _pat("philosophical").search(q):
        intent = "philosophical"
    elif _pat("causal") and _pat("causal").search(q):
        intent = "causal"
    elif _pat("factoid") and _pat("factoid").search(q):
        intent = "factoid"
    elif _pat("comparative") and _pat("comparative").search(q):
        intent = "comparative"
    elif _pat("narrative") and _pat("narrative").search(q):
        intent = "narrative"
    else:
        intent = "narrative" if len(q.split()) > 8 else "descriptive"

    return QueryClassification(intent=intent, script=script, language=language)
