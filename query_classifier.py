"""
Multi-dimensional query classifier (Sprint 1.1).
Rule-based: intent type + script + language. No model dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from chitchat import classify_chitchat_subtype

# ── Intent patterns ─────────────────────────────────────────
_CHITCHAT_ONLY = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|namaste|नमस्ते|धन्यवाद)\s*[!.?]*$",
    re.I,
)

_FACTOID = re.compile(
    r"\b(who|whom|when|where|which|what year|kaun|kab|kahan|kisne|kisko)\b|"
    r"(कौन|कब|कहाँ|किसने|किसको|किन)",
    re.I,
)
_CAUSAL = re.compile(
    r"\b(why|how come|because|reason|cause|kyon|kyun|kaise|kya wajah|karan)\b|"
    r"(क्यों|कैसे|कारण|वजह)",
    re.I,
)
_PHILOSOPHICAL = re.compile(
    r"\b(meaning|essence|nature|philosophy|dharma|moksha|atman|brahman)\b|"
    r"(अर्थ|तत्व|स्वरूप|क्या है|भाव|सार|तत्त्व)",
    re.I,
)
_NARRATIVE = re.compile(
    r"\b(story|happened|event|describe|narrate|katha|varnan|kya hua)\b|"
    r"(कथा|वर्णन|क्या हुआ|घटना|वृत्तांत)",
    re.I,
)
_LEXICAL = re.compile(
    r"\b(meaning of|definition of|matlab|shabd|word mean)\b|"
    r"(शब्द|मतलब|परिभाषा|अर्थ क्या)",
    re.I,
)

_SANSKRIT_MARKERS = re.compile(
    r"[\u0900-\u097F].*(ः|म्|त्व|श्र|क्ष|ज्ञ)|"
    r"\b(iti|ca|tu|eva|bhavati|asti)\b",
    re.I,
)


@dataclass
class QueryClassification:
    intent: str  # chitchat | factoid | causal | philosophical | narrative | lexical | descriptive
    chitchat_subtype: Optional[str] = None
    script: str = "roman"  # devanagari | roman_hindi | english | mixed | sanskrit
    language: str = "en"  # hi | sa | en | hinglish


def detect_script(query: str) -> str:
    has_deva = any("\u0900" <= c <= "\u097F" for c in query)
    has_latin = any(c.isascii() and c.isalpha() for c in query)
    if _SANSKRIT_MARKERS.search(query) and has_deva:
        return "sanskrit"
    if has_deva and has_latin:
        return "mixed"
    if has_deva:
        return "devanagari"
    hinglish_markers = {
        "hai", "kya", "ka", "ki", "ke", "ko", "ne", "aur", "mein", "se", "kyon", "kab",
    }
    if has_latin and set(query.lower().split()) & hinglish_markers:
        return "roman_hindi"
    return "english"


def detect_language(query: str, script: str) -> str:
    if script in ("devanagari", "sanskrit"):
        return "sa" if script == "sanskrit" else "hi"
    if script == "roman_hindi" or script == "mixed":
        return "hinglish"
    return "en"


def classify_query(query: str) -> QueryClassification:
    q = query.strip()
    script = detect_script(q)
    language = detect_language(q, script)

    chitchat_sub = classify_chitchat_subtype(q)
    if chitchat_sub or _CHITCHAT_ONLY.match(q):
        return QueryClassification(
            intent="chitchat",
            chitchat_subtype=chitchat_sub or "greeting",
            script=script,
            language=language,
        )

    if _LEXICAL.search(q):
        intent = "lexical"
    elif _PHILOSOPHICAL.search(q):
        intent = "philosophical"
    elif _CAUSAL.search(q):
        intent = "causal"
    elif _FACTOID.search(q):
        intent = "factoid"
    elif _NARRATIVE.search(q):
        intent = "narrative"
    else:
        intent = "narrative" if len(q.split()) > 8 else "descriptive"

    return QueryClassification(intent=intent, script=script, language=language)
