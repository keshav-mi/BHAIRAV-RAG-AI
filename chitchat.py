"""Chitchat short-circuit responses (Sprint 1.2)."""

from __future__ import annotations

import re
from typing import Optional, Tuple

CHITCHAT_RESPONSES = {
    "greeting": (
        "नमस्ते! मैं भैरव हूँ — हिंदू धर्म के प्राथमिक ग्रंथों का AI सहायक। "
        "महाभारत, रामायण, वेद, या भगवद्गीता से कोई प्रश्न पूछें।"
    ),
    "identity": (
        "मैं भैरव हूँ, एक RAG-आधारित AI जो हिंदू धार्मिक ग्रंथों — महाभारत, रामायण, "
        "ऋग्वेद, अथर्ववेद, यजुर्वेद, और भगवद्गीता — पर प्रश्नों का उत्तर देता है।"
    ),
    "thanks": "आपका स्वागत है। कोई और प्रश्न हो तो पूछें।",
    "farewell": "धन्यवाद। फिर मिलें।",
    "fallback": "कृपया धार्मिक ग्रंथों से संबंधित प्रश्न पूछें।",
}

_GREETING = re.compile(
    r"^(hi|hello|hey|namaste|namaskar|नमस्ते|नमस्कार|hii+)\b",
    re.I,
)
_IDENTITY = re.compile(
    r"(who are you|what are you|tum kaun|aap kaun|आप कौन|तुम कौन|"
    r"introduce yourself|about yourself)",
    re.I,
)
_THANKS = re.compile(
    r"^(thanks|thank you|dhanyavad|धन्यवाद|शुक्रिया)\b",
    re.I,
)
_FAREWELL = re.compile(
    r"^(bye|goodbye|alvida|फिर मिलेंगे)\b",
    re.I,
)


def classify_chitchat_subtype(query: str) -> Optional[str]:
    q = query.strip()
    if not q or len(q.split()) > 12:
        return None
    if _GREETING.search(q):
        return "greeting"
    if _IDENTITY.search(q):
        return "identity"
    if _THANKS.search(q):
        return "thanks"
    if _FAREWELL.search(q):
        return "farewell"
    return None


def chitchat_response(query: str) -> Optional[Tuple[str, str]]:
    """Return (subtype, response_text) or None if not chitchat."""
    subtype = classify_chitchat_subtype(query)
    if subtype is None:
        return None
    return subtype, CHITCHAT_RESPONSES[subtype]
