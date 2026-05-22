"""Chitchat short-circuit — responses and patterns from data/chitchat.json."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from bhairav_data import get_chitchat_config

_COMPILED: dict[str, re.Pattern] = {}


def _patterns() -> dict[str, str]:
    cfg = get_chitchat_config()
    return cfg.get("patterns", {})


def _responses() -> dict[str, str]:
    cfg = get_chitchat_config()
    return cfg.get("responses", {})


def _get_pattern(name: str) -> Optional[re.Pattern]:
    if name not in _COMPILED:
        raw = _patterns().get(name)
        if raw:
            _COMPILED[name] = re.compile(raw, re.I)
        else:
            return None
    return _COMPILED.get(name)


def classify_chitchat_subtype(query: str) -> Optional[str]:
    q = query.strip()
    if not q or len(q.split()) > 12:
        return None

    for subtype in ("greeting", "identity", "thanks", "farewell"):
        pat = _get_pattern(subtype)
        if pat and pat.search(q):
            return subtype
    return None


def chitchat_response(query: str) -> Optional[Tuple[str, str]]:
    subtype = classify_chitchat_subtype(query)
    only = _get_pattern("chitchat_only")
    if subtype is None and not (only and only.match(query.strip())):
        return None

    responses = _responses()
    subtype = subtype or "greeting"
    text = responses.get(subtype, responses.get("fallback", ""))
    return subtype, text


# Back-compat for main.py
def get_chitchat_responses() -> dict[str, str]:
    return _responses()
