"""
Layered Roman → Devanagari transliteration (Sprint 2).
1. Entity map (exact + fuzzy via entity_resolver)
2. indic-transliteration HK/IAST
3. ai4bharat-transliteration (local, optional)
4. HTTP IndicXlit API (fallback, gated)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import requests
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from config import (
    INDICXLIT_FALLBACK,
    INDICXLIT_TIMEOUT_SEC,
    XLIT_API_URL,
    XLIT_TIMEOUT,
    XLIT_TOP_K,
    XLIT_VALIDATION_THRESHOLD,
)
from entity_resolver import EntityResolution, get_resolver

logger = logging.getLogger(__name__)

_XLIT_ENGINE = None
_XLIT_ENGINE_FAILED = False


def is_valid_devanagari(text: str, threshold: float = XLIT_VALIDATION_THRESHOLD) -> bool:
    if not text:
        return False
    deva = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return deva / max(len(text), 1) > threshold


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


def _get_xlit_engine():
    """Lazy-load AI4Bharat XlitEngine (pip: ai4bharat-transliteration)."""
    global _XLIT_ENGINE, _XLIT_ENGINE_FAILED
    if _XLIT_ENGINE_FAILED:
        return None
    if _XLIT_ENGINE is not None:
        return _XLIT_ENGINE
    try:
        from ai4bharat.transliteration import XlitEngine

        _XLIT_ENGINE = XlitEngine("hi", beam_width=4, rescore=True)
        logger.info("ai4bharat XlitEngine loaded (hi)")
        return _XLIT_ENGINE
    except Exception as e:
        logger.warning(f"ai4bharat-transliteration unavailable: {e}")
        _XLIT_ENGINE_FAILED = True
        return None


def transliterate_hk(token: str) -> Optional[str]:
    if has_devanagari(token):
        return token
    clean = re.sub(r"[^\w]", "", token).lower()
    if len(clean) < 2:
        return None
    for scheme in (sanscript.HK, sanscript.ITRANS, sanscript.VELTHUIS, sanscript.SLP1):
        try:
            deva = transliterate(clean, scheme, sanscript.DEVANAGARI)
            if deva and is_valid_devanagari(deva):
                return deva
        except Exception:
            continue
    return None


def transliterate_ai4bharat(token: str) -> Optional[str]:
    engine = _get_xlit_engine()
    if engine is None:
        return None
    try:
        out = engine.translit_word(token.lower(), topk=1)
        candidates = out.get("hi", []) if isinstance(out, dict) else []
        if candidates and is_valid_devanagari(candidates[0]):
            return candidates[0]
    except Exception as e:
        logger.debug(f"ai4bharat xlit failed for {token!r}: {e}")
    return None


def transliterate_api(token: str) -> List[str]:
    if not INDICXLIT_FALLBACK:
        return []
    try:
        url = XLIT_API_URL.format(word=token.lower())
        timeout = min(XLIT_TIMEOUT, INDICXLIT_TIMEOUT_SEC)
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return []
        data = resp.json()
        candidates = data.get("output", [{}])[0].get("inDataList", [])
        valid = [c for c in candidates[:XLIT_TOP_K] if is_valid_devanagari(c)]
        return valid
    except Exception:
        return []


def transliterate_token(
    token: str,
) -> Tuple[Optional[str], str, Optional[EntityResolution]]:
    """
    Returns (devanagari_or_none, layer_used, entity_resolution).
    layer_used: entity_map | hk | ai4bharat | api | pass_through | already_deva
    """
    clean = re.sub(r"[^\w]", "", token)
    if len(clean) <= 2:
        return None, "skip", None

    if has_devanagari(clean):
        return clean, "already_deva", None

    res = get_resolver().resolve(clean)
    if res and res.devanagari:
        return res.devanagari, "entity_map", res

    hk = transliterate_hk(clean)
    if hk:
        return hk, "hk", None

    ab = transliterate_ai4bharat(clean)
    if ab:
        return ab, "ai4bharat", None

    api_cands = transliterate_api(clean)
    if api_cands:
        return api_cands[0], "api", None

    return None, "pass_through", None


def transliterate_query_tokens(
    query: str,
    stop_words: Optional[set] = None,
) -> Tuple[List[str], dict]:
    """
    Transliterate entity-like roman tokens in a query.
    Returns list of Devanagari tokens + status flags.
    """
    stops = stop_words or set()
    devanagari_out: List[str] = []
    flags = {
        "xlit_layer_used": None,
        "offline_entity_used": False,
        "entity_map_hits": 0,
    }

    for word in query.split():
        clean = re.sub(r"[^\w]", "", word).lower()
        if not clean or clean in stops or len(clean) <= 2:
            continue
        if has_devanagari(word):
            devanagari_out.append(word)
            continue

        deva, layer, res = transliterate_token(word)
        if res:
            flags["offline_entity_used"] = True
            flags["entity_map_hits"] += 1
        if deva:
            devanagari_out.append(deva)
            if flags["xlit_layer_used"] is None:
                flags["xlit_layer_used"] = layer
            elif layer != flags["xlit_layer_used"]:
                flags["xlit_layer_used"] = "mixed"

    return list(dict.fromkeys(devanagari_out)), flags
