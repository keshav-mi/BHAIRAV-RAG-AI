# ============================================================
# BHAIRAV AI — QUERY EXPANDER v4
# ============================================================
# ARCHITECTURE:
#   Layer 0: Script detection — route Devanagari vs Romanized
#   Layer 1: IndicXlit API   — Romanized → Devanagari
#   Layer 2: Entity synonyms — alternate Sanskrit epithets only
#   Layer 3: Groq expansion  — only when Devanagari still missing
#   Layer 4: Domain detection — for RRF soft boosting
#
# KEY FIX vs v3:
#   - BM25 receives BOTH original + expanded query (union search)
#   - FAISS receives the expanded query for semantic search
#   - Domain detection now also checks Devanagari tokens directly
#   - Expansion is strictly additive — never replaces original
# ============================================================

import re
import json
from pathlib import Path

import requests
from groq import Groq
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from rapidfuzz import process, fuzz
from typing import Tuple, List

from config import (
    DATA_DIR,
    GROQ_MODEL,
    INDICXLIT_FALLBACK,
    INDICXLIT_TIMEOUT_SEC,
    REQUIRE_HARVESTED_MAPS,
    XLIT_API_URL,
    XLIT_TIMEOUT,
    XLIT_TOP_K,
)


# ── Entity synonym map ──────────────────────────────────────
# Alternate names / Sanskrit epithets ONLY.
# Rule: these must appear in the actual texts, not inferred relations.
ENTITY_SYNONYMS = {
    # Ramayana
    "राम"       : ["रामचंद्र", "दशरथनंदन", "रघुनंदन", "मर्यादा पुरुषोत्तम"],
    "सीता"      : ["जानकी", "वैदेही", "मिथिलेश्वरी", "सीतादेवी"],
    "हनुमान"    : ["पवनपुत्र", "मारुतिनंदन", "बजरंगबली", "अंजनेय"],
    "रावण"      : ["दशानन", "लंकापति", "लंकेश", "दशकंठ"],
    "लक्ष्मण"   : ["सौमित्र", "शेषावतार"],
    "दशरथ"      : ["दशरथराज", "अयोध्यापति"],
    "सुग्रीव"   : ["वानरराज"],
    "विभीषण"    : ["लंकाविभीषण"],

    # Mahabharata
    "अर्जुन"    : ["पार्थ", "धनंजय", "किरीटी", "सव्यसाची", "गांडीवधारी"],
    "अभिमन्यु"  : ["सौभद्र", "अर्जुनपुत्र"],
    "भीष्म"     : ["देवव्रत", "गाङ्गेय", "भीष्मपितामह", "शांतनवपुत्र"],
    "कृष्ण"     : ["माधव", "केशव", "गोविंद", "वासुदेव", "द्वारकाधीश", "मुरारी"],
    "द्रौपदी"   : ["पांचाली", "यज्ञसेनी", "कृष्णा", "द्रुपदकन्या"],
    "युधिष्ठिर" : ["धर्मराज", "अजातशत्रु", "धर्मपुत्र"],
    "दुर्योधन"  : ["सुयोधन", "कौरवेंद्र"],
    "कर्ण"      : ["सूर्यपुत्र", "राधेय", "वसुषेण", "अंगराज"],
    "भीम"       : ["वृकोदर", "भीमसेन", "महाबली"],
    "कुंती"     : ["पृथा", "कुंतीदेवी"],
    "द्रोण"     : ["द्रोणाचार्य", "गुरु द्रोण"],
    "सुभद्रा"   : ["सुभद्रादेवी"],

    # Vedic deities
    "इंद्र"     : ["शक्र", "पुरंदर", "वज्रपाणि", "देवराज", "इंद्रदेव"],
    "अग्नि"     : ["वैश्वानर", "पावक", "हुताशन", "अग्निदेव"],
    "सोम"       : ["सोमदेव", "चंद्रदेव", "शशांक"],
    "सूर्य"     : ["आदित्य", "भास्कर", "दिनकर", "रवि", "सूर्यदेव"],
    "विष्णु"    : ["नारायण", "हरि", "जनार्दन", "अच्युत", "केशव"],
    "शिव"       : ["महादेव", "शंकर", "नीलकंठ", "भोलेनाथ", "त्र्यंबक"],
    "ब्रह्मा"   : ["प्रजापति", "विधाता", "चतुर्मुख"],
    "गणेश"      : ["गणपति", "विनायक", "लंबोदर", "एकदंत"],
    "सरस्वती"   : ["वाग्देवी", "शारदा", "भारती"],
    "वरुण"      : ["जलाधिपति", "वरुणदेव"],
}

# Reverse lookup: any known name → all synonyms including canonical
SYNONYM_LOOKUP: dict = {}
for _canonical, _synonyms in ENTITY_SYNONYMS.items():
    SYNONYM_LOOKUP[_canonical] = _synonyms
    for _syn in _synonyms:
        if _syn not in SYNONYM_LOOKUP:
            SYNONYM_LOOKUP[_syn] = [_canonical] + [s for s in _synonyms if s != _syn]

# ─────────────────────────────────────────────────────────────
# HARVESTED MAP LOADER
# ─────────────────────────────────────────────────────────────

ENGLISH_ENTITY_MAP = {}

# Seed roman → Devanagari when harvested maps are absent (BUG-1 quick wins)
ENTITY_ROMAN_SEED = {
    "ekalavya": "एकलव्य",
    "eklavaya": "एकलव्य",
    "arjuna": "अर्जुन",
    "arjun": "अर्जुन",
    "krishna": "कृष्ण",
    "bhishma": "भीष्म",
    "bhishm": "भीष्म",
    "draupadi": "द्रौपदी",
    "rama": "राम",
    "ram": "राम",
    "sita": "सीता",
    "hanuman": "हनुमान",
}


def transliterate_roman_token(token: str) -> str | None:
    """
    Deterministic HK/ITRANS transliteration for a single roman entity token.
    Mirrors query_normalizer.transliterate_entities (P3).
    """
    if any("\u0900" <= ch <= "\u097F" for ch in token):
        return token

    clean = re.sub(r"[^\w]", "", token).lower()
    if clean in ENTITY_ROMAN_SEED:
        return ENTITY_ROMAN_SEED[clean]
    if clean in ENGLISH_ENTITY_MAP:
        return ENGLISH_ENTITY_MAP[clean]

    for scheme in (sanscript.HK, sanscript.ITRANS, sanscript.VELTHUIS, sanscript.SLP1):
        try:
            deva = transliterate(clean, scheme, sanscript.DEVANAGARI)
            if deva and any("\u0900" <= ch <= "\u097F" for ch in deva):
                return deva
        except Exception:
            continue
    return None


def load_harvested_maps():
    """
    Loads harvested entity + epithet maps.

    Silent failure by design:
    If files are missing/corrupt, system continues using
    hardcoded fallback maps.
    """
    global SYNONYM_LOOKUP
    global ENGLISH_ENTITY_MAP

    entity_map_path = DATA_DIR / "entity_map_clean.json"
    epithet_map_path = DATA_DIR / "epithet_map.json"
    entity_loaded = epithet_loaded = False

    try:
        # ── ENTITY MAP ──────────────────────────────────────
        map_file = entity_map_path if entity_map_path.exists() else DATA_DIR / "entity_map_seed.json"
        if map_file.exists():
            entity_loaded = True
            with open(map_file, "r", encoding="utf-8") as f:
                entity_data = json.load(f)

            for english_name, payload in entity_data.items():
                devanagari = payload.get("devanagari", "").strip()

                if not devanagari:
                    continue

                ENGLISH_ENTITY_MAP[english_name.lower()] = devanagari

                # Add both directions into synonym lookup
                if devanagari not in SYNONYM_LOOKUP:
                    SYNONYM_LOOKUP[devanagari] = []

                existing = set(SYNONYM_LOOKUP[devanagari])
                existing.add(english_name)
                SYNONYM_LOOKUP[devanagari] = list(existing)

        # ── EPITHET MAP ─────────────────────────────────────
        if epithet_map_path.exists():
            epithet_loaded = True
            with open(epithet_map_path, "r", encoding="utf-8") as f:
                epithet_data = json.load(f)

            for epithet, payload in epithet_data.items():
                canonical = payload.get("canonical", "").strip()

                if not canonical:
                    continue

                if epithet not in SYNONYM_LOOKUP:
                    SYNONYM_LOOKUP[epithet] = []

                existing = set(SYNONYM_LOOKUP[epithet])
                existing.add(canonical)
                SYNONYM_LOOKUP[epithet] = list(existing)

                # Reverse direction
                if canonical not in SYNONYM_LOOKUP:
                    SYNONYM_LOOKUP[canonical] = []

                reverse_existing = set(SYNONYM_LOOKUP[canonical])
                reverse_existing.add(epithet)
                SYNONYM_LOOKUP[canonical] = list(reverse_existing)

        for roman, deva in ENTITY_ROMAN_SEED.items():
            ENGLISH_ENTITY_MAP.setdefault(roman, deva)

        print(
            f"Harvested maps | entity_json={entity_loaded} epithet_json={epithet_loaded} | "
            f"synonyms={len(SYNONYM_LOOKUP):,} english_entities={len(ENGLISH_ENTITY_MAP):,}"
        )

        if REQUIRE_HARVESTED_MAPS and not (entity_loaded and epithet_loaded):
            raise FileNotFoundError(
                f"REQUIRE_HARVESTED_MAPS=true but missing {entity_map_path} or {epithet_map_path}. "
                "See data/README.md"
            )

    except Exception as e:
        if REQUIRE_HARVESTED_MAPS:
            raise
        print(f"Harvested map loading skipped: {e}")
        for roman, deva in ENTITY_ROMAN_SEED.items():
            ENGLISH_ENTITY_MAP.setdefault(roman, deva)


load_harvested_maps()

# ── Domain signals ──────────────────────────────────────────
DOMAIN_SIGNALS = {
    "Valmiki Ramayana": [
        "राम", "रामचंद्र", "सीता", "लक्ष्मण", "हनुमान", "रावण",
        "कैकेयी", "दशरथ", "वनवास", "अयोध्या", "लंका", "सुग्रीव",
        "ram", "rama", "sita", "laxman", "hanuman", "ravan",
        "kaikeyi", "dashrath", "vanvas", "ayodhya", "lanka",
    ],
    "Ramcharitmanas": [
        "तुलसीदास", "मानस", "चौपाई", "दोहा",
        "tulsi", "tulsidas", "manas", "chaupai", "ramcharitmanas",
    ],
    "Mahabharata": [
        "अर्जुन", "कृष्ण", "भीष्म", "कुरुक्षेत्र", "पांडव",
        "कौरव", "द्रौपदी", "युधिष्ठिर", "दुर्योधन", "कर्ण",
        "अभिमन्यु", "सौभद्र", "सुभद्रा",
        "arjun", "arjuna", "krishna", "bhishma", "kurukshetra",
        "pandav", "kaurav", "draupadi", "yudhishthir", "karna",
        "abhimanyu", "saubhadra", "subhadra", "mahabharata",
    ],
    "Bhagavad Gita": [
        "गीता", "कर्म योग", "भक्ति योग", "ज्ञान योग",
        "gita", "geeta", "karma yoga", "bhakti yoga",
    ],
    "Rigveda": [
        "इंद्र", "अग्नि", "सोम", "सूक्त", "मंडल",
        "indra", "agni", "soma", "sukta", "mandala", "rigveda",
    ],
    "Atharvaveda": [
        "अथर्ववेद", "काण्ड", "atharvaveda", "atharva", "kaanda",
    ],
}


# ─────────────────────────────────────────────────────────────
# LAYER 0 — Script detection helpers
# ─────────────────────────────────────────────────────────────

def has_devanagari(text: str) -> bool:
    return any('\u0900' <= c <= '\u097F' for c in text)

def is_romanized_word(word: str) -> bool:
    clean = re.sub(r'[^\w]', '', word).lower()
    return bool(clean) and not has_devanagari(clean)

STOP_WORDS = {
    "ka", "ki", "ke", "ko", "hai", "tha", "the", "kya", "aur",
    "ya", "se", "me", "mein", "ne", "par", "who", "what", "why",
    "how", "did", "was", "is", "are", "of", "the", "a", "an",
    "in", "on", "at", "to", "for", "bete", "putra", "mata",
}


# ─────────────────────────────────────────────────────────────
# LAYER 1 — IndicXlit transliteration
# ─────────────────────────────────────────────────────────────

def transliterate_query(query: str) -> Tuple[str, List[str]]:
    """
    Layered transliteration (entity map → HK → ai4bharat → API).
    Returns (expanded_query, list_of_devanagari_tokens). Original preserved.
    """
    from transliteration import transliterate_query_tokens

    devanagari_found, xlit_flags = transliterate_query_tokens(query, STOP_WORDS)
    if devanagari_found:
        expanded = query + " " + " ".join(devanagari_found)
        return expanded, devanagari_found
    return query, []

# ─────────────────────────────────────────────────────────────
# LAYER 2 — Entity synonym expansion
# ─────────────────────────────────────────────────────────────

def expand_entity_synonyms(tokens: List[str]) -> List[str]:
    """
    For each Devanagari token, expand to known alternate epithets.
    Strict: only alternate names, never narrative relations.
    """
    extras: List[str] = []

    for token in tokens:
        if token in SYNONYM_LOOKUP:
            extras.extend(SYNONYM_LOOKUP[token][:3])
            continue

        # Fuzzy fallback — only high-confidence matches
        all_keys = list(SYNONYM_LOOKUP.keys())
        result = process.extractOne(
            token, all_keys, scorer=fuzz.ratio, score_cutoff=90
        )
        if result:
            extras.extend(SYNONYM_LOOKUP[result[0]][:3])

    return list(set(extras))


# ─────────────────────────────────────────────────────────────
# LAYER 3 — Groq expansion (last resort)
# ─────────────────────────────────────────────────────────────

def groq_expand(query: str, client: Groq) -> str:
    """
    Groq-based expansion.
    Only called when no Devanagari at all after layers 1+2.
    Prompt is strict: synonyms/alternate names only.
    """
    if has_devanagari(query):
        return query

    prompt = (
        "You are assisting a Sanskrit/Hindi search system.\n"
        "Given the romanized query below, output the ORIGINAL query followed by:\n"
        "  - Hindi/Devanagari script version of any proper nouns (names, places)\n"
        "  - Common Sanskrit synonyms/epithets for those names\n\n"
        "STRICT RULES:\n"
        "- Do NOT add family relationships, story context, or answers\n"
        "- Do NOT add words not already present in the query\n"
        "- Output: original query + space + Hindi expansions ONLY\n"
        "- No explanation, no punctuation, no markdown\n\n"
        f"Query: {query}\nOutput:"
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        expanded = resp.choices[0].message.content.strip()
        # Safety: reject if output inflated too much (LLM leaked relations)
        if len(expanded.split()) > len(query.split()) * 5:
            return query
        return expanded
    except Exception:
        return query


# ─────────────────────────────────────────────────────────────
# LAYER 4 — Domain detection
# ─────────────────────────────────────────────────────────────

def detect_domains(query: str, tokens: List[str]) -> List[str]:
    """
    Soft domain detection for RRF boosting.
    Checks original query + transliterated tokens.
    """
    combined = (query + " " + " ".join(tokens)).lower()
    matched: List[str] = []

    for source, signals in DOMAIN_SIGNALS.items():
        for signal in signals:
            if signal.lower() in combined:
                matched.append(source)
                break

    return matched


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def expand_query(query: str, client: Groq) -> Tuple[str, str, List[str]]:
    """
    Full pipeline. Returns:
        faiss_query     : expanded query for semantic (FAISS) search
        bm25_query      : original + devanagari tokens (for lexical BM25)
        detected_domains: source names for RRF soft boost

    WHY TWO QUERIES?
        FAISS (semantic) benefits from rich expansion.
        BM25 (lexical) must also match the original query's exact tokens,
        so we build a union string: original + Devanagari additions only.
        Synonym flood hurts BM25 precision but helps FAISS recall.
    """
    query = query.strip()

    # Layer 1: Romanized → Devanagari
    faiss_expanded, deva_tokens = transliterate_query(query)

    # Layer 2: Entity synonyms
    synonyms = expand_entity_synonyms(deva_tokens)
    if synonyms:
        faiss_expanded = faiss_expanded + " " + " ".join(synonyms)

    # Layer 3: Groq fallback (only if still all-roman)
    faiss_expanded = groq_expand(faiss_expanded, client)

    # BM25 query: original + devanagari tokens only (not full synonyms)
    bm25_query = query
    if deva_tokens:
        bm25_query = query + " " + " ".join(deva_tokens)

    # Layer 4: Domain detection
    domains = detect_domains(query, deva_tokens)

    return faiss_expanded, bm25_query, domains
