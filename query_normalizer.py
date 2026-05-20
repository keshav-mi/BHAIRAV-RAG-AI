"""
Bhairav AI — Query Normalization Layer v2
=========================================
Pipeline:
    G0: Gemini  — extract entity name tokens from any language (Hi/En/Hinglish)
    P1: SQLite  — cache check on extracted entity
    P2: Wikidata — alias resolution (primary)
    P3: indic-transliteration — script normalization on entity tokens only
    P4: Monier-Williams (Cologne API) — epithet/meaning queries only
    
All steps augment the original query — nothing is replaced.
"""

import os
import sqlite3
import json
import re
import logging
import requests
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from config import BASE_DIR, GEMINI_ENTITY_ENABLED, MW_NETWORK_ENABLED, WIKIDATA_ENABLED
from entity_resolver import get_resolver

logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DB_PATH = BASE_DIR / "entity_cache.db"

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
COLOGNE_MW_URL      = "https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/web/webtc/getword.php"

WIKIDATA_HEADERS = {
    "User-Agent": "BhairavAI/2.0 (Dharmic RAG research project; scholarly use)"
}

DHARMIC_KEYWORDS = {
    "hindu", "sanskrit", "mahabharata", "ramayana", "vedic",
    "purana", "epic", "mythology", "dharmic", "bhagavad",
    "upanishad", "character", "king", "sage", "rishi", "deity"
}

# Epithet pattern triggers for P4 gate
# Epithet pattern triggers for P4 gate (Strict markers only)
EPITHET_PATTERNS = [
    # English
    r"\bson of\b", r"\bdaughter of\b", r"\bmeaning of\b", r"\bwhat does\b", r"\bepithet\b",
    # Hindi
    r"पुत्र", r"पुत्री", r"का अर्थ", r"नाम का मतलब",
    # Hinglish
    r"\bputra\b", r"\bputri\b"
]
EPITHET_REGEX = re.compile("|".join(EPITHET_PATTERNS), re.IGNORECASE)

# ─────────────────────────────────────────────
# Gemini Setup
# ─────────────────────────────────────────────

def _init_gemini() -> Optional[genai.Client]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — entity extraction disabled")
        return None
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────
# G0 — Gemini Entity Extraction
# ─────────────────────────────────────────────

GEMINI_PROMPT = """You are a named entity extractor for Dharmic/Hindu texts.

Extract ONLY proper noun entity names (people, deities, sages, places, texts) from the query.
The query can be in Hindi, English, or Hinglish (Hindi written in Latin script).

Rules:
- Return ONLY a JSON array of strings, nothing else
- No preamble, no explanation, no markdown backticks
- Extract names exactly as they appear in the query — do not translate or normalize
- If no named entities exist, return []
- Maximum 5 entities

Examples:
Query: "who was devovrath" → ["devovrath"]
Query: "shantanu ke kitne putr the" → ["shantanu"]
Query: "arjun aur krishna ka sambandh" → ["arjun", "krishna"]
Query: "draupadi swayamvar mein kya hua" → ["draupadi"]
Query: "what is dharma" → []
Query: "शान्तनु के पुत्र कौन थे" → ["शान्तनु"]
Query: "who took the terrible vow" → []
Query: "ram ne ravan ko kyon mara" → ["ram", "ravan"]

Query: "{query}"
"""

def extract_entities_gemini(query: str, client: genai.Client) -> list[str]:
    """Uses Gemini to extract entity name tokens from any language query."""
    try:
        prompt = GEMINI_PROMPT.format(query=query)
        try:
            gen_config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except (TypeError, AttributeError):
            gen_config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
            )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gen_config,
        )

        if not response.text:
            return []

        raw = response.text.strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not match:
            return []
        return json.loads(match.group(0))

    except Exception as e:
        logger.warning(f"Gemini entity extraction failed: {e}")
        return []

# ─────────────────────────────────────────────
# P1 — SQLite Cache
# ─────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_cache (
            entity      TEXT PRIMARY KEY,
            aliases     TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, entity: str) -> Optional[list[str]]:
    row = conn.execute(
        "SELECT aliases FROM entity_cache WHERE entity = ?",
        (entity.lower().strip(),)
    ).fetchone()
    return json.loads(row[0]) if row else None


def cache_set(conn: sqlite3.Connection, entity: str, aliases: list[str], source: str):
    conn.execute(
        "INSERT OR REPLACE INTO entity_cache (entity, aliases, source) VALUES (?, ?, ?)",
        (entity.lower().strip(), json.dumps(aliases, ensure_ascii=False), source)
    )
    conn.commit()


# ─────────────────────────────────────────────
# P2 — Wikidata Entity Resolution
# ─────────────────────────────────────────────

def offline_extract_entities(query: str) -> list[str]:
    """
    When Gemini is unavailable: extract likely proper-noun tokens from the query.
    """
    stop = {
        "who", "what", "when", "where", "why", "how", "which", "the", "a", "an",
        "is", "was", "were", "are", "did", "does", "do", "in", "on", "at", "to",
        "for", "of", "and", "or", "ka", "ki", "ke", "ko", "kya", "hai", "tha",
        "the", "about", "tell", "me", "story", "of",
    }
    tokens = []
    for word in re.findall(r"[\w']+", query):
        w = word.strip("'")
        low = w.lower()
        if low in stop or len(low) < 3:
            continue
        if any("\u0900" <= ch <= "\u097F" for ch in w):
            tokens.append(w)
        elif w[0].isupper() or len(w) >= 4:
            tokens.append(w)
    return list(dict.fromkeys(tokens))[:5]


def wikidata_search(entity: str) -> Optional[str]:
    if not WIKIDATA_ENABLED:
        return None
    """
    Search Wikidata for entity QID.
    Appends 'Hindu mythology' to bias toward Dharmic results.
    Only returns QID if description matches Dharmic keywords.
    """
    try:
        enriched = f"{entity} Hindu mythology"
        resp = requests.get(
            WIKIDATA_SEARCH_URL,
            params={
                "action": "wbsearchentities",
                "search": enriched,
                "language": "en",
                "format": "json",
                "limit": 5,
                "type": "item"
            },
            headers=WIKIDATA_HEADERS,
            timeout=6
        )
        results = resp.json().get("search", [])
        if not results:
            return None

        # Only return QID if description is clearly Dharmic
        for r in results:
            desc = r.get("description", "").lower()
            if any(kw in desc for kw in DHARMIC_KEYWORDS):
                return r["id"]

        # No Dharmic match found — return None rather than wrong QID
        return None

    except Exception as e:
        logger.warning(f"Wikidata search failed: {e}")
        return None


def wikidata_aliases(qid: str) -> list[str]:
    """Fetch all labels + aliases in en, hi, sa for a given QID."""
    try:
        resp = requests.get(
            WIKIDATA_ENTITY_URL.format(qid=qid),
            headers=WIKIDATA_HEADERS,
            timeout=6
        )
        entity = resp.json()["entities"][qid]
        collected = []

        for lang in ["en", "hi", "sa"]:
            label = entity.get("labels", {}).get(lang, {}).get("value")
            if label:
                collected.append(label)
            for alias in entity.get("aliases", {}).get(lang, []):
                collected.append(alias["value"])

        return list(dict.fromkeys(collected))
    except Exception as e:
        logger.warning(f"Wikidata entity fetch failed for {qid}: {e}")
        return []


def resolve_wikidata(entity: str) -> tuple[list[str], bool]:
    if not WIKIDATA_ENABLED:
        return [], False
    qid = wikidata_search(entity)
    if not qid:
        return [], False
    aliases = wikidata_aliases(qid)
    if not aliases:
        return [], False
    logger.info(f"Wikidata: '{entity}' → {aliases}")
    return aliases, True


# ─────────────────────────────────────────────
# P3 — indic-transliteration (entity tokens only)
# ─────────────────────────────────────────────

def transliterate_entities(entities: list[str]) -> list[str]:
    """
    Attempts transliteration on extracted entity tokens only.
    Skips tokens already in Devanagari.
    Returns list of Devanagari variants.
    """
    variants = []
    for token in entities:
        # Already Devanagari — skip
        if any("\u0900" <= ch <= "\u097F" for ch in token):
            continue

        for scheme in [sanscript.ITRANS, sanscript.HK, sanscript.VELTHUIS, sanscript.SLP1]:
            try:
                deva = transliterate(token, scheme, sanscript.DEVANAGARI)
                if deva and any("\u0900" <= ch <= "\u097F" for ch in deva):
                    variants.append(deva)
                    break
            except Exception:
                continue

    return variants


# ─────────────────────────────────────────────
# P4 Gate + Monier-Williams
# ─────────────────────────────────────────────

def is_epithet_query(query: str) -> bool:
    return bool(EPITHET_REGEX.search(query))


def monier_williams_lookup(query: str) -> list[str]:
    """
    Reverse meaning lookup via Cologne MW API.
    Only called when no entity resolved AND query is epithet/meaning based.
    """
    if not MW_NETWORK_ENABLED:
        return []
    stop_words = {
        "who", "what", "is", "was", "the", "a", "an", "of",
        "in", "to", "and", "or", "did", "does", "how", "why",
        "took", "born", "from", "son", "daughter"
    }
    content_words = [
        w for w in query.lower().split()
        if w not in stop_words and len(w) > 3
    ]

    collected = []
    for word in content_words[:3]:
        try:
            resp = requests.get(
                COLOGNE_MW_URL,
                params={"key": word, "filter": "roman"},
                timeout=5
            )
            text = resp.text
            sanskrit_terms   = re.findall(r"<s>(.*?)</s>", text)
            devanagari_terms = re.findall(r"[\u0900-\u097F]+", text)
            collected.extend(sanskrit_terms[:5])
            collected.extend(devanagari_terms[:5])
        except Exception as e:
            logger.warning(f"Monier-Williams lookup failed for '{word}': {e}")

    return list(dict.fromkeys(collected))


# ─────────────────────────────────────────────
# Main Normalizer
# ─────────────────────────────────────────────

class QueryNormalizer:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn   = init_db(db_path)
        self.gemini = _init_gemini()

    def normalize(self, raw_query: str, offline_first: bool = True) -> dict:
        """
        Returns:
        {
            "original"      : str,
            "augmented"     : str,        ← send this to BM25 + FAISS
            "expansions"    : list[str],  ← all collected expansion terms
            "entities"      : list[str],  ← what Gemini extracted
            "name_resolved" : bool,
            "sources_used"  : list[str]
        }
        """
        query = raw_query.strip()
        expansions: list[str] = []
        sources_used: list[str] = []
        status_flags: dict = {}
        name_resolved = False
        resolver = get_resolver()

        # ── G0: Gemini Entity Extraction (enrichment; non-blocking) ──
        entities: list[str] = []
        if self.gemini and GEMINI_ENTITY_ENABLED and not offline_first:
            entities = extract_entities_gemini(query, self.gemini)
            if entities:
                status_flags["gemini_ok"] = True
                logger.info(f"Gemini extracted entities: {entities}")
        elif self.gemini and GEMINI_ENTITY_ENABLED:
            try:
                entities = extract_entities_gemini(query, self.gemini)
                if entities:
                    status_flags["gemini_ok"] = True
            except Exception:
                entities = []

        if not entities:
            entities = offline_extract_entities(query)

        search_tokens = entities if entities else [query]

        for entity in search_tokens:
            # ── P0: Offline entity map (primary) ─────────
            resolved = resolver.resolve(entity)
            if resolved:
                expansions.extend(resolved.aliases)
                if resolved.devanagari:
                    expansions.append(resolved.devanagari)
                sources_used.append(f"entity_{resolved.source}")
                status_flags["offline_entity_used"] = True
                name_resolved = True
                cache_set(
                    self.conn,
                    entity,
                    resolved.aliases,
                    resolved.source,
                )
                continue

            # ── P1: SQLite Cache ───────────────────────
            cached = cache_get(self.conn, entity)
            if cached:
                logger.info(f"Cache hit: '{entity}' → {cached}")
                expansions.extend(cached)
                sources_used.append("sqlite_cache")
                name_resolved = True
                continue

            # ── P2: Wikidata (optional enrichment) ─────
            if WIKIDATA_ENABLED and not offline_first:
                wiki_aliases, resolved = resolve_wikidata(entity)
                if wiki_aliases:
                    expansions.extend(wiki_aliases)
                    sources_used.append("wikidata")
                    status_flags["wikidata_ok"] = True
                    cache_set(self.conn, entity, wiki_aliases, "wikidata")
                    name_resolved = True
                    continue

            # ── P3: indic-transliteration ──────────────
            translit = transliterate_entities([entity])
            if translit:
                expansions.extend(translit)
                status_flags["xlit_fallback_used"] = True
                if "indic_transliteration" not in sources_used:
                    sources_used.append("indic_transliteration")

        # ── P4 Gate + Monier-Williams (network; optional) ─
        if not name_resolved and is_epithet_query(query):
            mw_terms = monier_williams_lookup(query)
            if mw_terms:
                expansions.extend(mw_terms)
                sources_used.append("monier_williams")

        # ── Merge + Deduplicate ────────────────────────
        seen = set()
        unique_expansions = []
        for term in expansions:
            key = term.lower().strip()
            if key and key != query.lower() and key not in seen:
                seen.add(key)
                unique_expansions.append(term)

        augmented = query
        if unique_expansions:
            augmented = query + " " + " ".join(unique_expansions)

        return {
            "original"      : query,
            "augmented"     : augmented.strip(),
            "expansions"    : unique_expansions,
            "entities"      : entities,
            "name_resolved" : name_resolved,
            "sources_used"  : sources_used,
            "status_flags"  : status_flags,
        }

    def close(self):
        self.conn.close()
