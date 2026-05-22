"""
mw_grounding.py — Bhairav AI · Normalization Sublayer (Steps 1b → 1f)
======================================================================

PIPELINE (matches architecture diagram):

    Step 1b  Token classification
             Split query into ENTITY tokens (handled upstream by QueryNormalizer)
             and NON-ENTITY tokens — this module handles the non-entity path.

    Step 1c  Corpus-driven fuzzy correction
             mratyu → mrityu
             Uses the BM25 vocab (already in memory) as the correction corpus.
             No hardcoded spelling maps — auto-adapts to your dataset.

    Step 1d  Script normalization
             mrityu → मृत्यु
             Uses indic-transliteration (your existing dependency).

    Step 1e  MW semantic grounding
             मृत्यु → ["death", "dying", "destruction"]
             Looks up the pre-built mw_index.json (depth-1 only, cap 5).

    Step 1f  Controlled merge
             Deduplicates and caps total expansion at MAX_EXPANSION_TOKENS.
             Returns a single augmented string — drop-in for retriever.

INTEGRATION:
    In retriever.py Retriever.__init__():
        from mw_grounding import NonEntityGrounder
        self.mw_grounder = NonEntityGrounder(bm25=self.bm25, mw_index_path=MW_INDEX_PATH)

    In retriever.py Retriever.retrieve():
        mw_result = self.mw_grounder.ground(query, norm_result["entities"])
        combined_faiss = f"{mw_result['augmented']} {norm_result['augmented']} {exp_faiss}"

REQUIREMENTS (all already in your env):
    rapidfuzz
    indic-transliteration

PREREQUISITES:
    Run build_mw_index.py once to produce indexes/mw_index.json
    Add MW_INDEX_PATH = INDEX_DIR / "mw_index.json" to config.py
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from config import MW_FUZZY_ENABLED

logger = logging.getLogger(__name__)

# ── Tuning constants ────────────────────────────────────────
FUZZY_CUTOFF         = 82   # 80-85 sweet spot; below 80 → false corrections
MIN_CORPUS_FREQ      = 5    # only correct toward tokens seen >= N times in BM25
MIN_TOKEN_LEN        = 4    # skip very short tokens (ka, ki, ko etc.)
MAX_MW_MEANINGS      = 3    # depth-1 cap per token
MAX_EXPANSION_TOKENS = 8    # total tokens appended to original query

def _stop_words() -> set:
    from bhairav_data import get_stopwords
    return set(get_stopwords())

_MW_STOP = {"of", "the", "a", "an", "or", "and", "by", "with", "from", "to", "in"}


# ── Script helpers ──────────────────────────────────────────

def _is_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097F" for c in text)


def _ascii_fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )


# ── Step 1c: Corpus vocab ───────────────────────────────────

class CorpusVocab:
    """
    Builds a frequency-filtered vocab list from the BM25 index already
    in memory. Tokens with corpus frequency < MIN_CORPUS_FREQ are excluded
    so we never correct toward rare / noisy terms.

    BM25Okapi IDF: idf = log((N - df + 0.5) / (df + 0.5))
    Low IDF  = token appears in many docs = high frequency.
    We keep tokens where df >= MIN_CORPUS_FREQ.
    """

    def __init__(self, bm25):
        self._vocab: list[str] = []
        self._build(bm25)

    def _build(self, bm25):
        if bm25 is None:
            logger.warning("BM25 not available — fuzzy correction disabled")
            return
        try:
            import math
            N = bm25.corpus_size
            # idf threshold where df == MIN_CORPUS_FREQ
            idf_threshold = math.log((N - MIN_CORPUS_FREQ + 0.5) / (MIN_CORPUS_FREQ + 0.5))
            self._vocab = [
                tok for tok, idf in bm25.idf.items()
                if idf <= idf_threshold and len(tok) >= MIN_TOKEN_LEN
            ]
            logger.info(f"MW grounder: corpus vocab = {len(self._vocab):,} tokens")
        except Exception as e:
            # Graceful fallback: use full vocab
            self._vocab = [t for t in bm25.idf.keys() if len(t) >= MIN_TOKEN_LEN]
            logger.warning(f"Freq filter skipped ({e}); full vocab = {len(self._vocab):,}")

    @property
    def vocab(self) -> list[str]:
        return self._vocab


def fuzzy_correct(token: str, vocab: list[str]) -> Optional[str]:
    """
    Step 1c: Correct one non-entity token against corpus vocab.
    Returns corrected form if score >= FUZZY_CUTOFF, else None.
    """
    if not vocab or len(token) < MIN_TOKEN_LEN:
        return None

    token_folded = _ascii_fold(token)

    result = process.extractOne(
        token_folded,
        vocab,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=FUZZY_CUTOFF,
    )

    if result is None:
        return None

    best_match, score, _ = result

    # No-op guard: don't return identical token
    if _ascii_fold(best_match) == token_folded:
        return None

    logger.debug(f"Fuzzy: '{token}' → '{best_match}' (score={score})")
    return best_match


# ── Step 1d: Script normalization ───────────────────────────

def to_devanagari(token: str) -> Optional[str]:
    """
    Step 1d: Romanized → Devanagari via indic-transliteration.
    Tries ITRANS, HK, VELTHUIS, SLP1 in order.
    """
    if _is_devanagari(token):
        return token

    for scheme in [sanscript.ITRANS, sanscript.HK, sanscript.VELTHUIS, sanscript.SLP1]:
        try:
            deva = transliterate(token, scheme, sanscript.DEVANAGARI)
            if deva and _is_devanagari(deva):
                return deva
        except Exception:
            continue

    return None


# ── Step 1e: MW Index ────────────────────────────────────────

class MWIndex:
    """
    Wraps mw_index.json (built by build_mw_index.py) for O(1) lookup.

    Lookup order:
        1. Devanagari key (most precise)
        2. Romanized key as-is
        3. ASCII-folded key (strips diacritics: ā → a)
    """

    def __init__(self, index_path: Path):
        self._index: dict = {}
        self._load(index_path)

    def _load(self, path: Path):
        if not path.exists():
            logger.warning(
                f"MW index not found at {path}. "
                "Run build_mw_index.py first. MW grounding disabled."
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
            logger.info(f"MW index loaded: {len(self._index):,} headwords")
        except Exception as e:
            logger.error(f"Failed to load MW index: {e}")

    def lookup(self, devanagari: Optional[str], romanized: str) -> list[str]:
        """Step 1e: Return depth-1 English meanings. Empty list on miss."""
        if not self._index:
            return []

        if devanagari and devanagari in self._index:
            return self._index[devanagari]["meanings"][:MAX_MW_MEANINGS]

        key = romanized.lower()
        if key in self._index:
            return self._index[key]["meanings"][:MAX_MW_MEANINGS]

        folded = _ascii_fold(romanized)
        if folded in self._index:
            return self._index[folded]["meanings"][:MAX_MW_MEANINGS]

        return []

    @property
    def loaded(self) -> bool:
        return bool(self._index)


# ── Main grounder ────────────────────────────────────────────

class NonEntityGrounder:
    """
    Orchestrates Steps 1b → 1f for non-entity tokens.

    Instantiate once in Retriever.__init__():
        self.mw_grounder = NonEntityGrounder(
            bm25=self.bm25,
            mw_index_path=MW_INDEX_PATH,
        )

    Call per query in Retriever.retrieve():
        mw_result = self.mw_grounder.ground(query, norm_result["entities"])
        combined_faiss = f"{mw_result['augmented']} {norm_result['augmented']} {exp_faiss}"
    """

    def __init__(self, bm25, mw_index_path: Path):
        self._vocab = CorpusVocab(bm25)
        self._mw    = MWIndex(mw_index_path)

    # ── Step 1b ─────────────────────────────────────────────

    def _classify(self, query: str, entity_tokens: list[str]) -> list[str]:
        """
        Return NON-ENTITY tokens: query words that are not in entity_tokens
        and not stop words.
        """
        entity_set = {e.lower().strip() for e in entity_tokens}
        non_entity = []
        for word in query.split():
            clean = word.lower().strip(".,?!")
            if (
                clean not in entity_set
                and clean not in _stop_words()
                and len(clean) >= MIN_TOKEN_LEN
                and not _is_devanagari(clean)  # Devanagari already handled by QueryNormalizer
            ):
                non_entity.append(clean)
        return non_entity

    # ── Steps 1c → 1e per token ─────────────────────────────

    def _process_token(self, token: str) -> dict:
        corrected = (
            fuzzy_correct(token, self._vocab.vocab) if MW_FUZZY_ENABLED else None
        )
        working    = corrected if corrected else token
        devanagari = to_devanagari(working)
        meanings   = self._mw.lookup(devanagari, working)
        return {
            "original"  : token,
            "corrected" : corrected,
            "devanagari": devanagari,
            "meanings"  : meanings,
        }

    # ── Step 1f ─────────────────────────────────────────────

    def _merge(self, query: str, token_results: list[dict]) -> str:
        """
        Controlled merge with hard cap at MAX_EXPANSION_TOKENS.
        Priority: corrected → devanagari → MW meanings.
        """
        query_lower = {w.lower() for w in query.split()}
        added: list[str] = []
        seen: set[str]   = set(query_lower)

        def _add(term: str):
            if len(added) >= MAX_EXPANSION_TOKENS:
                return
            key = term.lower().strip()
            if key and key not in seen and len(key) >= 2:
                seen.add(key)
                added.append(term)

        # P1: corrected romanized forms
        for r in token_results:
            if r["corrected"]:
                _add(r["corrected"])

        # P2: Devanagari forms
        for r in token_results:
            if r["devanagari"]:
                _add(r["devanagari"])

        # P3: MW meanings — content words only, max 2 words per meaning
        for r in token_results:
            for meaning in r["meanings"]:
                content_words = [
                    w for w in meaning.split()
                    if w not in _MW_STOP and len(w) > 2
                ]
                for w in content_words[:2]:
                    _add(w)

        if not added:
            return query

        return query + " " + " ".join(added)

    # ── Public API ───────────────────────────────────────────

    def ground(self, query: str, entity_tokens: list[str]) -> dict:
        """
        Full Steps 1b → 1f.

        Args:
            query         : raw query string
            entity_tokens : what Gemini extracted (from QueryNormalizer.normalize())

        Returns:
            {
                "original"      : str,
                "augmented"     : str,         # enriched query string
                "token_results" : list[dict],  # per-token debug info
                "sources_used"  : list[str],   # which steps fired
            }
        """
        query = query.strip()

        # Step 1b
        non_entity_tokens = self._classify(query, entity_tokens)

        if not non_entity_tokens:
            return {
                "original"      : query,
                "augmented"     : query,
                "token_results" : [],
                "sources_used"  : [],
            }

        # Steps 1c → 1e
        token_results = [self._process_token(t) for t in non_entity_tokens]

        # Debug logging
        for r in token_results:
            if r["corrected"] or r["devanagari"] or r["meanings"]:
                logger.info(
                    f"MW ground: '{r['original']}' → "
                    f"corrected='{r['corrected']}' "
                    f"deva='{r['devanagari']}' "
                    f"meanings={r['meanings']}"
                )

        # Sources fired
        sources_used = []
        if any(r["corrected"] for r in token_results):
            sources_used.append("fuzzy_correction")
        if any(r["devanagari"] for r in token_results):
            sources_used.append("script_normalization")
        if any(r["meanings"] for r in token_results):
            sources_used.append("monier_williams")

        # Step 1f
        augmented = self._merge(query, token_results)

        return {
            "original"      : query,
            "augmented"     : augmented,
            "token_results" : token_results,
            "sources_used"  : sources_used,
        }
