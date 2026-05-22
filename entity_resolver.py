"""
Unified offline entity resolver (Sprint 3).
Loads entity_map_clean.json + epithet_map.json; exact → normalized → fuzzy.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rapidfuzz import fuzz, process

from config import DATA_DIR, ENTITY_FUZZY_THRESHOLD, REQUIRE_HARVESTED_MAPS, SYNONYM_FUZZY_ENABLED

FUZZY_THRESHOLD = ENTITY_FUZZY_THRESHOLD


@dataclass
class EntityResolution:
    canonical_id: str
    devanagari: str
    aliases: List[str]
    confidence: float
    source: str  # exact | normalized | fuzzy_en | fuzzy_deva | epithet | miss


class EntityResolver:
    """Single source of truth for entity / epithet lookup."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or DATA_DIR
        self._entity_map: Dict[str, dict] = {}
        self._english_keys: List[str] = []
        self._norm_english_keys: Dict[str, str] = {}  # precomputed: normalized_key → original_key
        self._deva_to_canonical: Dict[str, str] = {}
        self._epithet_to_canonical: Dict[str, str] = {}
        self._synonym_lookup: Dict[str, List[str]] = {}
        self._devanagari_synonym_keys: List[str] = []
        self._load()

    def _load(self) -> None:
        entity_path = self.data_dir / "entity_map_clean.json"
        if not entity_path.exists():
            entity_path = self.data_dir / "entity_map_seed.json"

        epithet_path = self.data_dir / "epithet_map.json"
        entity_loaded = epithet_loaded = False

        if entity_path.exists():
            entity_loaded = True
            with open(entity_path, encoding="utf-8") as f:
                self._entity_map = json.load(f)
            self._english_keys = list(self._entity_map.keys())
            # Precompute normalised→original mapping once so fuzzy resolve is O(1) per query
            self._norm_english_keys = {self._normalize_key(k): k for k in self._english_keys}
            for eng, payload in self._entity_map.items():
                deva = (payload.get("devanagari") or "").strip()
                if deva:
                    self._deva_to_canonical[deva] = eng

        if epithet_path.exists():
            epithet_loaded = True
            with open(epithet_path, encoding="utf-8") as f:
                epithet_data = json.load(f)
            for epithet, payload in epithet_data.items():
                canonical = (payload.get("canonical") or "").strip()
                if canonical:
                    self._epithet_to_canonical[epithet] = canonical

        if REQUIRE_HARVESTED_MAPS and not (entity_loaded and epithet_loaded):
            raise FileNotFoundError(
                f"REQUIRE_HARVESTED_MAPS=true but missing maps under {self.data_dir}"
            )

        self._build_synonym_lookup()
        self._devanagari_synonym_keys = [
            k for k in self._synonym_lookup
            if any("\u0900" <= c <= "\u097F" for c in k)
        ]
        print(
            f"EntityResolver | entities={len(self._entity_map):,} "
            f"epithets={len(self._epithet_to_canonical):,} "
            f"synonym_keys={len(self._synonym_lookup):,}"
        )

    def _link_synonym(self, a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        for key, other in ((a, b), (b, a)):
            if key not in self._synonym_lookup:
                self._synonym_lookup[key] = []
            if other not in self._synonym_lookup[key]:
                self._synonym_lookup[key].append(other)

    def _build_synonym_lookup(self) -> None:
        """Bidirectional epithet ↔ canonical links from JSON maps only."""
        for eng, payload in self._entity_map.items():
            deva = (payload.get("devanagari") or "").strip()
            if deva:
                self._link_synonym(deva, eng)
            for alias in payload.get("aliases", []):
                if alias:
                    self._link_synonym(deva or eng, alias)

        for epithet, canonical in self._epithet_to_canonical.items():
            self._link_synonym(epithet, canonical)

    def expand_synonyms(self, tokens: List[str], max_per_token: int = 3) -> List[str]:
        """FAISS-only epithet expansion — exact keys only unless SYNONYM_FUZZY_ENABLED."""
        extras: List[str] = []
        if not self._synonym_lookup:
            return extras

        for token in tokens:
            if token in self._synonym_lookup:
                extras.extend(self._synonym_lookup[token][:max_per_token])
                continue
            if not SYNONYM_FUZZY_ENABLED:
                continue
            # Fuzzy only on Devanagari keys (smaller set than full epithet list)
            if not self._devanagari_synonym_keys:
                continue
            match = process.extractOne(
                token,
                self._devanagari_synonym_keys,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if match:
                extras.extend(self._synonym_lookup[match[0]][:max_per_token])

        return list(dict.fromkeys(extras))

    @staticmethod
    def _normalize_key(text: str) -> str:
        folded = unicodedata.normalize("NFKD", text.strip().lower())
        return "".join(c for c in folded if not unicodedata.combining(c))

    def _aliases_for(self, eng_key: str, payload: dict) -> List[str]:
        deva = payload.get("devanagari", "")
        aliases = [eng_key, deva]
        for a in payload.get("aliases", []):
            if a:
                aliases.append(a)
        return list(dict.fromkeys(a for a in aliases if a))

    def resolve(self, token: str) -> Optional[EntityResolution]:
        if not token or len(token.strip()) < 2:
            return None

        raw = token.strip()
        norm = self._normalize_key(raw)

        # 1. Exact English key
        for key in (raw, raw.title(), norm):
            if key in self._entity_map:
                p = self._entity_map[key]
                return EntityResolution(
                    canonical_id=key,
                    devanagari=p.get("devanagari", ""),
                    aliases=self._aliases_for(key, p),
                    confidence=float(p.get("confidence", 1.0)),
                    source="exact",
                )

        # 2. Epithet → canonical devanagari
        if raw in self._epithet_to_canonical:
            can = self._epithet_to_canonical[raw]
            return EntityResolution(
                canonical_id=can,
                devanagari=can,
                aliases=[raw, can],
                confidence=0.95,
                source="epithet",
            )

        # 3. Devanagari exact
        if raw in self._deva_to_canonical:
            eng = self._deva_to_canonical[raw]
            p = self._entity_map.get(eng, {})
            return EntityResolution(
                canonical_id=eng,
                devanagari=raw,
                aliases=self._aliases_for(eng, p),
                confidence=0.98,
                source="exact",
            )

        # 4. Unicode-normalized English keys
        for eng_key in self._english_keys:
            if self._normalize_key(eng_key) == norm:
                p = self._entity_map[eng_key]
                return EntityResolution(
                    canonical_id=eng_key,
                    devanagari=p.get("devanagari", ""),
                    aliases=self._aliases_for(eng_key, p),
                    confidence=0.92,
                    source="normalized",
                )

        # 5. Fuzzy English — uses precomputed _norm_english_keys (O(1), not rebuilt per call)
        if self._norm_english_keys:
            match = process.extractOne(
                norm,
                list(self._norm_english_keys.keys()),
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )
            if match:
                eng_key = self._norm_english_keys[match[0]]
                p = self._entity_map[eng_key]
                return EntityResolution(
                    canonical_id=eng_key,
                    devanagari=p.get("devanagari", ""),
                    aliases=self._aliases_for(eng_key, p),
                    confidence=match[1] / 100.0,
                    source="fuzzy_en",
                )

        # 6. Fuzzy Devanagari
        deva_keys = list(self._deva_to_canonical.keys())
        if deva_keys and any("\u0900" <= c <= "\u097F" for c in raw):
            match = process.extractOne(
                raw, deva_keys, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD
            )
            if match:
                eng = self._deva_to_canonical[match[0]]
                p = self._entity_map.get(eng, {})
                return EntityResolution(
                    canonical_id=eng,
                    devanagari=match[0],
                    aliases=self._aliases_for(eng, p),
                    confidence=match[1] / 100.0,
                    source="fuzzy_deva",
                )

        return None

    def resolve_query_tokens(self, query: str) -> List[EntityResolution]:
        """Resolve each word-like token in a query."""
        results: List[EntityResolution] = []
        seen: set = set()
        for word in re.findall(r"[\w']+", query):
            clean = word.strip("'")
            if len(clean) < 3:
                continue
            res = self.resolve(clean)
            if res and res.canonical_id not in seen:
                seen.add(res.canonical_id)
                results.append(res)
        return results


# Module singleton (loaded once at import)
_resolver: Optional[EntityResolver] = None


def get_resolver() -> EntityResolver:
    global _resolver
    if _resolver is None:
        _resolver = EntityResolver()
    return _resolver
