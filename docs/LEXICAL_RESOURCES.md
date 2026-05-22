# Lexical & Sanskrit resources

How external projects relate to Bhairav AI and what is integrated today.

## Integrated today

| Resource | URL | Bhairav usage |
|----------|-----|----------------|
| **Cologne Sanskrit Lexicons** | [sanskrit-lexicon.uni-koeln.de](https://www.sanskrit-lexicon.uni-koeln.de/) | Offline `indexes/mw_index.json` via `build_mw_index.py` / `scripts/setup_cologne_mw.py`. Powers **non-entity** word grounding in `mw_grounding.py`. Live Cologne API is **disabled by default** (`MW_NETWORK_ENABLED=false`). |
| **indic-transliteration** | PyPI | HK/IAST/Vedanta schemes in `transliteration.py` and `query_normalizer.py`. |
| **AI4Bharat IndicXlit** | [github.com/AI4Bharat/IndicXlit](https://github.com/AI4Bharat/IndicXlit) | Optional local layer. **Not in core `requirements.txt`** — pulls `fairseq`, often fails on Windows. Use HK + entity maps + HTTP XLIT instead. |
| **Harvester maps** | `data/entity_map_clean.json`, `data/epithet_map.json` | Primary entity/epithet resolution in `entity_resolver.py` (replaces hardcoded Python dicts). |

## Documented / optional (not wired into runtime)

| Resource | URL | Why not full integration yet |
|----------|-----|------------------------------|
| **indic-dict** | [github.com/indic-dict](https://github.com/indic-dict) | Large STARDict/spelling repos; best used as **batch input** to grow `entity_map` or MW-style indexes, not per-query HTTP. |
| **vidyut** | [github.com/ambuda-org/vidyut](https://github.com/ambuda-org/vidyut) | Rust linguistic engine (sandhi, morphology). High value for **future** query normalization; requires a separate service or bindings. |
| **scl** | [github.com/samsaadhanii/scl](https://github.com/samsaadhanii/scl) | Research SCL toolkit; evaluate for batch epithet harvesting, not hot-path API. |

## Recommended workflow

1. **Monier-Williams (Cologne)**  
   Download XML → `python scripts/setup_cologne_mw.py --xml path/to/mw.xml`

2. **Entity long-tail**  
   `python scripts/expand_entity_map_from_eval.py --apply` after retrieval eval.

3. **Domain RRF boosts**  
   `python scripts/build_domain_signals.py --write` after metadata changes.

4. **Optional indic-dict**  
   Clone a dictionary repo, extract headwords → merge into `entity_map_clean.json` with a one-off script (same pattern as eval expander).

## Future: vidyut

For queries with heavy sandhi (classical Sanskrit), a sidecar **vidyut** segmenter could run at index time or query time. Track as Sprint 5+; keep retrieval offline-first until latency is bounded.
