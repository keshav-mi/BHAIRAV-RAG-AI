# Data directory

Runtime configuration and lexical maps. **Prefer editing JSON here over hardcoding Python.
## Required for production quality

| File | Description |
|------|-------------|
| `entity_map_clean.json` | English/Hinglish name → `{ "devanagari": "...", "aliases": [], ... }` |
| `epithet_map.json` | Epithet → `{ "canonical": "..." }` (tens of thousands of entries) |

Fallback: `entity_map_seed.json` (minimal) if clean map is absent.

Set `REQUIRE_HARVESTED_MAPS=true` to fail startup if either main file is missing.

## Editable config (no code deploy)

| File | Used by |
|------|---------|
| `stopwords.json` | Transliteration, MW grounder, eval scripts |
| `domain_signals.json` | RRF source boosting in `retriever.py` |
| `chitchat.json` | Greeting / identity short-circuit |
| `intent_patterns.json` | `query_classifier.py` |
| `routing_policies.json` | `query_plan.py` (faiss_k, rerank, neighbors) |
| `query_variants.json` | Multi-query FAISS variants |

## Generated / optional

| File | How to create |
|------|----------------|
| `entity_graph.json` | `python scripts/build_entity_graph.py` |
| `entity_map_proposals.json` | `python scripts/expand_entity_map_from_eval.py` |

## Maintenance commands

```bash
# Grow entity map from eval failures
python scripts/expand_entity_map_from_eval.py --triplets eval/data/triplets.jsonl --max-queries 100 --apply

# Refresh domain signals from corpus vocabulary
python scripts/build_domain_signals.py --write

# Rebuild entity graph after map changes
python scripts/build_entity_graph.py
```

See root `README.md` and `docs/LEXICAL_RESOURCES.md`.
