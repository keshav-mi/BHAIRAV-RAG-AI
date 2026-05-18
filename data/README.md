# Harvested entity maps

Place your harvested JSON files here (or set `DATA_DIR` to another folder):

- `entity_map_clean.json` — English/Hinglish name → `{ "devanagari": "..." }`
- `epithet_map.json` — epithet → `{ "canonical": "..." }`

The repo ships `entity_map_seed.json` as a minimal fallback. For full eval parity, copy your 1,099-entry entity map and 46K epithet map into this directory and rename as above.

Set `REQUIRE_HARVESTED_MAPS=true` to fail fast at startup if either file is missing.
