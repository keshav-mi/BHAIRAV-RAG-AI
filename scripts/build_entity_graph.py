#!/usr/bin/env python3
"""
Build entity_graph.json from entity_map_clean.json + epithet_map.json (Sprint 3.2).

Usage:
  python scripts/build_entity_graph.py
  python scripts/build_entity_graph.py --out data/entity_graph.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR


def build_graph(data_dir: Path) -> dict:
    entity_path = data_dir / "entity_map_clean.json"
    if not entity_path.exists():
        entity_path = data_dir / "entity_map_seed.json"
    epithet_path = data_dir / "epithet_map.json"

    graph: dict = {}

    if entity_path.exists():
        with open(entity_path, encoding="utf-8") as f:
            entities = json.load(f)
        for eng, payload in entities.items():
            deva = (payload.get("devanagari") or "").strip()
            if not deva:
                continue
            graph.setdefault(deva, {
                "canonical_en": eng,
                "aliases": [eng],
                "epithets": [],
                "relations": payload.get("relations", {}),
                "sources": payload.get("sources", []),
            })
            aliases = payload.get("aliases", [])
            for a in aliases:
                if a and a not in graph[deva]["aliases"]:
                    graph[deva]["aliases"].append(a)

    if epithet_path.exists():
        with open(epithet_path, encoding="utf-8") as f:
            epithets = json.load(f)
        for epithet, payload in epithets.items():
            canonical = (payload.get("canonical") or "").strip()
            if not canonical:
                continue
            if canonical not in graph:
                graph[canonical] = {
                    "canonical_en": "",
                    "aliases": [],
                    "epithets": [],
                    "relations": {},
                    "sources": [],
                }
            if epithet not in graph[canonical]["epithets"]:
                graph[canonical]["epithets"].append(epithet)
            if epithet not in graph[canonical]["aliases"]:
                graph[canonical]["aliases"].append(epithet)

    return graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    graph = build_graph(args.data_dir)
    out = args.out or (args.data_dir / "entity_graph.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(graph):,} canonical nodes -> {out}")


if __name__ == "__main__":
    main()
