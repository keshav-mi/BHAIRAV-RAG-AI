"""
Build a lightweight chunk manifest from bhairav_metadata.json for triplet mining.
Avoids holding full Sanskrit text in memory during mining.

Usage:
  python -m eval.build_manifest
  python -m eval.build_manifest --per-source 500
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from config import METADATA_PATH

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EVAL_DIR / "data" / "chunk_manifest.json"


def build_manifest(
    metadata_path: str,
    out_path: Path,
    per_source: int | None = None,
    seed: int = 42,
) -> int:
    print(f"Loading metadata from {metadata_path} ...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    by_source: dict[str, list] = defaultdict(list)
    for cid, m in meta.items():
        summary = (m.get("hindi_summary") or "").strip()
        if len(summary) < 40:
            continue
        by_source[m.get("source", "unknown")].append(
            {
                "id": cid,
                "source": m.get("source", ""),
                "book": m.get("book", ""),
                "chapter": m.get("chapter"),
                "verse": m.get("verse"),
                "tier": m.get("tier", 2),
                "hindi_summary": summary[:800],
            }
        )

    rng = random.Random(seed)
    rows = []
    for source, items in sorted(by_source.items()):
        if per_source and len(items) > per_source:
            items = rng.sample(items, per_source)
        rows.extend(items)
        print(f"  {source}: {len(items)} chunks")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)

    print(f"Wrote {len(rows)} entries -> {out_path}")
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Build eval chunk manifest")
    p.add_argument("--metadata", default=METADATA_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--per-source",
        type=int,
        default=500,
        help="Max chunks per source (None = all)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_manifest(args.metadata, args.out, args.per_source, args.seed)


if __name__ == "__main__":
    main()
