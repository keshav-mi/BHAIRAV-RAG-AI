#!/usr/bin/env python3
"""
Mine top BM25 tokens per source from bhairav_metadata.json → augment domain_signals.json.

Usage:
  python scripts/build_domain_signals.py --dry-run
  python scripts/build_domain_signals.py --write --top-n 25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, METADATA_PATH


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[\u0900-\u097F]+|[a-zA-Z]{3,}", text.lower())
    return [w for w in words if len(w) >= 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--out", type=Path, default=DATA_DIR / "domain_signals.json")
    args = p.parse_args()
    if args.write:
        args.dry_run = False

    meta_path = Path(METADATA_PATH)
    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        sys.exit(1)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    existing = {}
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            existing = json.load(f)

    by_source: dict[str, Counter] = {}
    for m in meta.values():
        src = m.get("source", "unknown")
        text = (m.get("hindi_summary") or "") + " " + (m.get("text") or "")
        by_source.setdefault(src, Counter()).update(tokenize(text))

    merged = dict(existing)
    for src, ctr in sorted(by_source.items()):
        top = [w for w, _ in ctr.most_common(args.top_n)]
        old = set(merged.get(src, []))
        merged[src] = list(dict.fromkeys(list(old) + top))[: max(len(old), args.top_n + 10)]

    print(f"Sources: {len(merged)}")
    for src in sorted(merged.keys())[:8]:
        print(f"  {src}: {len(merged[src])} signals")

    if args.dry_run:
        print("Dry run — pass --write to save")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
