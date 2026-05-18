#!/usr/bin/env python3
"""
Truncate pathological hindi_summary lengths in bhairav_metadata.json.

Usage:
  python scripts/sanitize_metadata.py --dry-run
  python scripts/sanitize_metadata.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MAX_SUMMARY_WORDS, METADATA_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="Write sanitized file in place")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--max-words", type=int, default=MAX_SUMMARY_WORDS)
    args = p.parse_args()
    if args.write:
        args.dry_run = False

    path = Path(METADATA_PATH)
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    fixed = 0
    worst_before = 0
    for cid, m in meta.items():
        summary = m.get("hindi_summary") or ""
        n = len(summary.split())
        worst_before = max(worst_before, n)
        if n > args.max_words:
            fixed += 1
            if not args.dry_run:
                words = summary.split()[: args.max_words]
                m["hindi_summary"] = " ".join(words) + " …[truncated]"

    print(f"Metadata chunks: {len(meta)}")
    print(f"Max summary words before: {worst_before}")
    print(f"Chunks over {args.max_words} words: {fixed}")
    if args.dry_run:
        print("Dry run — pass --write to apply")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
