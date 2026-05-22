#!/usr/bin/env python3
"""
Mine Recall@5 failures from retrieval eval and propose entity_map additions (Sprint 2.3).

Usage:
  python -m eval.bhairav_retrieval_eval --triplets eval/data/triplets.jsonl --max-queries 100
  python scripts/expand_entity_map_from_eval.py --eval-results eval/results/latest.json

  # Or run inline mini-eval:
  python scripts/expand_entity_map_from_eval.py --triplets eval/data/triplets.jsonl --max-queries 50 --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bhairav_data import get_stopwords
from config import DATA_DIR
from entity_resolver import EntityResolver

STOP = get_stopwords()


def extract_roman_tokens(query: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[\w']+", query):
        w = word.strip("'")
        if w.lower() in STOP or len(w) < 4:
            continue
        if any("\u0900" <= ch <= "\u097F" for ch in w):
            continue
        if w[0].isupper() or len(w) >= 4:
            tokens.append(w)
    return list(dict.fromkeys(tokens))


def devanagari_from_chunk(meta: dict, chunk_id: str) -> str | None:
    chunk = meta.get(chunk_id, {})
    summary = chunk.get("hindi_summary", "") or ""
    for word in summary.split()[:30]:
        if any("\u0900" <= c <= "\u097F" for c in word) and len(word) >= 2:
            return word.strip(".,;:")
    return None


def run_failure_analysis(
    triplets_path: Path,
    max_queries: int | None,
    stage: str = "after_rerank",
) -> list[dict]:
    from eval.pipeline import retrieve_ids

    rows = []
    with open(triplets_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if max_queries:
        rows = rows[:max_queries]

    failures = []
    for row in rows:
        q = row["query"]
        gold = row.get("gold_chunk_ids") or [row["gold_chunk_id"]]
        retrieved, _ = retrieve_ids(q, stage=stage)
        hit5 = any(g in retrieved[:5] for g in gold)
        if not hit5:
            failures.append({
                "query": q,
                "gold": gold,
                "source": row.get("source", ""),
                "retrieved_top5": retrieved[:5],
            })
    return failures


def propose_additions(
    failures: list[dict],
    meta: dict,
    resolver: EntityResolver,
) -> dict[str, dict]:
    proposals: dict[str, dict] = defaultdict(dict)

    for fail in failures:
        for token in extract_roman_tokens(fail["query"]):
            if resolver.resolve(token):
                continue
            deva = None
            for gid in fail["gold"]:
                deva = devanagari_from_chunk(meta, gid)
                if deva:
                    break
            if not deva:
                continue
            key = token.lower()
            proposals[key] = {
                "devanagari": deva,
                "type": "mined_from_eval",
                "confidence": 0.7,
                "source": f"eval_fail:{fail.get('source', '')}",
                "aliases": [token],
            }
    return dict(proposals)


def merge_into_map(map_path: Path, proposals: dict, dry_run: bool) -> int:
    existing = {}
    if map_path.exists():
        with open(map_path, encoding="utf-8") as f:
            existing = json.load(f)

    added = 0
    for key, payload in proposals.items():
        if key not in existing:
            existing[key] = payload
            added += 1
        else:
            old_aliases = set(existing[key].get("aliases", []))
            old_aliases.update(payload.get("aliases", []))
            existing[key]["aliases"] = list(old_aliases)

    if not dry_run and added:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    return added


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", type=Path, default=ROOT / "eval" / "data" / "triplets.jsonl")
    p.add_argument("--eval-results", type=Path, default=None)
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--out", type=Path, default=DATA_DIR / "entity_map_proposals.json")
    args = p.parse_args()

    from config import METADATA_PATH

    meta = {}
    meta_path = Path(METADATA_PATH)
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    if args.eval_results and args.eval_results.exists():
        with open(args.eval_results, encoding="utf-8") as f:
            data = json.load(f)
        failures = [r for r in data.get("per_query", []) if not r.get("hit@5")]
        failures = [
            {"query": f["query"], "gold": f.get("gold", []), "source": f.get("source", "")}
            for f in failures
        ]
    elif args.triplets.exists():
        print(f"Running mini-eval on {args.triplets} (max {args.max_queries})...")
        failures = run_failure_analysis(args.triplets, args.max_queries)
    else:
        print("No triplets or eval results found.")
        return

    print(f"Recall@5 failures: {len(failures)}")
    resolver = EntityResolver()
    proposals = propose_additions(failures, meta, resolver)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(proposals, f, ensure_ascii=False, indent=2)
    print(f"Proposed {len(proposals)} new entity keys -> {args.out}")

    if args.apply and proposals:
        map_path = DATA_DIR / "entity_map_clean.json"
        n = merge_into_map(map_path, proposals, dry_run=False)
        print(f"Merged {n} new entries into {map_path}")
    elif proposals:
        print("Pass --apply to merge into data/entity_map_clean.json")


if __name__ == "__main__":
    main()
