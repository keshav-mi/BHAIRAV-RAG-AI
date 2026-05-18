"""
Pure retrieval evaluation: MRR, Recall@K, Precision@K, nDCG@K.

Uses labeled triplets (JSONL) with real chunk IDs from your index.

Usage:
  python -m eval.bhairav_retrieval_eval --triplets eval/data/triplets.jsonl
  python -m eval.bhairav_retrieval_eval --triplets eval/data/triplets.jsonl --stage after_rerank
  python -m eval.bhairav_retrieval_eval --triplets eval/data/triplets.jsonl --max-queries 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from eval.metrics import aggregate_metrics, score_one_query
from eval.pipeline import retrieve_ids, unload

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_TRIPLETS = EVAL_DIR / "data" / "triplets.jsonl"
DEFAULT_OUT = EVAL_DIR / "results"


def load_triplets(path: Path, max_queries: int | None) -> list[dict]:
    rows = []
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("triplets", [])
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if max_queries:
        rows = rows[:max_queries]
    return rows


def gold_ids(row: dict) -> list[str]:
    if "gold_chunk_ids" in row and row["gold_chunk_ids"]:
        return list(row["gold_chunk_ids"])
    return [row["gold_chunk_id"]]


def run_eval(
    triplets_path: Path,
    stage: str,
    k_values: list[int],
    top_k_retrieve: int,
    rerank_top_n: int,
    max_queries: int | None,
    out_dir: Path,
) -> dict:
    triplets = load_triplets(triplets_path, max_queries)
    if not triplets:
        raise ValueError(f"No triplets in {triplets_path}")

    per_query = []
    by_source_rows: dict[str, list] = defaultdict(list)

    if stage == "after_neighbors":
        print(
            "NOTE: after_neighbors is diagnostic only — use after_rerank for regression gates."
        )
    print(f"Evaluating {len(triplets)} queries | stage={stage} | K={k_values}")

    for i, row in enumerate(triplets, 1):
        q = row["query"]
        gold = gold_ids(row)
        source = row.get("source", "unknown")

        try:
            retrieved, trace = retrieve_ids(
                q,
                stage=stage,
                top_k=top_k_retrieve,
                rerank_top_n=rerank_top_n,
            )
        except Exception as e:
            print(f"  [{i}] ERROR: {q[:60]!r} -> {e}")
            retrieved = []
            trace = {}

        scores = score_one_query(retrieved, gold, k_values)
        scores["query"] = q
        scores["gold"] = gold
        scores["source"] = source
        scores["hit@5"] = scores.get("recall@5", 0) >= 1.0
        scores["rerank_used"] = trace.get("rerank_used", False)
        scores["neighbors_used"] = trace.get("neighbors_used", False)
        scores["confidence_band"] = trace.get("confidence_band")
        scores["faiss_top1_score"] = trace.get("faiss_top1_score")
        scores["intent"] = trace.get("intent")
        scores["latency_retrieve_ms"] = trace.get("latency_retrieve_ms")
        scores["latency_rerank_ms"] = trace.get("latency_rerank_ms")
        flags = trace.get("status_flags") or {}
        scores["gemini_ok"] = flags.get("gemini_ok", False)
        scores["wikidata_ok"] = flags.get("wikidata_ok", False)
        scores["offline_entity_used"] = flags.get("offline_entity_used", False)
        per_query.append(scores)
        by_source_rows[source].append(scores)

        if i % 50 == 0:
            mrr_so_far = sum(s["mrr"] for s in per_query) / len(per_query)
            r5 = sum(s["recall@5"] for s in per_query) / len(per_query)
            print(f"  [{i}/{len(triplets)}] running MRR={mrr_so_far:.3f} Recall@5={r5:.3f}")

    overall = aggregate_metrics(per_query, k_values)
    overall["stage"] = stage
    overall["triplets_file"] = str(triplets_path)

    by_source = {}
    for src, rows in by_source_rows.items():
        by_source[src] = aggregate_metrics(rows, k_values)

    # Recall@5 by source (primary health metric)
    recall5_by_source = {
        src: agg.get("Recall@5", 0) for src, agg in by_source.items()
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "overall": overall,
        "recall_at_5_by_source": recall5_by_source,
        "by_source": by_source,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"retrieval_{stage}_{ts}"

    with open(out_dir / f"{tag}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / f"{tag}_per_query.jsonl", "w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("RETRIEVAL EVAL SUMMARY")
    print("=" * 60)
    print(f"Queries : {overall['num_queries']}")
    print(f"MRR     : {overall['MRR']:.4f}")
    for k in k_values:
        print(f"Recall@{k:<2}: {overall[f'Recall@{k}']:.4f}")
        print(f"nDCG@{k:<4}: {overall[f'nDCG@{k}']:.4f}")
    print("\nRecall@5 by source:")
    for src, v in sorted(recall5_by_source.items(), key=lambda x: -x[1]):
        print(f"  {src:<22} {v:.4f}")
    print(f"\nSaved -> {out_dir / f'{tag}_report.json'}")

    unload()
    return report


def main():
    p = argparse.ArgumentParser(description="Bhairav retrieval-only eval")
    p.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    p.add_argument(
        "--stage",
        choices=["after_retrieve", "after_rerank", "after_neighbors"],
        default="after_rerank",
    )
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--top-k-retrieve", type=int, default=40)
    p.add_argument("--rerank-top-n", type=int, default=15)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.triplets.exists():
        print(f"Missing {args.triplets}")
        print("Run: python -m eval.build_manifest && python -m eval.mine_triplets")
        return

    run_eval(
        args.triplets,
        args.stage,
        args.k,
        args.top_k_retrieve,
        args.rerank_top_n,
        args.max_queries,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
