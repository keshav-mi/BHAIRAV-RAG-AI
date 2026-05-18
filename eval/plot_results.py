"""
Plot Bhairav eval results from eval/results/.

Usage:
  python -m eval.plot_results
  python -m eval.plot_results --retrieval eval/results/retrieval_after_rerank_20260517_031123_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
OUT_DIR = RESULTS_DIR / "figures"


def latest_file(pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def plot_retrieval(report_path: Path, out_dir: Path) -> list[Path]:
    with open(report_path, encoding="utf-8") as f:
        r = json.load(f)

    overall = r["overall"]
    by_src = r.get("recall_at_5_by_source", {})
    saved = []

    # 1) Overall @K
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = [1, 5, 10]
    recalls = [overall.get(f"Recall@{k}", 0) for k in ks]
    ax.bar([f"@{k}" for k in ks], recalls, color=["#4a90d9", "#2ecc71", "#9b59b6"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title(f"Retrieval (Harrier) — {overall.get('num_queries', '?')} queries\n"
                 f"MRR={overall.get('MRR', 0):.3f}  stage={overall.get('stage', '')}")
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    p1 = out_dir / "retrieval_recall_at_k.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # 2) Recall@5 by source
    if by_src:
        fig, ax = plt.subplots(figsize=(9, 5))
        sources = sorted(by_src.keys(), key=lambda s: by_src[s])
        vals = [by_src[s] for s in sources]
        colors = ["#e74c3c" if v < 0.75 else "#2ecc71" for v in vals]
        ax.barh(sources, vals, color=colors)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Recall@5")
        ax.set_title("Recall@5 by source (after rerank)")
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        fig.tight_layout()
        p2 = out_dir / "retrieval_recall5_by_source.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        saved.append(p2)

    return saved


def plot_ragas(ragas_path: Path, out_dir: Path) -> list[Path]:
    with open(ragas_path, encoding="utf-8") as f:
        r = json.load(f)

    saved = []
    scores = r.get("scores", {})
    if not scores:
        return saved

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(scores.keys())
    vals = [float(scores[n]) for n in names]
    ax.bar(names, vals, color="#8e44ad")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"RAGAS — {r.get('num_queries', '?')} queries")
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    p = out_dir / "ragas_scores.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    per_q = r.get("per_query", [])
    faith = []
    for row in per_q:
        v = row.get("faithfulness")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            faith.append(float(v))
    if faith:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(faith, bins=min(15, max(5, len(faith) // 3)), color="#8e44ad", edgecolor="white")
        ax.set_xlabel("faithfulness")
        ax.set_ylabel("count")
        ax.set_title("Faithfulness per query")
        fig.tight_layout()
        p2 = out_dir / "ragas_faithfulness_hist.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        saved.append(p2)

    return saved


def main():
    p = argparse.ArgumentParser(description="Plot eval results")
    p.add_argument("--retrieval", type=Path, default=None)
    p.add_argument("--ragas", type=Path, default=None)
    args = p.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval = args.retrieval or latest_file("retrieval_after_rerank_*_report.json")
    ragas = args.ragas or latest_file("ragas_*.json")

    print("=" * 50)
    if retrieval and retrieval.exists():
        print(f"Retrieval: {retrieval.name}")
        paths = plot_retrieval(retrieval, out_dir)
        for path in paths:
            print(f"  saved {path}")
    else:
        print("No retrieval report found.")

    if ragas and ragas.exists():
        print(f"RAGAS: {ragas.name}")
        paths = plot_ragas(ragas, out_dir)
        for path in paths:
            print(f"  saved {path}")
    else:
        print("No RAGAS report yet — run bhairav_ragas_eval first.")

    print(f"\nFigures folder: {out_dir}")


if __name__ == "__main__":
    main()
