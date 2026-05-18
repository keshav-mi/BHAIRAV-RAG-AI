"""
RAGAS-only charts from eval/results/ragas_*.json

Run AFTER: python -m eval.bhairav_ragas_eval --limit 35

Usage:
  python -m eval.plot_ragas
  python -m eval.plot_ragas --ragas eval/results/ragas_20260517_120000.json
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
OUT_DIR = RESULTS_DIR / "figures" / "ragas"


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def latest_ragas() -> Path | None:
    files = sorted(RESULTS_DIR.glob("ragas_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_rows(report: dict) -> list[dict]:
    rows = []
    for i, row in enumerate(report.get("per_query", [])):
        q = row.get("user_input") or row.get("question") or f"query_{i}"
        ctx = row.get("retrieved_contexts") or row.get("contexts") or []
        scores = {}
        for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            v = _safe_float(row.get(key))
            if v is not None:
                scores[key] = v
        rows.append(
            {
                "question": q,
                "n_contexts": len(ctx) if isinstance(ctx, list) else 0,
                "answer_len": len(row.get("response") or row.get("answer") or ""),
                "scores": scores,
            }
        )
    return rows


def plot_summary_scores(report: dict, out_dir: Path) -> Path | None:
    scores = report.get("scores", {})
    clean = {k: _safe_float(v) for k, v in scores.items()}
    clean = {k: v for k, v in clean.items() if v is not None}
    if not clean:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(clean.keys())
    vals = [clean[n] for n in names]
    colors = ["#2ecc71" if v >= 0.7 else "#f39c12" if v >= 0.5 else "#e74c3c" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="0.7 guide")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"RAGAS summary — {report.get('num_queries', '?')} queries")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "01_summary_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_query_faithfulness(rows: list[dict], out_dir: Path) -> Path | None:
    items = [(r["question"], r["scores"]["faithfulness"]) for r in rows if "faithfulness" in r["scores"]]
    if not items:
        return None

    items.sort(key=lambda x: x[1])
    labels = [textwrap.shorten(q, width=42) for q, _ in items]
    vals = [v for _, v in items]

    h = max(5, len(items) * 0.28)
    fig, ax = plt.subplots(figsize=(10, h))
    colors = ["#2ecc71" if v >= 0.7 else "#f39c12" if v >= 0.5 else "#e74c3c" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("faithfulness")
    ax.set_title("Faithfulness per question")
    ax.axvline(0.7, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    path = out_dir / "02_faithfulness_per_question.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_faithfulness_histogram(rows: list[dict], out_dir: Path) -> Path | None:
    faith = [r["scores"]["faithfulness"] for r in rows if "faithfulness" in r["scores"]]
    if not faith:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = min(12, max(5, len(faith) // 4))
    ax.hist(faith, bins=bins, color="#8e44ad", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(faith), color="#e74c3c", linestyle="--", label=f"mean={np.mean(faith):.2f}")
    ax.set_xlabel("faithfulness")
    ax.set_ylabel("count")
    ax.set_title("Faithfulness distribution")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "03_faithfulness_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_contexts_vs_faithfulness(rows: list[dict], out_dir: Path) -> Path | None:
    xs, ys = [], []
    for r in rows:
        if "faithfulness" in r["scores"]:
            xs.append(r["n_contexts"])
            ys.append(r["scores"]["faithfulness"])
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, alpha=0.7, s=60, c="#3498db", edgecolors="white")
    ax.set_xlabel("Number of retrieved contexts")
    ax.set_ylabel("faithfulness")
    ax.set_title("Contexts vs faithfulness")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    path = out_dir / "04_contexts_vs_faithfulness.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dashboard(report: dict, rows: list[dict], out_dir: Path) -> Path | None:
    scores = report.get("scores", {})
    faith_mean = _safe_float(scores.get("faithfulness"))
    if faith_mean is None and rows:
        vals = [r["scores"]["faithfulness"] for r in rows if "faithfulness" in r["scores"]]
        faith_mean = float(np.mean(vals)) if vals else None

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Summary text
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    lines = [
        "Bhairav RAGAS Report",
        f"Queries: {report.get('num_queries', len(rows))}",
        f"Timestamp: {report.get('timestamp', 'n/a')[:19]}",
        "",
    ]
    for k, v in scores.items():
        fv = _safe_float(v)
        if fv is not None:
            lines.append(f"{k}: {fv:.3f}")
    if faith_mean is not None:
        lines.append("")
        lines.append(f"Mean faithfulness: {faith_mean:.3f}")
    ax0.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")

    # Mini bar summary
    ax1 = fig.add_subplot(gs[0, 1])
    clean = {k: _safe_float(v) for k, v in scores.items()}
    clean = {k: v for k, v in clean.items() if v is not None}
    if clean:
        ax1.bar(list(clean.keys()), list(clean.values()), color="#8e44ad")
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Metric scores")
        plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")
    else:
        ax1.text(0.5, 0.5, "No aggregate scores", ha="center")

    # Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    faith = [r["scores"]["faithfulness"] for r in rows if "faithfulness" in r["scores"]]
    if faith:
        ax2.hist(faith, bins=min(10, max(4, len(faith) // 3)), color="#8e44ad", edgecolor="white")
        ax2.set_xlabel("faithfulness")
        ax2.set_ylabel("count")
        ax2.set_title("Distribution")
    else:
        ax2.text(0.5, 0.5, "No per-query faithfulness", ha="center")

    # Context scatter
    ax3 = fig.add_subplot(gs[1, 1])
    xs = [r["n_contexts"] for r in rows if "faithfulness" in r["scores"]]
    ys = [r["scores"]["faithfulness"] for r in rows if "faithfulness" in r["scores"]]
    if xs:
        ax3.scatter(xs, ys, c="#3498db", alpha=0.75)
        ax3.set_xlabel("# contexts")
        ax3.set_ylabel("faithfulness")
        ax3.set_title("Contexts vs score")
    else:
        ax3.text(0.5, 0.5, "No data", ha="center")

    fig.suptitle("RAGAS evaluation dashboard", fontsize=13, fontweight="bold")
    path = out_dir / "00_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Plot RAGAS eval charts only")
    p.add_argument("--ragas", type=Path, default=None)
    args = p.parse_args()

    ragas_path = args.ragas or latest_ragas()
    if not ragas_path or not ragas_path.exists():
        print("No ragas_*.json found. Run first:")
        print("  python -m eval.bhairav_ragas_eval --limit 35")
        return

    with open(ragas_path, encoding="utf-8") as f:
        report = json.load(f)
    rows = load_rows(report)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for plot_fn, arg in (
        (plot_dashboard, (report, rows, out_dir)),
        (plot_summary_scores, (report, out_dir)),
        (plot_per_query_faithfulness, (rows, out_dir)),
        (plot_faithfulness_histogram, (rows, out_dir)),
        (plot_contexts_vs_faithfulness, (rows, out_dir)),
    ):
        path = plot_fn(*arg)
        if path:
            saved.append(path)
            print(f"  saved {path.name}")

    print(f"\nRAGAS file: {ragas_path.name}")
    print(f"Figures: {out_dir}")
    print(f"Open: {out_dir / '00_dashboard.png'}")


if __name__ == "__main__":
    main()
