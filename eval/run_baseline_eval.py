"""
Run full baseline eval before any embedding model change.

Steps:
  1. (optional) Build manifest + mine triplets
  2. Retrieval metrics at after_retrieve and after_rerank
  3. RAGAS on a small query set

Usage:
  python -m eval.run_baseline_eval
  python -m eval.run_baseline_eval --skip-mine --skip-ragas
  python -m eval.run_baseline_eval --mine-max-chunks 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TRIPLETS = EVAL_DIR / "data" / "triplets.jsonl"
MANIFEST = EVAL_DIR / "data" / "chunk_manifest.json"


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser(description="Run Bhairav baseline eval suite")
    p.add_argument("--skip-mine", action="store_true")
    p.add_argument("--skip-ragas", action="store_true")
    p.add_argument("--mine-max-chunks", type=int, default=2500)
    p.add_argument("--retrieval-max", type=int, default=500)
    p.add_argument("--ragas-limit", type=int, default=12)
    p.add_argument("--per-source", type=int, default=500)
    p.add_argument(
        "--overnight",
        action="store_true",
        help="manifest(75/src) + mine 500 + retrieve 220, skip RAGAS",
    )
    args = p.parse_args()

    if args.overnight:
        args.skip_ragas = True
        args.mine_max_chunks = 500
        args.retrieval_max = 220
        args.per_source = 75

    py = sys.executable

    if not args.skip_mine:
        if not MANIFEST.exists():
            code = run(
                [py, "-m", "eval.build_manifest", "--per-source", str(args.per_source)]
            )
            if code != 0:
                return code

        if not TRIPLETS.exists():
            code = run(
                [
                    py,
                    "-m",
                    "eval.mine_triplets",
                    "--max-chunks",
                    str(args.mine_max_chunks),
                ]
            )
            if code != 0:
                return code

    if not TRIPLETS.exists():
        print(f"Missing {TRIPLETS}. Run mining first or pass --skip-mine after creating triplets.")
        return 1

    for stage in ("after_retrieve", "after_rerank"):
        code = run(
            [
                py,
                "-m",
                "eval.bhairav_retrieval_eval",
                "--triplets",
                str(TRIPLETS),
                "--stage",
                stage,
                "--max-queries",
                str(args.retrieval_max),
            ]
        )
        if code != 0:
            return code

    if not args.skip_ragas:
        code = run(
            [
                py,
                "-m",
                "eval.bhairav_ragas_eval",
                "--direct",
                "--limit",
                str(args.ragas_limit),
            ]
        )
        if code != 0:
            return code

    print("\nBaseline eval complete. Compare reports in eval/results/")
    print("Save harrier report JSON before any embedding swap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
