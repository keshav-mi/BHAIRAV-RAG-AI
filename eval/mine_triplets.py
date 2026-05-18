"""
Mine synthetic (query, gold_chunk_id) triplets from hindi_summary via Groq.

Usage:
  python -m eval.build_manifest --per-source 500
  python -m eval.mine_triplets --questions-per-chunk 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = EVAL_DIR / "data" / "chunk_manifest.json"
DEFAULT_OUT = EVAL_DIR / "data" / "triplets.jsonl"

QUESTION_PROMPT = """You create search queries for a Dharmic text retrieval system.

Given this Hindi summary of one passage, write exactly {n} short questions a user might ask
that THIS passage alone would answer well. Mix Hindi and English if natural.

Rules:
- Questions must be answerable from this summary only
- No markdown, no numbering, no explanation
- Return ONLY a JSON array of strings, e.g. ["question one", "question two"]

Summary (source: {source}, id: {chunk_id}):
{summary}
"""


def parse_questions(raw: str, n: int) -> list[str]:
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        if not isinstance(arr, list):
            return []
        out = [str(q).strip() for q in arr if str(q).strip()]
        return out[:n]
    except json.JSONDecodeError:
        return []


def mine_triplets(
    manifest_path: Path,
    out_path: Path,
    questions_per_chunk: int = 2,
    max_chunks: int | None = None,
    seed: int = 42,
    delay_s: float = 0.3,
) -> int:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(manifest)
    if max_chunks:
        manifest = manifest[:max_chunks]

    client = Groq(api_key=GROQ_API_KEY)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(manifest, 1):
            cid = chunk["id"]
            summary = chunk["hindi_summary"]
            source = chunk.get("source", "")

            prompt = QUESTION_PROMPT.format(
                n=questions_per_chunk,
                source=source,
                chunk_id=cid,
                summary=summary,
            )

            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content or ""
                questions = parse_questions(raw, questions_per_chunk)
            except Exception as e:
                print(f"  [{i}] skip {cid}: {e}")
                questions = []

            for q in questions:
                row = {
                    "query": q,
                    "gold_chunk_id": cid,
                    "gold_chunk_ids": [cid],
                    "source": source,
                    "synthetic": True,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if i % 25 == 0:
                print(f"  processed {i}/{len(manifest)} chunks -> {written} triplets")

            if delay_s:
                time.sleep(delay_s)

    print(f"Done: {written} triplets -> {out_path}")
    return written


def main():
    p = argparse.ArgumentParser(description="Mine synthetic eval triplets")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--questions-per-chunk", type=int, default=2)
    p.add_argument("--max-chunks", type=int, default=2500, help="Cap chunks to process")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--delay", type=float, default=0.25)
    args = p.parse_args()

    if not args.manifest.exists():
        print("Manifest missing. Run: python -m eval.build_manifest")
        return

    mine_triplets(
        args.manifest,
        args.out,
        args.questions_per_chunk,
        args.max_chunks,
        args.seed,
        args.delay,
    )


if __name__ == "__main__":
    main()
