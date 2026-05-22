#!/usr/bin/env python3
"""One-shot retrieval timing breakdown (set PROFILE_RETRIEVAL=true in env)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PROFILE_RETRIEVAL", "true")
os.environ.setdefault("FAST_RETRIEVAL", "true")

QUERY = sys.argv[1] if len(sys.argv) > 1 else "arjun ne kurukshetra mein kya kiya"


def main():
    print(f"Query: {QUERY!r}")
    print(f"FAST_RETRIEVAL={os.getenv('FAST_RETRIEVAL')}")
    t0 = time.time()
    from retriever import Retriever
    from query_plan import build_query_plan

    print("Loading Retriever (cold start)...")
    r = Retriever()
    print(f"Load time: {time.time() - t0:.1f}s\n")

    plan = build_query_plan(QUERY)
    t1 = time.time()
    chunks, fq, bq, domains, meta = r.retrieve(QUERY, plan=plan)
    elapsed = time.time() - t1
    print(f"\nRetrieve: {elapsed:.2f}s | chunks={len(chunks)}")
    print(f"timing_ms: {meta.get('timing_ms')}")
    print(f"faiss_query: {fq[:80]}...")


if __name__ == "__main__":
    main()
