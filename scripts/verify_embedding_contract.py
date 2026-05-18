#!/usr/bin/env python3
"""
Verify Harrier query vs document encoding alignment.

Usage:
  python scripts/verify_embedding_contract.py
  python scripts/verify_embedding_contract.py --chunk-id <id_from_metadata>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, FAISS_PATH, ID_MAP_PATH, METADATA_PATH


def embed(model, texts, prefix: str | None) -> np.ndarray:
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Harrier embedding contract check")
    p.add_argument("--chunk-id", default=None, help="Metadata chunk id to probe")
    args = p.parse_args()

    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(ID_MAP_PATH, encoding="utf-8") as f:
        id_map = json.load(f)

    cid = args.chunk_id or next(iter(meta))
    chunk = meta[cid]
    summary = (chunk.get("hindi_summary") or "").strip()
    text = (chunk.get("text") or "").strip()
    doc_plain = f"{summary}\n{text}".strip()
    doc_meaning = f"Meaning: {doc_plain}"

    query = "Who was Ekalavya?"
    query_meaning = f"Meaning: {query}"

    print(f"Chunk id: {cid}")
    print(f"Source  : {chunk.get('source')}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 512

    variants = {
        "query_plain": query,
        "query_meaning_prefix": query_meaning,
        "doc_plain": doc_plain[:2000],
        "doc_meaning_prefix": doc_meaning[:2000],
    }

    vecs = {k: embed(model, [v], None)[0] for k, v in variants.items()}

    print("\nCosine similarity (higher = better alignment):")
    pairs = [
        ("query_plain", "doc_plain"),
        ("query_meaning_prefix", "doc_plain"),
        ("query_meaning_prefix", "doc_meaning_prefix"),
        ("query_plain", "doc_meaning_prefix"),
    ]
    for a, b in pairs:
        sim = float(np.dot(vecs[a], vecs[b]))
        print(f"  {a:22} vs {b:22} -> {sim:.4f}")

    idx = faiss.read_index(FAISS_PATH)
    pos = None
    for k, v in id_map.items():
        if v == cid and str(k).isdigit():
            pos = int(k)
            break
    if pos is not None:
        stored = idx.reconstruct(pos)
        for label, v in [
            ("stored_index", stored),
            ("query_meaning_prefix", vecs["query_meaning_prefix"]),
        ]:
            sim = float(np.dot(stored, v))
            print(f"  FAISS stored vs {label:22} -> {sim:.4f}")
    else:
        print("  (chunk id not found in id_map for FAISS reconstruct)")

    print(
        "\nIf query_meaning_prefix vs doc_plain is much lower than vs doc_meaning_prefix,"
        "\nre-embed the corpus or change retriever embed_query() to match build-time text."
    )


if __name__ == "__main__":
    main()
