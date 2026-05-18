# ============================================================
# BHAIRAV AI - RERANKER v6
# Adaptive gate via FAISS confidence (passed in retrieval_meta)
# ============================================================

from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANK_TOP_N


class Reranker:
    def __init__(self):
        print(f"Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_n: int = RERANK_TOP_N,
        script: str = "english",
        retrieval_meta: Optional[Dict] = None,
    ) -> List[Dict]:
        if not chunks:
            return []

        meta = retrieval_meta or {}
        if meta.get("skip_rerank"):
            print("   Reranker skipped (high FAISS confidence)")
            for i, c in enumerate(chunks[:top_n]):
                c["rerank_score"] = c.get("score", 1.0 - i * 0.01)
            return chunks[:top_n]

        cap = meta.get("rerank_top_n", top_n)
        cap = min(cap, 30, len(chunks))
        chunks_to_rerank = chunks[:cap]
        score_floor = 0.3 if script == "devanagari" else 0.05

        pairs = []
        for chunk in chunks_to_rerank:
            summary = chunk.get("hindi_summary", "") or ""
            text = chunk.get("text", "") or ""
            passage = f"{summary}\n{text}".strip()
            pairs.append((query, passage))

        scores = self.model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks_to_rerank, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(
            chunks_to_rerank,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        top5_scores = [round(c["rerank_score"], 3) for c in reranked[:5]]
        print(f"   Reranker top-5 scores: {top5_scores}")
        print(f"   Reranker floor       : {score_floor}")
        print(f"   Confidence band    : {meta.get('confidence_band', 'n/a')}")
        print(f"   Chunks reranked      : {len(chunks_to_rerank)} (cap {cap})")

        filtered = [c for c in reranked if c["rerank_score"] >= score_floor]
        if len(filtered) < 3:
            print("   Low-score query - forcing minimum 3 chunks")
            return reranked[: min(3, len(reranked))]

        return filtered[:top_n]
