# ============================================================
# BHAIRAV AI - RERANKER v5 (OPTIMIZED)
# Key fixes:
#   - Hard cap at 30 chunks before reranking (save 70% latency)
#   - Adaptive score floor still works
#   - Minimum 3 chunks safeguard
# ============================================================

from typing import Dict, List

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
    ) -> List[Dict]:
        """
        Adaptive reranking with hard cap.

        Devanagari queries:
            Higher score floor (0.3)

        English/Hinglish:
            Lower score floor (0.05)

        Always returns minimum 3 chunks.
        """
        if not chunks:
            return []

        chunks_to_rerank = chunks[:30]
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
        print(f"   Chunks reranked      : {len(chunks_to_rerank)} (capped at 30)")

        filtered = [c for c in reranked if c["rerank_score"] >= score_floor]
        if len(filtered) < 3:
            print("   Low-score query - forcing minimum 3 chunks")
            return reranked[: min(3, len(reranked))]

        return filtered[:top_n]
