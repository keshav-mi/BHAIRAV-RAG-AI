# ============================================================
# BHAIRAV AI - RETRIEVER v9
# FAISS confidence gate metadata, intent-aware top_k, neighbor append
# ============================================================

import json
import pickle
from typing import Dict, List, Tuple

import faiss
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from confidence import confidence_from_faiss
from config import (
    BM25_PATH,
    DOMAIN_BOOST,
    EMBEDDING_MODEL,
    FAISS_PATH,
    FAISS_TOP_K,
    GROQ_API_KEY,
    ID_MAP_PATH,
    METADATA_PATH,
    RRF_K,
    YAJURVEDA_DOMAIN_BOOST,
)
from mw_grounding import NonEntityGrounder
from query_expander import expand_query
from query_normalizer import QueryNormalizer
from query_plan import QueryPlan, apply_confidence_to_plan, build_query_plan, faiss_variants_for_plan

try:
    from config import MW_INDEX_PATH
except ImportError:
    from config import INDEX_DIR

    MW_INDEX_PATH = INDEX_DIR / "mw_index.json"


class Retriever:
    def __init__(self):
        print("Loading retriever components...")

        self.normalizer = QueryNormalizer()

        print(f"   Embedding model : {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embed_model.max_seq_length = 512

        print("   FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_PATH)
        print(f"   FAISS vectors   : {self.faiss_index.ntotal:,}")

        print("   BM25 index...")
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        print("   Metadata store...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("   ID map...")
        with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        self.chunk_ids = [self.id_map[k] for k in sorted(self.id_map.keys(), key=int)]
        self.rev_id_map = {v: int(k) for k, v in self.id_map.items() if k.isdigit()}

        self.groq_client = Groq(api_key=GROQ_API_KEY)

        print("   MW grounder...")
        self.mw_grounder = NonEntityGrounder(
            bm25=self.bm25,
            mw_index_path=MW_INDEX_PATH,
        )

        print(f"Retriever ready - {len(self.metadata):,} chunks")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def embed_queries_batch(self, queries: List[str]) -> np.ndarray:
        prefixed = [q.strip() for q in queries]
        vecs = self.embed_model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def faiss_search(self, vec, top_k):
        scores, positions = self.faiss_index.search(vec, top_k)
        return [
            (self.id_map[str(pos)], float(score))
            for score, pos in zip(scores[0], positions[0])
            if pos != -1
        ]

    def bm25_search(self, query, top_k):
        tokens = query.split()
        if not tokens:
            return []
        # Full-corpus scan — O(N); see config / plan for sparse-index future work
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in idxs if scores[i] > 0]

    def reciprocal_rank_fusion(self, faiss_res, bm25_res, domains, k=RRF_K):
        rrf = {}

        for rank, (cid, _) in enumerate(faiss_res):
            rrf[cid] = rrf.get(cid, 0) + 1 / (k + rank + 1)

        for rank, (cid, _) in enumerate(bm25_res):
            rrf[cid] = rrf.get(cid, 0) + 1 / (k + rank + 1)

        for cid in rrf:
            source = self.metadata.get(cid, {}).get("source", "")
            if domains and source in domains:
                rrf[cid] *= DOMAIN_BOOST
            if source == "Yajurveda":
                rrf[cid] *= YAJURVEDA_DOMAIN_BOOST

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    def get_neighbor_chunks(self, ranked_ids, window=1):
        """Legacy: index-order neighbors (used only if append helper not called)."""
        chunks, seen = [], set()

        for cid, score in ranked_ids:
            idx = self.rev_id_map.get(cid)
            if idx is None:
                continue

            for i in range(idx - window, idx + window + 1):
                nid = self.id_map.get(str(i))
                if nid and nid not in seen and nid in self.metadata:
                    c = dict(self.metadata[nid])
                    c["score"] = score
                    chunks.append(c)
                    seen.add(nid)

        return chunks

    def append_neighbor_chunks(
        self,
        ranked_chunks: List[Dict],
        window: int = 1,
    ) -> List[Dict]:
        """
        Preserve rerank order; append adjacent verses after each hit (generation context).
        """
        result: List[Dict] = []
        seen: set = set()

        for chunk in ranked_chunks:
            cid = chunk["id"]
            score = chunk.get("rerank_score", chunk.get("score", 1.0))

            if cid not in seen:
                result.append(dict(chunk))
                seen.add(cid)

            idx = self.rev_id_map.get(cid)
            if idx is None:
                continue

            for i in range(idx - window, idx + window + 1):
                if i == idx:
                    continue
                nid = self.id_map.get(str(i))
                if not nid or nid in seen or nid not in self.metadata:
                    continue
                neighbor = dict(self.metadata[nid])
                neighbor["score"] = score
                neighbor["is_neighbor"] = True
                result.append(neighbor)
                seen.add(nid)

        return result

    def retrieve(
        self,
        query: str,
        top_k: int = FAISS_TOP_K,
        plan: QueryPlan | None = None,
    ):
        plan = plan or build_query_plan(query)
        faiss_k = plan.faiss_k if plan.faiss_k else top_k
        bm25_k = plan.bm25_k if plan.bm25_k else top_k

        norm = self.normalizer.normalize(query, offline_first=True)

        mw = self.mw_grounder.ground(query, norm["entities"])
        if mw["sources_used"]:
            print(f"   MW sources fired : {mw['sources_used']}")
            print(f"   MW augmented     : {mw['augmented'][:120]}")

        exp_faiss, exp_bm25, domains = expand_query(query, self.groq_client)

        combined_faiss = f"{norm['augmented']} {mw['augmented']} {exp_faiss}"
        faiss_query = " ".join(dict.fromkeys(combined_faiss.split()))

        combined_bm25 = f"{norm['augmented']} {exp_bm25}"
        valid_tokens = [
            t for t in combined_bm25.split() if t in query.split() or t in self.bm25.idf
        ]
        bm25_query = " ".join(dict.fromkeys(valid_tokens))

        variants = faiss_variants_for_plan(plan)

        if len(variants) > 1:
            vecs = self.embed_queries_batch(variants)
            all_faiss_results = []
            for vec in vecs:
                all_faiss_results.extend(self.faiss_search(vec[np.newaxis, :], faiss_k))
        else:
            vec = self.embed_query(variants[0])
            all_faiss_results = self.faiss_search(vec, faiss_k)

        all_bm25_results = self.bm25_search(bm25_query, bm25_k)

        print(f"   Variants         : {len(variants)}")
        print(f"   FAISS base query : {faiss_query[:120]}...")
        print(f"   BM25 tokens      : {bm25_query.split()}")

        ranked_ids = self.reciprocal_rank_fusion(
            all_faiss_results,
            all_bm25_results,
            domains,
        )

        conf = confidence_from_faiss(all_faiss_results)
        if plan.rerank_policy == "skip":
            conf["skip_rerank"] = True
            conf["rerank_top_n"] = min(5, faiss_k)

        plan = apply_confidence_to_plan(plan, conf)
        status_flags = dict(norm.get("status_flags", {}))
        status_flags["mw_sources"] = mw.get("sources_used", [])
        status_flags.update(plan.status_flags)

        retrieval_meta = {
            **conf,
            "intent": plan.intent,
            "plan": plan,
            "rerank_policy": plan.rerank_policy,
            "neighbor_policy": plan.neighbor_policy,
            "generation_mode": plan.generation_mode,
            "status_flags": status_flags,
            "faiss_query": faiss_query,
            "bm25_query": bm25_query,
            "domains": domains,
        }

        chunks = [dict(self.metadata[cid], score=score) for cid, score in ranked_ids]
        return chunks, faiss_query, bm25_query, domains, retrieval_meta
