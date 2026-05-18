# ============================================================
# BHAIRAV AI - RETRIEVER v8 (OPTIMIZED)
# Key fixes:
#   - Batch embed all variants at once (not serial)
#   - Remove window=1 before reranker (do it AFTER)
#   - Simplify multi-query to single variant for now
# ============================================================

import json
import pickle
from typing import List

import faiss
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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
)
from multi_query import generate_query_variants
from mw_grounding import NonEntityGrounder
from query_expander import expand_query
from query_normalizer import QueryNormalizer

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
        prefixed = f"Meaning: {query.strip()}"
        vec = self.embed_model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def embed_queries_batch(self, queries: List[str]) -> np.ndarray:
        prefixed = [f"Meaning: {q.strip()}" for q in queries]
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
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in idxs if scores[i] > 0]

    def reciprocal_rank_fusion(self, faiss_res, bm25_res, domains, k=RRF_K):
        rrf = {}

        for rank, (cid, _) in enumerate(faiss_res):
            rrf[cid] = rrf.get(cid, 0) + 1 / (k + rank + 1)

        for rank, (cid, _) in enumerate(bm25_res):
            rrf[cid] = rrf.get(cid, 0) + 1 / (k + rank + 1)

        if domains:
            for cid in rrf:
                source = self.metadata.get(cid, {}).get("source", "")
                if source in domains:
                    rrf[cid] *= DOMAIN_BOOST

        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    def get_neighbor_chunks(self, ranked_ids, window=1):
        """Expand context around top chunks. Call AFTER reranking, not before."""
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

    def retrieve(self, query: str, top_k: int = FAISS_TOP_K):
        # STEP 1 - normalize
        norm = self.normalizer.normalize(query)

        # STEP 2 - MW grounding
        mw = self.mw_grounder.ground(query, norm["entities"])
        if mw["sources_used"]:
            print(f"   MW sources fired : {mw['sources_used']}")
            print(f"   MW augmented     : {mw['augmented'][:120]}")

        # STEP 3 - expansion
        exp_faiss, exp_bm25, domains = expand_query(query, self.groq_client)

        # STEP 4 - FAISS query
        combined_faiss = f"{norm['augmented']} {mw['augmented']} {exp_faiss}"
        faiss_query = " ".join(dict.fromkeys(combined_faiss.split()))

        # STEP 5 - BM25 query
        combined_bm25 = f"{norm['augmented']} {exp_bm25}"
        valid_tokens = [
            t for t in combined_bm25.split() if t in query.split() or t in self.bm25.idf
        ]
        bm25_query = " ".join(dict.fromkeys(valid_tokens))

        # STEP 6 - Multi-query
        variants = generate_query_variants(faiss_query)

        if len(variants) > 1:
            vecs = self.embed_queries_batch(variants)
            all_faiss_results = []
            for vec in vecs:
                all_faiss_results.extend(self.faiss_search(vec[np.newaxis, :], top_k))
        else:
            vec = self.embed_query(variants[0])
            all_faiss_results = self.faiss_search(vec, top_k)

        all_bm25_results = self.bm25_search(bm25_query, top_k)

        print(f"   Variants         : {len(variants)}")
        print(f"   FAISS base query : {faiss_query[:120]}...")
        print(f"   BM25 tokens      : {bm25_query.split()}")

        # STEP 7 - Merge
        ranked_ids = self.reciprocal_rank_fusion(
            all_faiss_results,
            all_bm25_results,
            domains,
        )

        # STEP 8 - Return ranked ids only
        chunks = [dict(self.metadata[cid], score=score) for cid, score in ranked_ids]
        return chunks, faiss_query, bm25_query, domains
