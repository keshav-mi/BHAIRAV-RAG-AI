# ============================================================
# BHAIRAV AI - FASTAPI MAIN APPLICATION
# ============================================================
# Run with:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# API docs at:
#   http://localhost:8000/docs
# ============================================================

import time
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import EMBEDDING_MODEL, GROQ_MODEL, RERANK_TOP_N
from generator import Generator
from reranker import Reranker
from retriever import Retriever


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    include_citations: Optional[bool] = True


class Citation(BaseModel):
    id: str
    source: str
    book: str
    chapter: Optional[Union[str, int]] = None
    verse: Optional[Union[str, int]] = None
    text: str
    tier: int


class RetrievedChunk(BaseModel):
    id: str
    source: str
    book: str
    chapter: Optional[Union[str, int]] = None
    verse: Optional[Union[str, int]] = None
    hindi_summary: str
    tier: int
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    sources_used: List[str]


class HealthResponse(BaseModel):
    status: str
    faiss_vectors: int
    bm25_corpus_size: int
    metadata_entries: int
    embedding_model: str
    llm_model: str


# Global instances
retriever: Retriever = None
reranker: Reranker = None
generator: Generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, reranker, generator

    print("\n" + "=" * 50)
    print("BHAIRAV AI - STARTING UP")
    print("=" * 50)

    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()

    print("\nAll components loaded - Bhairav AI is ready\n")

    yield

    print("\nBhairav AI shutting down")
    if retriever and hasattr(retriever, "normalizer"):
        retriever.normalizer.close()


app = FastAPI(
    title="Bhairav AI",
    description="Multilingual RAG system for Dharmic primary sources",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Bhairav AI",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Health check - verifies all components are loaded correctly.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    return HealthResponse(
        status="healthy",
        faiss_vectors=retriever.faiss_index.ntotal,
        bm25_corpus_size=len(retriever.chunk_ids),
        metadata_entries=len(retriever.metadata),
        embedding_model=EMBEDDING_MODEL,
        llm_model=GROQ_MODEL,
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(request: QueryRequest):
    """
    Main query endpoint.

    Full pipeline:
    1. Hybrid retrieval (FAISS + BM25 + RRF)
    2. Cross-encoder reranking
    3. LLM generation
    4. Return answer + citations
    """
    if not retriever or not reranker or not generator:
        raise HTTPException(status_code=503, detail="System not ready")

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n--- PROFILING QUERY: '{query_text}' ---")
    start_time = time.time()

    # Step 1: Hybrid retrieval
    t0 = time.time()
    candidates, faiss_query, bm25_query, detected_domains = await run_in_threadpool(
        retriever.retrieve, query_text, request.top_k or 20
    )
    t_retrieve = time.time() - t0
    print(f"1. Retrieval & APIs took : {t_retrieve:.2f} seconds")

    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # Step 2: Rerank
    t0 = time.time()
    script = "devanagari" if any("\u0900" <= c <= "\u097F" for c in query_text) else "english"
    top_chunks = await run_in_threadpool(
        reranker.rerank,
        query_text,
        candidates,
        top_n=request.top_k or RERANK_TOP_N,
        script=script,
    )

    # Step 2.5: Expand neighbors after rerank
    reranked_ids = [(c["id"], c.get("rerank_score", 1.0)) for c in top_chunks]
    expanded_chunks = retriever.get_neighbor_chunks(reranked_ids, window=1)

    seen = set()
    final_chunks = []
    for c in expanded_chunks:
        if c["id"] not in seen:
            final_chunks.append(c)
            seen.add(c["id"])

    t_rerank = time.time() - t0
    print(f"2. Reranker (BGE-M3) took: {t_rerank:.2f} seconds")

    # Step 3: LLM generation
    t0 = time.time()
    aggregated_chunks = final_chunks[:15]
    answer, citations = await run_in_threadpool(
        generator.generate, query_text, aggregated_chunks
    )
    t_gen = time.time() - t0
    print(f"3. LLM generation took  : {t_gen:.2f} seconds")

    # Step 4: Build response
    elapsed = round(time.time() - start_time, 2)
    print(f"TOTAL TIME: {elapsed} seconds\n")

    retrieved_chunks = [
        RetrievedChunk(
            id=c["id"],
            source=c["source"],
            book=c["book"],
            chapter=c.get("chapter"),
            verse=c.get("verse"),
            hindi_summary=c["hindi_summary"],
            tier=c["tier"],
            score=round(c.get("rerank_score", c.get("score", 0.0)), 4),
        )
        for c in top_chunks
    ]

    citation_objects = (
        [
            Citation(
                id=cit["id"],
                source=cit["source"],
                book=cit.get("book_canonical", cit.get("book", "")),
                chapter=cit.get("chapter"),
                verse=cit.get("verse"),
                text=cit["text"],
                tier=cit["tier"],
            )
            for cit in citations
        ]
        if request.include_citations
        else []
    )

    sources_used = list({c["source"] for c in top_chunks})
    print(f"Query processed in {elapsed}s | Sources: {sources_used}")

    return QueryResponse(
        query=query_text,
        answer=answer,
        citations=citation_objects,
        retrieved_chunks=retrieved_chunks,
        sources_used=sources_used,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
