# ============================================================
# BHAIRAV AI - FASTAPI MAIN APPLICATION
# ============================================================

import time
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import EMBEDDING_MODEL, GROQ_MODEL, NEIGHBOR_WINDOW, RERANK_TOP_N
from confidence import neighbors_allowed
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
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Bhairav AI",
        "version": "2.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
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
    if not retriever or not reranker or not generator:
        raise HTTPException(status_code=503, detail="System not ready")

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n--- PROFILING QUERY: '{query_text}' ---")
    start_time = time.time()

    t0 = time.time()
    candidates, faiss_query, bm25_query, detected_domains, retrieval_meta = (
        await run_in_threadpool(
            retriever.retrieve, query_text, request.top_k or 40
        )
    )
    t_retrieve = time.time() - t0
    print(f"1. Retrieval & APIs took : {t_retrieve:.2f} seconds")
    print(
        f"   FAISS confidence     : band={retrieval_meta.get('confidence_band')} "
        f"top1={retrieval_meta.get('faiss_top1_score')} "
        f"margin={retrieval_meta.get('faiss_margin')}"
    )

    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    t0 = time.time()
    script = "devanagari" if any("\u0900" <= c <= "\u097F" for c in query_text) else "english"
    top_k = request.top_k or RERANK_TOP_N
    top_chunks = await run_in_threadpool(
        reranker.rerank,
        query_text,
        candidates,
        top_k,
        script,
        retrieval_meta,
    )

    if neighbors_allowed(retrieval_meta.get("confidence_band", "low")):
        final_chunks = await run_in_threadpool(
            retriever.append_neighbor_chunks,
            top_chunks,
            NEIGHBOR_WINDOW,
        )
    else:
        final_chunks = top_chunks

    t_rerank = time.time() - t0
    print(f"2. Reranker (BGE-M3) took: {t_rerank:.2f} seconds")

    t0 = time.time()
    aggregated_chunks = final_chunks[:15]
    answer, citations = await run_in_threadpool(
        generator.generate, query_text, aggregated_chunks
    )
    t_gen = time.time() - t0
    print(f"3. LLM generation took  : {t_gen:.2f} seconds")

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
