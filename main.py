# ============================================================
# BHAIRAV AI - FASTAPI MAIN APPLICATION
# ============================================================
import sys
import time

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')
except Exception:
    pass
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chitchat import get_chitchat_responses
from config import EMBEDDING_MODEL, GROQ_MODEL, NEIGHBOR_WINDOW, RERANK_TOP_N
from generator import Generator
from query_plan import build_query_plan, neighbors_enabled
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
    intent: Optional[str] = None
    confidence_band: Optional[str] = None
    latencies: Optional[dict] = None


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
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root index.html is served via StaticFiles mounted at "/" below


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

    plan = build_query_plan(query_text)

    if plan.skip_pipeline:
        subtype = plan.chitchat_subtype or "fallback"
        responses = get_chitchat_responses()
        answer = responses.get(subtype, responses.get("fallback", ""))
        return QueryResponse(
            query=query_text,
            answer=answer,
            citations=[],
            retrieved_chunks=[],
            sources_used=[],
            intent="chitchat",
            confidence_band="high",
            latencies={
                "vidyut": 0.0,
                "retrieve": 0.0,
                "rerank": 0.0,
                "context": 0.0,
                "generate": 0.0,
                "total": 0.0
            }
        )

    safe_query_text = query_text.encode('ascii', 'backslashreplace').decode('ascii')
    print(f"\n--- PROFILING QUERY: '{safe_query_text}' | intent={plan.intent} ---")
    start_time = time.time()

    t0 = time.time()
    candidates, faiss_query, bm25_query, detected_domains, retrieval_meta = (
        await run_in_threadpool(retriever.retrieve, query_text, plan.faiss_k, plan)
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

    plan = retrieval_meta.get("plan", plan)
    t0 = time.time()
    script = plan.script  # already computed by query_classifier — avoids duplicated detection logic
    rerank_top_n = retrieval_meta.get("rerank_top_n", request.top_k or RERANK_TOP_N)
    top_chunks = await run_in_threadpool(
        reranker.rerank,
        query_text,
        candidates,
        rerank_top_n,
        script,
        retrieval_meta,
    )

    band = retrieval_meta.get("confidence_band", "low")
    if neighbors_enabled(plan, band):
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
    context_cap = 5 if plan.intent == "lexical" else 15
    aggregated_chunks = final_chunks[:context_cap]
    answer, citations = await run_in_threadpool(
        generator.generate,
        query_text,
        aggregated_chunks,
        plan.generation_mode,
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

    # Vidyut sandhi splitter not yet integrated (Sprint 5+) — report 0.0 until live
    t_vidyut = 0.0
    t_retrieve_net = max(0.001, t_retrieve)

    latencies = {
        "vidyut": round(t_vidyut, 3),
        "retrieve": round(t_retrieve_net, 3),
        "rerank": round(t_rerank, 3),
        "context": 0.001,
        "generate": round(t_gen, 3),
        "total": elapsed
    }

    return QueryResponse(
        query=query_text,
        answer=answer,
        citations=citation_objects,
        retrieved_chunks=retrieved_chunks,
        sources_used=sources_used,
        intent=plan.intent,
        confidence_band=band,
        latencies=latencies,
    )


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
