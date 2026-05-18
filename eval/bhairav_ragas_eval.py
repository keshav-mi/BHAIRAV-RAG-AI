"""
End-to-end RAG eval via RAGAS (faithfulness, context precision/recall).

Requires API running OR use --direct to call Generator + Retriever in-process.

Usage:
  uvicorn main:app --host 127.0.0.1 --port 8000
  python -m eval.bhairav_ragas_eval --queries eval/data/ragas_queries.json --limit 20

  python -m eval.bhairav_ragas_eval --direct --limit 10
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests

from config import GROQ_API_KEY, GROQ_MODEL

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = EVAL_DIR / "data" / "ragas_queries.json"
DEFAULT_OUT = EVAL_DIR / "results"

API_URL = "http://127.0.0.1:8000/query"


def default_queries() -> list[dict]:
    """Small hand-picked set — extend eval/data/ragas_queries.json."""
    return [
        {"query": "Who was Devavrata?", "source_hint": "Mahabharata"},
        {"query": "गीता के अनुसार कर्म क्या है?", "source_hint": "Bhagavad Gita"},
        {"query": "राम ने रावण को क्यों मारा?", "source_hint": "Valmiki Ramayana"},
        {"query": "ऋग्वेद में इंद्र का वर्णन", "source_hint": "Rigveda"},
        {"query": "karna ko radheya kyu kaha gaya", "source_hint": "Mahabharata"},
        {"query": "Did Arjuna use nuclear weapons?", "source_hint": None},
    ]


def load_queries(path: Path | None, limit: int | None) -> list[dict]:
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("queries", [])
    else:
        rows = default_queries()

    if limit:
        rows = rows[:limit]
    return rows


def fetch_via_api(query: str, top_k: int = 15, timeout: int = 300) -> dict:
    r = requests.post(
        API_URL,
        json={"query": query, "top_k": top_k, "include_citations": True},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def fetch_direct(query: str, top_k: int = 15) -> dict:
    from eval.pipeline import load_components, retrieve_ids, unload
    from generator import Generator

    retriever, reranker = load_components()
    generator = Generator()

    candidates, _, _, _ = retriever.retrieve(query, top_k=40)
    script = (
        "devanagari"
        if any("\u0900" <= c <= "\u097f" for c in query)
        else "english"
    )
    top_chunks = reranker.rerank(query, candidates, top_n=top_k, script=script)
    reranked_ids = [(c["id"], c.get("rerank_score", 1.0)) for c in top_chunks]
    expanded = retriever.get_neighbor_chunks(reranked_ids, window=1)

    seen = set()
    final = []
    for c in expanded:
        if c["id"] not in seen:
            final.append(c)
            seen.add(c["id"])

    answer, citations = generator.generate(query, final[:15])
    unload()

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": [
            {
                "id": c["id"],
                "hindi_summary": c.get("hindi_summary", ""),
                "text": c.get("text", ""),
                "source": c.get("source", ""),
            }
            for c in top_chunks
        ],
        "citations": citations,
    }


def contexts_from_response(data: dict) -> list[str]:
    """RAGAS needs actual passage strings, not chunk IDs."""
    ctx = []
    for c in data.get("retrieved_chunks", []):
        summary = (c.get("hindi_summary") or "").strip()
        text = (c.get("text") or "").strip()
        passage = f"{summary}\n{text}".strip() if text else summary
        if passage:
            ctx.append(passage)
    return ctx


def _build_metrics(has_reference: bool, include_answer_relevancy: bool) -> list:
    """
    RAGAS 0.4.x: use legacy Metric instances (ragas.metrics._*).
    Collections metrics (ragas.metrics.collections) are NOT compatible with evaluate().
    """
    from ragas.metrics._faithfulness import faithfulness

    metrics = [faithfulness]

    if has_reference:
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall

        metrics.extend([context_precision, context_recall])

    if include_answer_relevancy:
        from ragas.metrics._answer_relevance import answer_relevancy

        metrics.append(answer_relevancy)

    return metrics


def _build_embeddings(include_answer_relevancy: bool):
    if not include_answer_relevancy:
        return None
    # answer_relevancy needs embeddings; use small local model (no OpenAI key)
    from ragas.embeddings import HuggingfaceEmbeddings

    return HuggingfaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def run_ragas(
    rows: list[dict],
    out_dir: Path,
    include_answer_relevancy: bool = False,
) -> dict:
    try:
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        raise ImportError(
            "Install eval deps: pip install -r requirements-eval.txt"
        ) from e

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY required for RAGAS judge")

    questions, answers, contexts_list, references = [], [], [], []
    has_reference = False

    for i, item in enumerate(rows, 1):
        q = item["query"] if isinstance(item, dict) else item
        print(f"  [{i}/{len(rows)}] {q[:70]}...")
        data = item.get("_response")
        if not data:
            raise ValueError("Pre-fetch responses before calling run_ragas")

        questions.append(q)
        answers.append(data.get("answer", ""))
        contexts_list.append(contexts_from_response(data))
        ref = (item.get("reference") or item.get("ground_truth") or "").strip()
        references.append(ref)
        if ref:
            has_reference = True
        print(f"       {len(contexts_list[-1])} contexts")

    payload = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    if has_reference:
        payload["reference"] = references

    dataset = Dataset.from_dict(payload)

    # Faithfulness runs multi-step NLI; default max_tokens can truncate on long answers
    llm = LangchainLLMWrapper(
        ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=1024,      # was 4096 — too slow for 8B
            request_timeout=120,  # was default 60s — caused timeouts
        )
    )

    metrics = _build_metrics(has_reference, include_answer_relevancy)
    embeddings = _build_embeddings(include_answer_relevancy)

    names = [getattr(m, "name", type(m).__name__) for m in metrics]
    print(f"  Metrics: {', '.join(names)}")
    if not has_reference:
        print("  Tip: add \"reference\" in ragas_queries.json for context_precision")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        batch_size=1,
        raise_exceptions=False,
    )

    scores = {}
    df = result.to_pandas()
    for col in df.columns:
        if col not in ("question", "contexts", "answer"):
            try:
                scores[col] = float(df[col].mean())
            except (TypeError, ValueError):
                pass

    report = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(rows),
        "scores": scores,
        "per_query": df.to_dict(orient="records"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"ragas_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RAGAS SUMMARY")
    print("=" * 60)
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved -> {path}")
    return report


def main():
    p = argparse.ArgumentParser(description="Bhairav RAGAS end-to-end eval")
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--direct", action="store_true", help="In-process pipeline (no API)")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--with-answer-relevancy",
        action="store_true",
        help="Also run answer_relevancy (downloads MiniLM; slow on 8GB RAM)",
    )
    args = p.parse_args()

    rows = load_queries(args.queries if args.queries.exists() else None, args.limit)

    for item in rows:
        q = item["query"] if isinstance(item, dict) else item
        try:
            if args.direct:
                item["_response"] = fetch_direct(q, top_k=args.top_k)
            else:
                item["_response"] = fetch_via_api(q, top_k=args.top_k)
        except Exception as e:
            print(f"Failed: {q} -> {e}")
            item["_response"] = {
                "query": q,
                "answer": "",
                "retrieved_chunks": [],
            }

    ok = [r for r in rows if contexts_from_response(r.get("_response", {}))]
    if not ok:
        print("No successful responses with contexts. Start API or fix errors.")
        return

    run_ragas(ok, args.out_dir, include_answer_relevancy=args.with_answer_relevancy)


if __name__ == "__main__":
    main()
