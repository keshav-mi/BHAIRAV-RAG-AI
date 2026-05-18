# Bhairav RAG AI

Private multilingual Dharmic RAG system (Hindi/English/Hinglish) built with FastAPI, hybrid retrieval (FAISS + BM25), reranking, and grounded answer generation.

## What’s in this repo
- API service: `main.py`
- Retrieval stack: `retriever.py`, `query_normalizer.py`, `query_expander.py`, `mw_grounding.py`
- Reranker: `reranker.py`
- Generator: `generator.py`
- Evaluation scripts: `eval/`

## Setup
1. Create and activate Python 3.10+ environment
2. Install deps:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-eval.txt` (for eval pipeline)
3. Add `.env` with required API keys (not committed)

## Run API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
