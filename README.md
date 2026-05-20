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

## Plan v3 (through Sprint 3)

Intent router, chitchat guard, layered transliteration, offline entity resolver, and maintenance scripts. See `data/README.md`.

**Offline eval baseline:**
```bash
set MW_NETWORK_ENABLED=false
set WIKIDATA_ENABLED=false
set INDICXLIT_FALLBACK=false
python -m eval.bhairav_retrieval_eval --stage after_rerank
```

Optional: `pip install ai4bharat-transliteration` ([AI4Bharat IndicXlit](https://github.com/AI4Bharat/IndicXlit)).

## Run API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
