# Bhairav evaluation

## Primary retrieval KPI

Use **`--stage after_rerank`** for regression gates (MRR, Recall@k).

`after_neighbors` is **diagnostic only** — neighbor verses are appended for generation context and will deflate Recall@1 if scored as a ranked list.

## Baseline

```bash
python -m eval.run_baseline_eval --skip-mine
```

## Offline eval (Kaggle / no external APIs)

```bash
set MW_NETWORK_ENABLED=false
set WIKIDATA_ENABLED=false
set INDICXLIT_FALLBACK=false
python -m eval.bhairav_retrieval_eval --stage after_rerank
```

## Harrier embedding contract

```bash
python scripts/verify_embedding_contract.py
```

## Corpus expansion (separate from latency sprints)

Adding texts requires re-embedding and a new FAISS/BM25 build — track as its own milestone after Sprint 4 stability.
