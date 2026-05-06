# Merge Notes: Downloaded Member 2 Repo

Source inspected:

`/Users/yiyi/Downloads/5293-LLM-financial-rag-main/5293-LLM-financial-rag-main`

Merged into active repo:

- `mem2/`: RAG, reranking, prompt, LLM-client, company-filter, batch-runner, and dev-eval code.
- `results/mem2_runs/`: prior Member 2 dev-eval JSONL outputs.
- `docs/imported/mem2_source_DATA_README.md`: source data-layer snapshot from the downloaded repo.
- `sec_filings/`: local EDGAR filing cache copied from the downloaded repo for local demos; ignored by Git because it is large.

Not copied into active repo:

- The large downloaded `results/` payload, because it is better treated as a reproducible local artifact than GitHub source files. The SEC filing cache was copied locally but is ignored by Git.

Compatibility patch:

- `mem2/retriever_adapter.py` now first tries the active `financial_rag` vector store and falls back to the original FAISS + MiniLM adapter if needed.
- `mem2/configs.yaml` defaults to `echo` so smoke tests do not download Qwen weights.
- `mem2/reranker.py` now includes an offline lexical reranker. The original BGE and MS-MARCO cross-encoder options remain available in config.
