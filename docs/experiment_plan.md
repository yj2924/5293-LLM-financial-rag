# Experiment Plan

## Dataset Roles

- EDGAR is the retrieval corpus. Use `python main.py build-index --corpus edgar --download-edgar` to collect 10-K filings, clean them, chunk them, embed them, and store them in the local vector store.
- FinanceBench is the main QA benchmark. Use `python main.py evaluate --benchmark financebench --limit 150` for a small public run, or increase the limit when full benchmark access is available.
- FinDER is the optional retrieval/reranking stress test. Use `python main.py evaluate --benchmark finder --limit 200` after the core EDGAR and FinanceBench experiments are complete.

## Experimental Conditions

1. Baseline LLM without retrieval: answers without document evidence.
2. Standard RAG: retrieves top-k chunks and generates an answer from those chunks.
3. Reranked RAG: retrieves a larger candidate set, reranks it, then generates from the top reranked chunks.

The active `financial_rag` pipeline provides the reproducible default path. The merged `mem2/` pipeline provides the richer system-architecture path for demos and ablations: `baseline`, `standard_rag`, `reranked_bge`, and `reranked_msmarco`.

## Metrics

- Retrieval quality: Recall@1, Recall@3, Recall@5, and MRR when evidence annotations are available.
- Answer quality: token F1 against benchmark answers.
- Grounding: citation rate and faithfulness proxy.
- Hallucination: `1 - faithfulness proxy`, plus manual review of `results/evaluation_records.jsonl`.

## Team Ownership

- Member 1, Data/Retrieval/Reproducibility: EDGAR collection, chunking, embeddings, vector store, retrieval metrics, README/setup.
- Member 2, RAG/Reranking/System Architecture: baseline QA, standard RAG, reranked RAG, prompt/answer generation, context selection and tradeoffs.
- Member 3, Hallucination Evaluation/Analysis/Demo: faithfulness criteria, benchmark scripts, comparisons, error analysis, presentation figures, and demo flow.

## Report Figures And Tables

- Data table from `results/data_source_summary.csv`.
- Metrics table from `results/evaluation_summary.csv`.
- Case studies from `results/evaluation_records.jsonl`, especially cases where reranking changes the top source.
- Member 2 ablation outputs from `results/mem2_runs/`, especially the imported dev run and new smoke-test runs.
