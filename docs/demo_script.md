# Demo Script

1. Show the repository structure and point to `financial_rag/`, merged `mem2/`, `data/`, `tests/`, `docs/`, and the local ignored `sec_filings/` EDGAR cache.
2. Run `python main.py run-all --corpus sample --benchmark sample` to demonstrate a full offline workflow.
3. Open `results/evaluation_summary.csv` and compare baseline, standard RAG, and reranked RAG.
4. Ask a question with `python main.py ask "What were Apple's 2023 net sales?"`.
5. Launch `streamlit run demo_app.py` and show the interactive RAG mode.
6. In the dashboard, show **Build / Refresh Index**, **Compare Methods**, **Evidence Trace**, and **Run Evaluation**.
7. Run `python -m mem2.runner --questions data/sample_benchmarks/sample_financial_qa.jsonl --out results/mem2_runs/sample --modes baseline standard_rag reranked_lexical` to show the merged Member 2 batch-runner output schema.
8. Explain the two corpus modes: EDGAR for the live filing demo and `benchmark` evidence corpus for controlled FinanceBench/FinDER evaluation.
9. Show `results/experiments/experiment_summary.csv` and highlight that the MS MARCO cross-encoder reranker improves FinDER Recall@1 from 0.030 to 0.100 and MRR from 0.047 to 0.128 on the 100-question neural reranker subset.
