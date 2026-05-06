# Financial RAG: EDGAR + FinanceBench + Reranking

This project implements the proposal topic: financial-domain question answering with Retrieval-Augmented Generation (RAG), reranking, and hallucination/faithfulness evaluation.

## What Is Implemented

- EDGAR corpus pipeline: downloads and loads SEC 10-K filings, cleans text, chunks documents, embeds chunks, and stores a vector index.
- Baseline and RAG systems: compares no-retrieval QA, standard RAG, and RAG with a reranking stage.
- Benchmarks: supports FinanceBench as the main benchmark and FinDER as an optional retrieval/reranking stress test.
- Evaluation: reports Recall@k, MRR, answer token F1, citation rate, faithfulness proxy, and hallucination proxy rate.
- Reproducibility: includes offline sample corpus, sample benchmark, CLI, demo app, imported Member 2 pipeline code, and tests.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py run-all --corpus sample --benchmark sample
python main.py ask "What were Apple's 2023 net sales?"
```

The sample run is intentionally small so the whole pipeline works without network access. Results are written to `results/evaluation_metrics.json`, `results/evaluation_summary.csv`, and `results/evaluation_records.jsonl`.

## Real EDGAR Corpus

Set a SEC-compliant identity before downloading filings:

```bash
export SEC_COMPANY_NAME="Your Name or Organization"
export SEC_EMAIL="your.email@example.com"
python main.py build-index --corpus edgar --download-edgar --limit-per-company 1 --after 2023-01-01
```

By default the downloader targets major public companies and 10-K filings. Local filings are read from `~/sec-edgar-filings` unless `FINRAG_EDGAR_DIR` is set.

This merged workspace also contains a local EDGAR cache at `sec_filings/sec-edgar-filings` copied from the downloaded `5293-LLM-financial-rag-main` folder. It is ignored by Git because it is large, but the pipeline automatically uses it when present.

## Benchmark Runs

```bash
python main.py build-index --corpus benchmark
python main.py evaluate --benchmark financebench --limit 150
python main.py evaluate --benchmark finder --limit 200
```

FinanceBench is the main benchmark. FinDER is optional and is best used after the EDGAR + FinanceBench experiments are complete.

For controlled benchmark evaluation, `--corpus benchmark` builds the vector store from cached FinanceBench/FinDER evidence strings. This complements the EDGAR demo corpus by ensuring the benchmark questions have their annotated evidence in the searchable corpus.

To run the neural reranker comparison used in the final analysis:

```bash
FINRAG_RERANKER_BACKEND=cross-encoder \
FINRAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
python main.py evaluate --benchmark finder --limit 100
```

Experiment snapshots are stored under `results/experiments/`, including `experiment_summary.csv`.

## Demo

```bash
streamlit run demo_app.py
```

The demo can build the index, run an evaluation, and answer questions with standard or reranked RAG.

Recommended recording flow:

1. Select `sample` for a fast walkthrough, or `edgar` to rebuild on the merged local SEC filing cache.
2. Click **Build / Refresh Index**.
3. Ask a concrete question such as `What were Apple's 2023 net sales?`.
4. Click **Compare Methods** and show baseline vs standard RAG vs reranked RAG.
5. Open the **Evidence** tab and point out the cited filing chunks.
6. Click **Run Evaluation** and show the metrics table.

## Imported Member 2 Pipelines

The downloaded `5293-LLM-financial-rag-main` folder has been merged under `mem2/`. This keeps the richer Member 2 architecture available: prompt templates, baseline/standard/reranked pipeline classes, batch runner, company-aware filtering, cross-encoder reranker options, and the prior dev-eval outputs.

```bash
python main.py build-index --corpus sample
python -m mem2.runner --questions data/sample_benchmarks/sample_financial_qa.jsonl --out results/mem2_runs/sample --modes baseline standard_rag reranked_lexical
```

`mem2/configs.yaml` defaults to the `echo` LLM backend and an offline lexical reranker so smoke tests do not download large model weights. For real generation, set `MEM2_LLM_BACKEND=openai_compat` with an OpenAI-compatible endpoint, or switch the config to `qwen_local`. For neural reranking, uncomment `bge` or `ms-marco` in `mem2/configs.yaml`.

## Tests

```bash
python -m unittest discover tests
```

## Repository Layout

- `financial_rag/`: ingestion, embeddings, retrieval, reranking, generation, evaluation, and CLI.
- `mem2/`: merged Member 2 RAG/reranking architecture from the downloaded project.
- `data/sample_corpus/`: offline EDGAR-style filing excerpts for reproducible smoke tests.
- `data/sample_benchmarks/`: offline QA examples with evidence annotations.
- `docs/`: rubric compliance, experiment plan, and demo script.
- `results/`: generated vector store, chunks, metrics, records, and summaries.

## Sources

- SEC EDGAR data access: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- FinanceBench paper: https://arxiv.org/abs/2311.11944
- FinanceBench dataset page: https://huggingface.co/datasets/PatronusAI/financebench
- FinDER paper: https://arxiv.org/abs/2504.15800
- FinDER dataset page: https://huggingface.co/datasets/Linq-AI-Research/FinDER
