# Financial RAG Evidence Dashboard: EDGAR Retrieval, Reranking, and Hallucination Evaluation

## Abstract

This project builds and evaluates a retrieval-augmented generation system for financial question answering. The system uses SEC EDGAR filings as a real financial document corpus, supports FinanceBench as the main financial QA benchmark, and uses FinDER as an optional stress test for short, ambiguous financial search queries. We compare three methods: a no-retrieval baseline, standard RAG, and reranked RAG. The final system includes data ingestion, document chunking, vector indexing, hybrid retrieval, lexical reranking, neural cross-encoder reranking, answer generation with citations, automated evaluation metrics, and a Streamlit dashboard for demonstration.

The strongest experimental finding is that retrieval and reranking improve groundedness relative to the no-retrieval baseline. In the neural reranker experiment on a 100-question FinDER subset, MS MARCO cross-encoder reranking improved Recall@1 from 0.030 to 0.100, Recall@5 from 0.090 to 0.170, and MRR from 0.047 to 0.128 compared with standard RAG. This supports the project hypothesis that reranking can improve evidence selection in finance RAG, especially for realistic, search-style queries.

## 1. Introduction

Financial filings are long, technical, and numerically dense. A general language model can often produce fluent answers, but in finance this is not enough: answers must be tied to source evidence because small factual errors can change the interpretation of a company’s performance, liquidity, risk exposure, or valuation. This project addresses that problem by building a financial RAG system grounded in SEC filings and evaluating whether retrieval and reranking reduce unsupported or hallucinated answers.

The project follows the original proposed division of datasets:

- **EDGAR** is the document source for the retrieval corpus. It provides real SEC filings such as 10-Ks.
- **FinanceBench** is the main benchmark for financial open-book question answering.
- **FinDER** is a supplementary benchmark focused on short, realistic, ambiguity-heavy financial queries.

The core research question is:

> Does reranking improve the factual grounding and retrieval quality of a finance RAG system compared with standard RAG and a no-retrieval baseline?

## 2. Team Contributions

### Member 1: Data, Retrieval, and Reproducibility

Member 1 built the data layer. Their work focused on converting raw SEC filings into a searchable index. The pipeline downloads or loads EDGAR filings, strips SEC/SGML-style noise, normalizes text, chunks documents, embeds chunks, and writes a reusable vector store. The original report emphasized practical reproducibility, including a fallback sample corpus and CPU-friendly execution for local hardware. This work provides the foundation for all downstream retrieval and evaluation.

In the final repository, this contribution is represented by:

- `financial_rag/data.py` for EDGAR loading, cleaning, benchmark evidence loading, and chunking.
- `financial_rag/embeddings.py` for local embedding backends.
- `financial_rag/retrieval.py` for vector and hybrid retrieval.
- `results/data_source_summary.csv` for data source documentation.
- `tests/test_pipeline.py` for reproducibility checks.

### Member 2: RAG, Reranking, and System Architecture

Member 2 built the answer-generation and reranking layer. Their report describes three answering modes: baseline no-retrieval QA, standard RAG, and reranked RAG. It also discusses the tradeoff between retrieval quality and latency for neural rerankers such as BGE and MS MARCO. The final repository preserves this work in `mem2/` and integrates the same architecture into the main package.

In the final repository, this contribution is represented by:

- `financial_rag/pipeline.py` for baseline, standard RAG, and reranked RAG comparisons.
- `financial_rag/reranking.py` for lexical and cross-encoder reranking.
- `financial_rag/generation.py` for answer generation with citations.
- `mem2/` for the merged Member 2 batch runner, prompt templates, and reranker variants.

### Member 3: Hallucination Evaluation, Analysis, and Demo

Member 3 owns the evaluation story and final delivery. This includes defining faithfulness and hallucination criteria, running benchmark evaluations, comparing baseline/standard/reranked systems, performing error analysis, and producing the final demo experience.

In the final repository, this contribution is represented by:

- `financial_rag/evaluation.py` for answer F1, citation rate, faithfulness proxy, hallucination proxy, Recall@k, and MRR.
- `results/experiments/experiment_summary.csv` for final experimental comparisons.
- `demo_app.py` for the professional interactive dashboard.
- `docs/demo_script.md` for the recording flow.
- This final report.

## 3. System Architecture

The system has five stages.

1. **Document ingestion.** The pipeline loads SEC EDGAR filings from `sec_filings/sec-edgar-filings` or downloads them with `sec_edgar_downloader`. It also supports a controlled benchmark-evidence corpus built from cached FinanceBench and FinDER evidence strings.

2. **Cleaning and chunking.** Raw filings are cleaned by removing SEC headers, HTML tags, non-breaking spaces, and repeated boilerplate. Documents are chunked into overlapping word windows so long filings can be embedded and retrieved efficiently.

3. **Embedding and retrieval.** Chunks are embedded and stored in a NumPy/FAISS-compatible vector index. The final retriever uses a hybrid dense-plus-lexical score, because financial questions often require exact matching of table labels, years, and accounting terms.

4. **Reranking.** Standard RAG uses the top retrieved chunks directly. Reranked RAG first retrieves a larger candidate set and then reranks it. The final implementation supports an offline lexical reranker and a neural MS MARCO MiniLM cross-encoder reranker.

5. **Answer generation and citations.** The generator produces answers from retrieved context and attaches chunk citations. The current default is an extractive grounded generator for reproducibility. This keeps experiments deterministic and avoids requiring paid API keys.

## 4. Datasets

### EDGAR

EDGAR is used as the real financial document corpus. The local EDGAR cache includes filings for major public companies, and CBOE was added because the first FinDER evaluation subset contained many CBOE questions. After rebuilding the EDGAR index, the local filing corpus contained 11 documents and 24,855 chunks.

### FinanceBench

FinanceBench is used as the main benchmark. The available Hugging Face split cached in this environment contains 150 examples. Each example includes a question, answer, evidence annotation, and document metadata.

### FinDER

FinDER is used as the supplementary stress-test benchmark. The cached local dataset contains 5,703 examples. It is especially useful because the questions are short and search-like, which makes evidence retrieval and reranking more important.

### Controlled Benchmark-Evidence Corpus

One practical issue is that a small EDGAR corpus cannot cover all companies and years referenced by FinanceBench and FinDER. To run a fair controlled benchmark, the final system adds a `benchmark` corpus mode. This mode builds the searchable index from cached FinanceBench and FinDER evidence strings. It does not replace EDGAR as the real filing corpus; it complements EDGAR by making the benchmark evidence available during open-book evaluation.

The controlled benchmark index contains 5,727 documents and 10,647 chunks.

## 5. Methods Compared

### Baseline: No Retrieval

The baseline answers without using retrieved context. In the offline implementation, this baseline intentionally refuses to make unsupported financial claims. This gives a conservative lower bound for citation and faithfulness.

### Standard RAG

Standard RAG retrieves the top chunks and generates an answer from those chunks. It tests whether adding retrieval improves answer support and reduces hallucination risk.

### Reranked RAG

Reranked RAG retrieves a larger candidate set, reranks it, then generates an answer from the strongest evidence. Two reranking setups were tested:

- **Lexical reranking:** local token-overlap reranking for reproducible offline runs.
- **MS MARCO cross-encoder reranking:** neural reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## 6. Evaluation Metrics

The evaluation reports both answer quality and retrieval quality.

- **Answer token F1:** token overlap between generated answer and gold answer.
- **Citation rate:** whether the generated answer includes retrieved evidence citations.
- **Faithfulness proxy:** fraction of answer content tokens supported by retrieved context tokens.
- **Hallucination proxy rate:** `1 - faithfulness_proxy`.
- **Recall@k:** whether gold evidence or expected source appears in the top-k retrieved chunks.
- **MRR:** reciprocal rank of the first evidence hit.

The faithfulness and hallucination metrics are proxies, not human judgments. They are useful for comparing methods consistently, but final claims should acknowledge that they do not replace expert review.

## 7. Experimental Results

### Main Larger Runs

| Experiment | Method | Answer F1 | Faithfulness | Hallucination Proxy | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinanceBench 150, lexical | Baseline | 0.0107 | 0.0000 | 1.0000 | n/a | n/a | n/a |
| FinanceBench 150, lexical | Standard RAG | 0.0384 | 0.7688 | 0.2312 | 0.0467 | 0.0733 | 0.0572 |
| FinanceBench 150, lexical | Reranked RAG | 0.0357 | 0.7907 | 0.2093 | 0.0467 | 0.0667 | 0.0550 |
| FinDER 500, lexical | Baseline | 0.0388 | 0.0000 | 1.0000 | n/a | n/a | n/a |
| FinDER 500, lexical | Standard RAG | 0.0541 | 0.7730 | 0.2270 | 0.0340 | 0.0880 | 0.0511 |
| FinDER 500, lexical | Reranked RAG | 0.0552 | 0.7779 | 0.2221 | 0.0340 | 0.0800 | 0.0475 |

The larger lexical runs show that both RAG methods substantially improve faithfulness over the no-retrieval baseline. On FinanceBench, reranking improves the faithfulness proxy from 0.7688 to 0.7907 and reduces hallucination proxy from 0.2312 to 0.2093. On FinDER, reranking slightly improves answer F1 and faithfulness, but lexical reranking does not improve retrieval Recall@5.

This suggests that lexical reranking helps answer grounding modestly, but it is not strong enough to solve the hardest evidence selection cases.

### Neural Reranker Runs

| Experiment | Method | Answer F1 | Faithfulness | Hallucination Proxy | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinanceBench 50, MS MARCO | Baseline | 0.0076 | 0.0000 | 1.0000 | n/a | n/a | n/a |
| FinanceBench 50, MS MARCO | Standard RAG | 0.0323 | 0.7514 | 0.2486 | 0.0200 | 0.0200 | 0.0200 |
| FinanceBench 50, MS MARCO | Reranked RAG | 0.0327 | 0.7876 | 0.2124 | 0.0200 | 0.0400 | 0.0300 |
| FinDER 100, MS MARCO | Baseline | 0.0405 | 0.0000 | 1.0000 | n/a | n/a | n/a |
| FinDER 100, MS MARCO | Standard RAG | 0.0582 | 0.7779 | 0.2221 | 0.0300 | 0.0900 | 0.0470 |
| FinDER 100, MS MARCO | Reranked RAG | 0.0655 | 0.7677 | 0.2323 | 0.1000 | 0.1700 | 0.1283 |

The neural reranker gives the clearest positive result. On FinDER, MS MARCO reranking increases Recall@1 by 0.070, Recall@5 by 0.080, and MRR by 0.0813. Answer F1 also improves from 0.0582 to 0.0655. This supports the argument from Member 2’s report that neural reranking can improve evidence selection, though it adds latency and model dependency.

FinanceBench remains harder. Even with MS MARCO, retrieval scores are low, likely because many FinanceBench questions use financial analyst phrasing that does not directly match the table row labels in the evidence. For example, a question may ask for capital expenditure while the filing line says “Purchases of property, plant and equipment.” This mismatch limits first-stage recall and makes reranking less effective if the right evidence is not in the candidate set.

## 8. Error Analysis

The main error source is first-stage retrieval. Reranking can only improve results if the correct evidence appears in the candidate pool. When the hybrid retriever misses the relevant evidence entirely, both lexical and neural reranking fail.

Common failure cases include:

- **Synonym mismatch:** Financial concepts are phrased differently in questions and filings, such as “capex” versus “purchases of PP&E.”
- **Table structure loss:** Fixed-size chunks can separate row labels, years, and values.
- **Document mismatch:** A small EDGAR corpus cannot cover all benchmark companies and years.
- **Numerical reasoning:** Extractive answer generation can cite the right area but may not compute ratios or differences correctly.
- **Benchmark ambiguity:** FinDER intentionally uses short, search-style questions, which makes entity resolution harder.

The dashboard helps communicate these errors because it shows the retrieved evidence trace, citations, and method-level metrics in one view.

## 9. Demo and User Interface

The final demo is a Streamlit dashboard called **Financial RAG Evidence Dashboard**. It provides:

- corpus and benchmark controls;
- benchmark limit control;
- index building;
- baseline versus standard RAG versus reranked RAG comparison;
- KPI cards for Recall@1, MRR, and hallucination reduction;
- a method comparison table;
- an evidence trace accordion;
- reproducibility outputs.

The dashboard is designed for a professional SaaS-style desktop presentation so the recorded demo can focus on the research story rather than command-line details.

## 10. Limitations

The current project is strong enough for a course demo and report, but it is not a production-grade finance QA system.

First, the default answer generator is extractive and deterministic. This improves reproducibility, but it is less fluent and less capable of multi-step numerical reasoning than a full LLM. Second, FinanceBench remains difficult because the retriever does not yet have finance-specific synonym expansion or table-aware parsing. Third, the hallucination metric is a token-overlap proxy, not a human audit. Fourth, neural reranking improves FinDER results but adds latency and requires model downloads.

## 11. Future Work

The next improvements are clear:

- add table-aware chunking for financial statements;
- add finance synonym expansion for terms such as capex, PP&E, operating margin, liquidity, and working capital;
- add a real LLM generation backend using OpenAI-compatible APIs or local Ollama models;
- evaluate with human faithfulness labels on a smaller case-study set;
- expand the EDGAR corpus to cover all benchmark companies and years;
- report latency and cost alongside answer quality.

## 12. Conclusion

The final project meets the core proposal goal: it builds a financial RAG pipeline over real financial documents, compares no-retrieval, standard RAG, and reranked RAG methods, and evaluates factual grounding and hallucination risk. Member 1 delivered the reproducible data and retrieval foundation. Member 2 delivered the RAG and reranking architecture. Member 3 delivered the evaluation framework, result analysis, dashboard, and final report.

The most important conclusion is that retrieval matters: RAG methods produce cited and substantially more faithful answers than the no-retrieval baseline. Reranking is beneficial when the reranker is strong enough and when the candidate pool contains the right evidence. The clearest evidence is the FinDER neural reranker result, where MS MARCO cross-encoder reranking improves Recall@1 from 0.030 to 0.100 and MRR from 0.047 to 0.128.

## References

- SEC EDGAR data access: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- FinanceBench paper: https://arxiv.org/abs/2311.11944
- FinanceBench dataset: https://huggingface.co/datasets/PatronusAI/financebench
- FinDER paper: https://arxiv.org/abs/2504.15800
- FinDER dataset: https://huggingface.co/datasets/Linq-AI-Research/FinDER
- MS MARCO MiniLM cross-encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
