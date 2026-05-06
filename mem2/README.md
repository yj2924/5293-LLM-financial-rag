# mem2 - RAG, Reranking, and System Architecture

Owner: Member 2.
Depends on: mem1's persisted FAISS index (`results/index.faiss`) and chunks
(`results/chunks.json`).
Consumed by: mem3 (faithfulness / hallucination evaluation).

## What this delivers

Four pipelines, all exposing the same `pipeline.answer(question) -> dict`:

| Pipeline name      | Mode             | Retrieval  | Reranker                                |
|--------------------|------------------|------------|-----------------------------------------|
| `baseline`         | `baseline`       | none       | n/a                                     |
| `standard_rag`     | `standard_rag`   | FAISS k=5  | none                                    |
| `reranked_bge`     | `reranked_rag`   | FAISS k=20 | `BAAI/bge-reranker-base`                |
| `reranked_msmarco` | `reranked_rag`   | FAISS k=20 | `cross-encoder/ms-marco-MiniLM-L-6-v2`  |

Two rerankers run in parallel so the team can directly answer
**RQ1 (does reranking help?)** with a within-system ablation.

## Module map

```
mem2/
    __init__.py             public surface: build_pipelines, run_pipeline, run_all_modes
    llm_client.py           LLMClient protocol + QwenLocalClient + EchoClient + OpenAICompatibleClient
    prompts.py              system prompts, [Context N] formatting, token budget
    retriever_adapter.py    wraps mem1 FAISS + chunks.json -> search(q, k)
    reranker.py             Reranker protocol + CrossEncoderReranker (bge / ms-marco)
    pipelines/
        __init__.py
        baseline.py
        standard_rag.py
        reranked_rag.py
    factory.py              build_pipelines(modes=...) - lazy by resource
    runner.py               batch runner + CLI
    dev_eval.py             24-question + 4 negative-control internal sanity check
    company_filter.py       ticker detection + pre-rerank candidate filter
    configs.yaml            single source of truth for k / budgets / models
    requirements.txt        deps on top of mem1's
    README.md               this file
```

## Setup

From the repo root:

```bash
pip install -r requirements.txt           # mem1's deps
pip install -r mem2/requirements.txt      # transformers / torch / pyyaml
python main.py                            # builds results/index.faiss + chunks.json
```

The first Qwen 2.5-7B load downloads ~15 GB of weights from HuggingFace and
needs roughly 16 GB of GPU memory in bfloat16. To smoke-test plumbing without
a GPU, set `MEM2_LLM_BACKEND=echo` - pipelines will run with the echo backend
instead of a real LLM. To hit any OpenAI-compatible API instead (real OpenAI,
vLLM, Ollama, LM Studio), point `MEM2_LLM_BACKEND=openai_compat` and set
`MEM2_LLM_BASE_URL` (and `MEM2_LLM_API_KEY` if needed).

## Quick start

```python
from mem2 import build_pipelines, run_all_modes

# All four pipelines:
pipelines = build_pipelines()

# OR: only build the ones you need (saves loading rerankers / index):
pipelines = build_pipelines(modes=["baseline", "standard_rag"])

questions = ["How fast did Microsoft Azure grow in fiscal 2023?"]
run_all_modes(questions, pipelines, "results/mem2_runs/demo")
```

Or from the command line:

```bash
python -m mem2.runner \
    --questions path/to/questions.txt \
    --out results/mem2_runs/eval

# Restrict to a subset (lazy: retriever / reranker not loaded if not needed):
python -m mem2.runner --questions q.txt --out out/ --modes baseline standard_rag
```

To run the 25-question dev sanity check + summary:

```bash
python -m mem2.dev_eval
```

## Output schema (for mem3)

One JSONL per pipeline, one record per question:

```json
{
  "question": "...",
  "pipeline_name": "reranked_bge",
  "mode": "reranked_rag",
  "answer": "Azure revenue grew 30% in fiscal 2023 [Context 1].",
  "retrieved_chunks": [
    {"id": "...", "text": "...", "ticker": "MSFT", "source_doc": "...",
     "filing_date": "...", "corpus_source": "sec_edgar",
     "retrieval_score": 0.812, "rerank_score": 8.71}
  ],
  "prompt": "<full prompt sent to LLM>",
  "latency_ms": 4123,
  "retrieve_ms": 18,
  "rerank_ms": 142,
  "generate_ms": 3963,
  "llm_model": "Qwen/Qwen2.5-7B-Instruct",
  "reranker_model": "BAAI/bge-reranker-base",
  "config": {"retrieve_k": 20, "rerank_top_k": 5, "context_budget": 3000,
             "max_per_source": 2}
}
```

`pipeline_name` is the unique key (`reranked_bge` vs `reranked_msmarco`).
`mode` is the semantic class (both reranked_* share `reranked_rag`).
Aggregate by `pipeline_name` for ablations; aggregate by `mode` for the
top-level retrieval-vs-rerank comparison.

`[Context N]` markers in `answer` map directly to indices in
`retrieved_chunks`, so mem3 can do attribution / citation faithfulness
without fuzzy text matching.

Baseline records also carry `retrieve_ms`, `rerank_ms`, and `config`
(set to 0 / minimal) so every JSONL has the same key set.

## Architectural decisions

- **Protocol-based interfaces.** `LLMClient` and `Reranker` are
  `typing.Protocol`s. Switching from local Qwen to an API client (or from
  bge to a different cross-encoder) is a one-line change in
  `configs.yaml`; pipeline code does not move.
- **Lazy by resource.** `build_pipelines(modes=[...])` only loads what
  the requested pipelines need. `modes=["baseline"]` does NOT touch
  FAISS, sentence-transformers, or any cross-encoder.
- **Disk LLM cache.** `(prompt, max_new_tokens, temperature)` keyed. mem3
  iterating on evaluation prompts re-uses prior generations for free.
- **Wide-then-narrow retrieval for reranked mode.** Dense recall at k=20
  gives the cross-encoder enough candidates; the LLM still sees only top-5
  to keep context focused.
- **Per-source de-duplication (`max_per_source=2`).** A single 10-K
  often has multiple chunks scoring high; capping prevents the context
  from collapsing to one document.
- **3k-token context budget.** Mitigates lost-in-the-middle on Qwen's
  long-context input. Counted against the *formatted* chunk (with metadata
  header), not the raw text.
- **Inline citation `[Context N]`.** Forces grounding into a structured
  form so faithfulness evaluation can be deterministic.

## Known limitations

- Qwen 2.5-7B is the default. For machines without GPU, use
  `MEM2_LLM_BACKEND=echo` (dry-run) or `openai_compat` (any API).
- mem1's corpus is 10 companies x FY2023 10-K. Out-of-corpus questions
  (different year, different company) will trigger the
  "Insufficient context to answer." refusal - this is by design but
  mem3 should restrict its evaluation set to corpus-covered questions.
- `truncate_chunks_to_budget` is greedy and order-preserving; under tight
  budgets it can drop a more relevant late chunk in favor of an earlier
  one. Acceptable because the input is already ranked.
- `dev_eval.py` is a plumbing sanity check ONLY. `top1/top5_ticker_match`
  and `top1/top5_evidence_hit` are coarse proxies and MUST NOT be reported
  as final retrieval quality. mem3's evaluation is the source of truth.
- `retrieval_score` is cosine similarity in [-1, 1] (FAISS IndexFlatIP on
  L2-normalised embeddings). Higher = more similar.
