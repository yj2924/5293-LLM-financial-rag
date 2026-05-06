from __future__ import annotations

import logging
from typing import Dict, List

from .config import DEFAULT_COMPANIES, RagConfig
from .company_filter import filter_results_by_query
from .data import download_edgar_filings, prepare_corpus
from .embeddings import get_embedder
from .evaluation import evaluate, get_benchmark
from .generation import get_generator
from .io_utils import write_json
from .reranking import get_reranker
from .retrieval import build_vector_store, load_vector_store
from .schemas import AnswerResult, QAExample

logger = logging.getLogger(__name__)


def build_index(
    config: RagConfig,
    corpus: str = "auto",
    download_edgar: bool = False,
    limit_per_company: int = 1,
    after: str | None = "2023-01-01",
    before: str | None = None,
) -> int:
    config.ensure_dirs()
    if download_edgar:
        download_edgar_filings(
            config,
            tickers=list(DEFAULT_COMPANIES.keys()),
            forms=("10-K",),
            after=after,
            before=before,
            limit_per_company=limit_per_company,
        )
    chunks = prepare_corpus(config, corpus=corpus)
    embedder = get_embedder(config)
    build_vector_store(config, chunks, embedder, metadata={"corpus": corpus, "documents": len({chunk.source_doc for chunk in chunks})})
    return len(chunks)


def run_evaluation(config: RagConfig, benchmark: str = "sample", limit: int | None = None) -> Dict[str, Dict[str, float]]:
    embedder = get_embedder(config)
    store = load_vector_store(config)
    reranker = get_reranker(config)
    generator = get_generator(config)
    examples: List[QAExample] = get_benchmark(config, benchmark=benchmark, limit=limit)
    logger.info("Evaluating %d %s examples", len(examples), benchmark)
    summary = evaluate(config, examples, embedder, store, reranker, generator)
    write_json(
        config.run_metadata_path,
        {
            "corpus": store.metadata.get("corpus", "saved index"),
            "benchmark": benchmark,
            "limit": "all" if limit is None else limit,
            "examples": len(examples),
            "embedding_backend": store.metadata.get("embedding_backend", config.embedding_backend),
            "reranker_backend": config.reranker_backend,
        },
    )
    return summary


def answer_question(config: RagConfig, question: str, rerank: bool = True) -> str:
    result = answer_one(config, question, rerank=rerank)
    return result.answer


def answer_one(config: RagConfig, question: str, rerank: bool = True) -> AnswerResult:
    embedder = get_embedder(config)
    store = load_vector_store(config)
    generator = get_generator(config)
    if rerank:
        reranker = get_reranker(config)
        candidates = store.search(question, embedder, k=max(config.candidate_k, config.top_k))
        candidates, _ = filter_results_by_query(question, candidates)
        contexts = reranker.rerank(question, candidates, top_k=config.top_k)
        return generator.answer_with_context(question, contexts, method="reranked_rag")
    else:
        contexts = store.search(question, embedder, k=config.top_k)
        return generator.answer_with_context(question, contexts, method="standard_rag")


def compare_methods(config: RagConfig, question: str) -> dict[str, AnswerResult]:
    embedder = get_embedder(config)
    store = load_vector_store(config)
    generator = get_generator(config)
    reranker = get_reranker(config)

    baseline = generator.answer_without_retrieval(question)
    standard_contexts = store.search(question, embedder, k=config.top_k)
    standard = generator.answer_with_context(question, standard_contexts, method="standard_rag")
    candidates = store.search(question, embedder, k=max(config.candidate_k, config.top_k))
    candidates, _ = filter_results_by_query(question, candidates)
    reranked_contexts = reranker.rerank(question, candidates, top_k=config.top_k)
    reranked = generator.answer_with_context(question, reranked_contexts, method="reranked_rag")
    return {
        "baseline_llm_no_retrieval": baseline,
        "standard_rag": standard,
        "reranked_rag": reranked,
    }
