from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .config import RagConfig
from .company_filter import filter_results_by_query
from .generation import AnswerGenerator
from .io_utils import read_jsonl, write_json, write_jsonl
from .reranking import Reranker
from .retrieval import VectorStore
from .schemas import AnswerResult, QAExample, RetrievalResult
from .text_utils import content_tokens, token_f1

logger = logging.getLogger(__name__)


def load_sample_benchmark(config: RagConfig) -> List[QAExample]:
    return [
        QAExample(
            id=row["id"],
            question=row["question"],
            answer=row.get("answer", ""),
            evidence=row.get("evidence", ""),
            expected_source_doc=row.get("expected_source_doc", ""),
            dataset=row.get("dataset", "sample"),
            metadata=row.get("metadata", {}),
        )
        for row in read_jsonl(config.sample_benchmark_path)
    ]


def load_hf_benchmark(config: RagConfig, name: str, split: str = "train", limit: int | None = None) -> List[QAExample]:
    from datasets import load_dataset

    cache_path = config.financebench_cache_path if name == "financebench" else config.finder_cache_path
    if cache_path.exists():
        examples = [QAExample(**row) for row in read_jsonl(cache_path)]
        return examples[:limit] if limit else examples

    dataset_name = config.financebench_dataset if name == "financebench" else config.finder_dataset
    dataset = load_dataset(dataset_name, split=split)
    examples: List[QAExample] = []
    for idx, row in enumerate(dataset):
        examples.append(_normalize_hf_row(row, idx, name))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_path, (asdict(example) for example in examples))
    return examples[:limit] if limit else examples


def _normalize_hf_row(row: dict, idx: int, dataset_name: str) -> QAExample:
    question = str(row.get("question") or row.get("query") or row.get("input") or row.get("text") or "")
    answer = row.get("answer") or row.get("gold_answer") or row.get("output") or row.get("response") or ""
    evidence = _extract_evidence(row)
    expected_source_doc = _extract_expected_source_doc(row)
    return QAExample(
        id=str(row.get("id") or row.get("qid") or row.get("financebench_id") or row.get("_id") or f"{dataset_name}_{idx}"),
        question=question,
        answer=_stringify(answer),
        evidence=evidence,
        expected_source_doc=_stringify(expected_source_doc),
        dataset=dataset_name,
        metadata={
            k: _stringify(v)
            for k, v in row.items()
            if k not in {"question", "text", "answer", "evidence", "references"}
        },
    )


def _extract_evidence(row: dict) -> str:
    for key in ("evidence", "evidence_text", "references", "contexts", "context"):
        if row.get(key):
            return _evidence_to_text(row[key])
    return ""


def _evidence_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("evidence_text", "evidence_text_full_page", "text", "content", "context"):
            if value.get(key):
                return _evidence_to_text(value[key])
        return " ".join(_evidence_to_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return "\n\n".join(part for part in (_evidence_to_text(item) for item in value) if part)
    return str(value)


def _extract_expected_source_doc(row: dict) -> str:
    for key in ("doc_name", "document", "source_doc", "filename", "ticker"):
        if row.get(key):
            return _stringify(row[key])
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and item.get("doc_name"):
                return _stringify(item["doc_name"])
    if isinstance(evidence, dict) and evidence.get("doc_name"):
        return _stringify(evidence["doc_name"])
    return ""


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "evidence_text", "answer", "value", "content"):
            if key in value:
                return _stringify(value[key])
        return " ".join(_stringify(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(v) for v in value)
    return str(value)


def get_benchmark(config: RagConfig, benchmark: str, limit: int | None = None) -> List[QAExample]:
    benchmark = benchmark.lower()
    if benchmark == "sample":
        examples = load_sample_benchmark(config)
        return examples[:limit] if limit else examples
    if benchmark in {"financebench", "finder"}:
        return load_hf_benchmark(config, benchmark, limit=limit)
    raise ValueError("benchmark must be sample, financebench, or finder")


def retrieved_evidence_hit(example: QAExample, results: Sequence[RetrievalResult]) -> bool:
    if not results:
        return False
    if example.expected_source_doc:
        expected = example.expected_source_doc.lower()
        if any(expected in result.chunk.source_doc.lower() or expected in result.chunk.ticker.lower() for result in results):
            return True
    if example.evidence:
        gold_tokens = set(content_tokens(example.evidence))
        if len(gold_tokens) < 4:
            return False
        for result in results:
            chunk_tokens = set(content_tokens(result.chunk.text))
            overlap = len(gold_tokens & chunk_tokens) / max(1, len(gold_tokens))
            if overlap >= 0.35:
                return True
    return False


def reciprocal_rank(example: QAExample, results: Sequence[RetrievalResult]) -> float:
    for rank, result in enumerate(results, start=1):
        if retrieved_evidence_hit(example, [result]):
            return 1.0 / rank
    return 0.0


def faithfulness_score(answer: str, contexts: Sequence[RetrievalResult]) -> float:
    if not contexts:
        return 0.0
    answer_tokens = set(content_tokens(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for ctx in contexts:
        context_tokens.update(content_tokens(ctx.chunk.text))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def citation_rate(answer: AnswerResult) -> float:
    return 1.0 if answer.citations else 0.0


def evaluate(
    config: RagConfig,
    examples: Sequence[QAExample],
    embedder,
    store: VectorStore,
    reranker: Reranker,
    generator: AnswerGenerator,
) -> Dict[str, Dict[str, float]]:
    rows = []
    aggregates: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        baseline = generator.answer_without_retrieval(example.question)
        standard_contexts = store.search(example.question, embedder, k=config.top_k)
        standard_answer = generator.answer_with_context(example.question, standard_contexts, method="standard_rag")
        candidates = store.search(example.question, embedder, k=max(config.candidate_k, config.top_k))
        candidates, _ = filter_results_by_query(example.question, candidates)
        reranked_contexts = reranker.rerank(example.question, candidates, top_k=config.top_k)
        reranked_answer = generator.answer_with_context(example.question, reranked_contexts, method="reranked_rag")
        for answer in (baseline, standard_answer, reranked_answer):
            row = score_answer(example, answer)
            rows.append(row)
            for metric, value in row["metrics"].items():
                aggregates[answer.method][metric].append(value)
        for method_name, contexts in (("standard_rag", standard_contexts), ("reranked_rag", reranked_contexts)):
            for k in config.retrieval_ks():
                hit = retrieved_evidence_hit(example, contexts[:k])
                aggregates[method_name][f"retrieval_recall_at_{k}"].append(1.0 if hit else 0.0)
            aggregates[method_name]["retrieval_mrr"].append(reciprocal_rank(example, contexts))
    summary = {
        method: {metric: float(np.mean(values)) if values else 0.0 for metric, values in metric_values.items()}
        for method, metric_values in aggregates.items()
    }
    write_json(config.metrics_path, summary)
    write_jsonl(config.records_path, rows)
    write_summary_csv(config.summary_csv_path, summary)
    return summary


def score_answer(example: QAExample, answer: AnswerResult) -> dict:
    faithful = faithfulness_score(answer.answer, answer.contexts)
    return {
        "id": example.id,
        "dataset": example.dataset,
        "question": example.question,
        "gold_answer": example.answer,
        "gold_evidence": example.evidence,
        "expected_source_doc": example.expected_source_doc,
        "method": answer.method,
        "answer": answer.answer,
        "citations": answer.citations,
        "metrics": {
            "answer_token_f1": token_f1(answer.answer, example.answer),
            "faithfulness": faithful,
            "hallucination_proxy_rate": 1.0 - faithful,
            "citation_rate": citation_rate(answer),
        },
    }


def write_summary_csv(path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({metric for metrics in summary.values() for metric in metrics})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *metric_names])
        writer.writeheader()
        for method, metrics in summary.items():
            row = {"method": method}
            row.update({metric: metrics.get(metric, 0.0) for metric in metric_names})
            writer.writerow(row)
