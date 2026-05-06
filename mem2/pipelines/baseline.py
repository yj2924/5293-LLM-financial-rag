"""Baseline LLM-only pipeline (no retrieval).

Serves as the upper bound on hallucination: the model has to answer
financial questions purely from its parametric memory.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from mem2.llm_client import LLMClient
from mem2.prompts import build_baseline_prompt


class BaselinePipeline:
    mode = "baseline"

    def __init__(self, llm: LLMClient, max_new_tokens: int = 512, temperature: float = 0.0) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def answer(self, question: str) -> Dict[str, Any]:
        prompt = build_baseline_prompt(question)
        t0 = time.time()
        ans = self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        latency_ms = int((time.time() - t0) * 1000)

        # Schema is intentionally a superset that matches the RAG pipelines:
        # retrieve_ms / rerank_ms are 0 (no retrieval, no rerank) and `config`
        # is present (empty of retrieval knobs). This lets mem3 read every
        # JSONL with the same key set instead of branching on `mode`.
        return {
            "question": question,
            "mode": self.mode,
            "answer": ans,
            "retrieved_chunks": [],
            "prompt": prompt,
            "latency_ms": latency_ms,
            "retrieve_ms": 0,
            "rerank_ms": 0,
            "generate_ms": latency_ms,
            "llm_model": self.llm.model_name,
            "reranker_model": None,
            "config": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        }
