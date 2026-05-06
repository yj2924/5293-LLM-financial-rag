"""Standard RAG pipeline: dense retrieve -> generate.

No reranker. Used as the comparison point for "does reranking help?" (RQ1).
"""
from __future__ import annotations

import time
from typing import Any, Dict

from mem2.llm_client import LLMClient
from mem2.prompts import build_rag_prompt, deduplicate_chunks, truncate_chunks_to_budget
from mem2.retriever_adapter import Retriever


class StandardRAGPipeline:
    mode = "standard_rag"

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        retrieve_k: int = 5,
        context_budget: int = 3000,
        max_per_source: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.retrieve_k = retrieve_k
        self.context_budget = context_budget
        self.max_per_source = max_per_source
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def answer(self, question: str) -> Dict[str, Any]:
        # 1. Retrieve
        t_r = time.time()
        chunks = self.retriever.search(question, k=self.retrieve_k)
        retrieve_ms = int((time.time() - t_r) * 1000)

        # 2. Per-source de-dup + token-budget truncation
        chunks = deduplicate_chunks(chunks, max_per_source=self.max_per_source)
        chunks = truncate_chunks_to_budget(
            chunks, self.llm.count_tokens, budget=self.context_budget
        )

        # 3. Generate
        prompt = build_rag_prompt(question, chunks)
        t_g = time.time()
        ans = self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        generate_ms = int((time.time() - t_g) * 1000)

        return {
            "question": question,
            "mode": self.mode,
            "answer": ans,
            "retrieved_chunks": chunks,
            "prompt": prompt,
            "latency_ms": retrieve_ms + generate_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": 0,
            "generate_ms": generate_ms,
            "llm_model": self.llm.model_name,
            "reranker_model": None,
            "config": {
                "retrieve_k": self.retrieve_k,
                "context_budget": self.context_budget,
                "max_per_source": self.max_per_source,
            },
        }
