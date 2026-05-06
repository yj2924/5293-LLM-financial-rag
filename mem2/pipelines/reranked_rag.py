"""Reranked RAG pipeline: dense retrieve (wide) -> cross-encoder rerank -> generate.

Wider initial recall (`retrieve_k`, default 20) gives the cross-encoder enough
candidates to discriminate; the top `rerank_top_k` (default 5) feed the LLM.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from mem2.company_filter import filter_candidates_by_query
from mem2.llm_client import LLMClient
from mem2.prompts import build_rag_prompt, deduplicate_chunks, truncate_chunks_to_budget
from mem2.reranker import Reranker
from mem2.retriever_adapter import Retriever


class RerankedRAGPipeline:
    mode = "reranked_rag"

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        reranker: Reranker,
        retrieve_k: int = 20,
        rerank_top_k: int = 5,
        context_budget: int = 3000,
        max_per_source: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.rerank_top_k = rerank_top_k
        self.context_budget = context_budget
        self.max_per_source = max_per_source
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def answer(self, question: str) -> Dict[str, Any]:
        # 1. Wide retrieval
        t_r = time.time()
        candidates = self.retriever.search(question, k=self.retrieve_k)
        retrieve_ms = int((time.time() - t_r) * 1000)

        # 2a. Pre-rerank company filter: if the question unambiguously names
        # one company, restrict candidates to chunks from that ticker before
        # reranking. Avoids the cross-encoder promoting "semantically similar
        # but wrong company" chunks (a known failure mode on long SEC bodies).
        # If 0 or 2+ companies are mentioned, no filter is applied.
        n_pre_filter = len(candidates)
        candidates, detected_tickers = filter_candidates_by_query(question, candidates)
        n_after_filter = len(candidates)

        # 2b. Rerank to top-k
        t_rr = time.time()
        ranked = self.reranker.rerank(question, candidates, top_k=self.rerank_top_k)
        rerank_ms = int((time.time() - t_rr) * 1000)

        # 3. Per-source de-dup + token-budget truncation
        ranked = deduplicate_chunks(ranked, max_per_source=self.max_per_source)
        ranked = truncate_chunks_to_budget(
            ranked, self.llm.count_tokens, budget=self.context_budget
        )

        # 4. Generate
        prompt = build_rag_prompt(question, ranked)
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
            "retrieved_chunks": ranked,
            "prompt": prompt,
            "latency_ms": retrieve_ms + rerank_ms + generate_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "generate_ms": generate_ms,
            "llm_model": self.llm.model_name,
            "reranker_model": self.reranker.model_name,
            "config": {
                "retrieve_k": self.retrieve_k,
                "rerank_top_k": self.rerank_top_k,
                "context_budget": self.context_budget,
                "max_per_source": self.max_per_source,
                "company_filter": {
                    "detected_tickers": detected_tickers,
                    "n_candidates_before": n_pre_filter,
                    "n_candidates_after": n_after_filter,
                    "filter_applied": n_after_filter < n_pre_filter,
                },
            },
        }
