from __future__ import annotations

from typing import List, Protocol

from .config import RagConfig
from .schemas import RetrievalResult
from .text_utils import content_tokens, jaccard, token_f1


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        ...


class LexicalReranker:
    name = "lexical-token-overlap"

    def score(self, query: str, text: str) -> float:
        query_tokens = content_tokens(query)
        text_tokens = content_tokens(text)
        return 0.65 * token_f1(text, query) + 0.35 * jaccard(query_tokens, text_tokens)

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        scored = []
        for candidate in candidates:
            rerank_score = self.score(query, candidate.chunk.text)
            scored.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    score=candidate.score,
                    rank=candidate.rank,
                    rerank_score=rerank_score,
                )
            )
        scored.sort(key=lambda item: (item.rerank_score or 0.0, item.score), reverse=True)
        return [
            RetrievalResult(
                chunk=item.chunk,
                score=item.score,
                rank=rank + 1,
                rerank_score=item.rerank_score,
            )
            for rank, item in enumerate(scored[:top_k])
        ]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self.model.predict(pairs)
        scored = []
        for candidate, score in zip(candidates, scores):
            scored.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    score=candidate.score,
                    rank=candidate.rank,
                    rerank_score=float(score),
                )
            )
        scored.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return [
            RetrievalResult(
                chunk=item.chunk,
                score=item.score,
                rank=rank + 1,
                rerank_score=item.rerank_score,
            )
            for rank, item in enumerate(scored[:top_k])
        ]


def get_reranker(config: RagConfig) -> Reranker:
    backend = config.reranker_backend.lower()
    if backend in {"lexical", "offline", "local"}:
        return LexicalReranker()
    if backend in {"cross-encoder", "cross_encoder", "ce"}:
        return CrossEncoderReranker(config.reranker_model)
    raise ValueError(f"Unknown reranker backend: {config.reranker_backend}")
