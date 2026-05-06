"""Reranker abstraction.

Two backends are wired up to answer Research Question 1
("does neural reranking improve relevance?") via direct comparison:

    bge         ->  BAAI/bge-reranker-base
    ms-marco    ->  cross-encoder/ms-marco-MiniLM-L-6-v2

Both are cross-encoders that score (query, passage) pairs. They take the
top-N retrieval candidates from FAISS and re-order them.

Public surface:
    Reranker (Protocol)
    CrossEncoderReranker
    IdentityReranker         -- for ablation: "no rerank" with same call shape
    load_reranker(name)      -- factory
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


RERANKER_REGISTRY: Dict[str, str] = {
    "bge": "BAAI/bge-reranker-base",
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def _tokens(text: str) -> List[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
    return [
        tok.lower().strip(".,;:()[]{}")
        for tok in re.findall(r"[A-Za-z0-9$%.\-]+", text or "")
        if tok.lower().strip(".,;:()[]{}") not in stop
    ]


def _token_f1(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0:
        return 0.0
    precision = common / len(a)
    recall = common / len(b)
    return 2 * precision * recall / (precision + recall)


# Cross-encoder context is 512 tokens. With the query + metadata header taking
# ~30-50 tokens, the passage gets ~450 tokens which corresponds to roughly
# 250 words. Truncating here means the cross-encoder actually sees the start
# of the chunk; without this, a 500-word chunk gets silently sliced and the
# tail is invisible.
RERANK_PASSAGE_MAX_WORDS = 250


def format_chunk_for_rerank(chunk: Dict[str, Any]) -> str:
    """Build the (passage) input the cross-encoder sees during scoring.

    The header lifts company / ticker / section signals into the cross-encoder's
    attention window. SEC 10-K chunks often do not repeat the company name in
    every paragraph, so a chunk about "supply chain risks" can look near-
    identical across companies if the reranker only sees the body text.
    """
    text = chunk.get("text", "") or ""
    words = text.split()
    if len(words) > RERANK_PASSAGE_MAX_WORDS:
        text = " ".join(words[:RERANK_PASSAGE_MAX_WORDS])
    company = chunk.get("company") or ""
    ticker = chunk.get("ticker") or ""
    section = chunk.get("section") or ""
    return (
        f"Company: {company}\n"
        f"Ticker: {ticker}\n"
        f"Section: {section}\n"
        f"Text: {text}"
    )


@runtime_checkable
class Reranker(Protocol):
    model_name: str

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Return the top_k chunks re-ordered by relevance to `query`.
        Each returned chunk has `rerank_score` attached (higher = better)."""
        ...


# ---------------------------------------------------------------------------
# Cross-encoder backend (covers both bge and ms-marco)
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        logger.info("Loading reranker: %s", model_name)
        # max_length keeps long 10-K chunks within the cross-encoder context.
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        if top_k <= 0:
            return []

        # Score on metadata-augmented, truncated passages so the cross-encoder
        # sees company/ticker/section + the first ~250 words of the chunk
        # rather than a silently truncated body-only string.
        pairs = [[query, format_chunk_for_rerank(c)] for c in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored: List[Dict[str, Any]] = []
        for c, s in zip(chunks, scores):
            new = dict(c)  # don't mutate caller's list
            new["rerank_score"] = float(s)
            scored.append(new)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Identity (no-op) reranker
# ---------------------------------------------------------------------------


class IdentityReranker:
    """Returns top_k of the input order unchanged. Useful for ablation
    so that 'standard RAG' and 'reranked RAG' share the same plumbing."""

    model_name = "identity"

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        out = []
        for c in chunks[:top_k]:
            new = dict(c)
            new.setdefault("rerank_score", new.get("retrieval_score", 0.0))
            out.append(new)
        return out


class LexicalReranker:
    """Offline reranker used for smoke tests and demos.

    It is not a neural cross-encoder, but it exercises the same wide-then-
    narrow reranked RAG path without downloading model weights.
    """

    model_name = "lexical-token-overlap"

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for c in chunks:
            new = dict(c)
            new["rerank_score"] = _token_f1(query, format_chunk_for_rerank(c))
            scored.append(new)
        scored.sort(key=lambda item: (item.get("rerank_score", 0.0), item.get("retrieval_score", 0.0)), reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_reranker(name: str) -> Reranker:
    """name in {'lexical', 'bge', 'ms-marco', 'identity'} or a raw HF cross-encoder ID."""
    if name == "lexical":
        return LexicalReranker()
    if name == "identity":
        return IdentityReranker()
    if name in RERANKER_REGISTRY:
        return CrossEncoderReranker(RERANKER_REGISTRY[name])
    # Allow passing an arbitrary HF id for experimentation.
    if "/" in name:
        return CrossEncoderReranker(name)
    raise ValueError(
        f"Unknown reranker {name!r}. Known names: {list(RERANKER_REGISTRY) + ['lexical', 'identity']}, "
        "or pass a HF cross-encoder model ID like 'org/model'."
    )
