"""Prompt templates and context formatting for all three modes.

Design notes:
- Baseline (no retrieval) and RAG modes share the analyst persona but RAG modes
  add a strict grounding instruction and a refusal token "Insufficient context".
- Each context block is formatted with explicit `[Context N]` headers so that
  mem3 can match cited spans deterministically when scoring faithfulness.
- `truncate_chunks_to_budget` enforces the 3k-token context cap (lost-in-middle
  mitigation; greedy from highest-ranked chunk down).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_BASELINE = (
    "You are a financial analyst answering questions about SEC 10-K filings. "
    "Answer concisely and factually. If you are not certain, say so explicitly."
)

SYSTEM_RAG = (
    "You are a financial analyst answering questions about SEC 10-K filings.\n"
    "Answer ONLY using the provided context. When you make a factual claim, "
    "cite the supporting passage inline using the bracket form [Context N], "
    "matching the headers in the context block.\n"
    "If the context does not contain the answer, respond exactly with:\n"
    '"Insufficient context to answer."'
)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def format_chunk(idx: int, chunk: Dict[str, Any]) -> str:
    """Render one chunk with metadata header.

    `idx` is 1-indexed so the LLM cites [Context 1] etc.
    """
    src = chunk.get("source_doc") or chunk.get("id") or "unknown"
    ticker = chunk.get("ticker", "?")
    date = chunk.get("filing_date", "?")
    text = chunk.get("text", "")
    return f"[Context {idx}] Source: {src} | Ticker: {ticker} | Filing: {date}\n{text}"


def format_context_block(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "(no retrieved context)"
    return "\n\n".join(format_chunk(i + 1, c) for i, c in enumerate(chunks))


# ---------------------------------------------------------------------------
# Full prompt builders
# ---------------------------------------------------------------------------


def build_baseline_prompt(question: str) -> str:
    return f"{SYSTEM_BASELINE}\n\nQuestion: {question}\n\nAnswer:"


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_text = format_context_block(chunks)
    return (
        f"{SYSTEM_RAG}\n\n"
        f"=== Context ===\n{context_text}\n\n"
        f"=== Question ===\n{question}\n\n"
        f"=== Answer ===\n"
    )


# ---------------------------------------------------------------------------
# Context length budgeting
# ---------------------------------------------------------------------------


def truncate_chunks_to_budget(
    chunks: List[Dict[str, Any]],
    token_counter: Callable[[str], int],
    budget: int = 3000,
) -> List[Dict[str, Any]]:
    """Greedy: keep chunks in order until adding the next would exceed `budget`.

    Counts tokens of the *formatted* chunk (so metadata headers + separators
    are charged). Caller is expected to pass `token_counter = llm.count_tokens`.

    Order is preserved because callers (reranker / retriever) already sorted
    the list by relevance — reordering would defeat their work.
    """
    if not chunks:
        return []

    kept: List[Dict[str, Any]] = []
    used = 0
    for c in chunks:
        formatted = format_chunk(len(kept) + 1, c)
        # +2 for the "\n\n" separator we add between chunks
        cost = token_counter(formatted) + 2
        if kept and used + cost > budget:
            break
        kept.append(c)
        used += cost
    return kept


# ---------------------------------------------------------------------------
# Lightweight de-duplication helper used by RAG pipelines
# ---------------------------------------------------------------------------


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    max_per_source: int = 2,
) -> List[Dict[str, Any]]:
    """Drop near-duplicates: keep at most `max_per_source` chunks per source_doc.

    Preserves order (assumes input is already ranked).
    """
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for c in chunks:
        src = c.get("source_doc") or c.get("id", "?")
        if seen.get(src, 0) >= max_per_source:
            continue
        seen[src] = seen.get(src, 0) + 1
        out.append(c)
    return out
