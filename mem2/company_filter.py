"""Detect company mentions in a query and filter candidate chunks accordingly.

Used by the reranked RAG pipeline as a pre-rerank step:

    candidates_after_filter = filter_candidates_by_query(query, candidates_top_n)

Rationale:
    SEC 10-K chunks are long and often look near-identical across companies
    (e.g. "supply chain risk", "interest rate risk", "competition"). The dense
    retriever can correctly surface a candidate from MSFT for a Microsoft
    question, but the cross-encoder may then promote a similar Apple chunk
    above it. If the question explicitly names a company, restricting the
    candidate pool to that ticker before reranking removes that failure mode.

    This deliberately does NOT promote / boost - it only filters, and only
    when the query unambiguously names exactly one company. If zero or two+
    companies are detected, the candidate list is returned unchanged so the
    reranker still has something to do.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Tuple


# Lowercased aliases mapped to the canonical ticker. Keep aliases tight so
# we do not over-fire (e.g. "tier 1" should NOT match anything; "apple pie"
# should NOT match AAPL). Prefer multi-word forms over single words.
COMPANY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "AAPL":  ("apple", "aapl"),
    "MSFT":  ("microsoft", "msft"),
    "AMZN":  ("amazon", "amzn"),
    "GOOGL": ("google", "alphabet", "googl"),
    "META":  ("meta platforms", "meta", "facebook", "instagram", "whatsapp"),
    "NVDA":  ("nvidia", "nvda"),
    "BRK-B": ("berkshire hathaway", "berkshire", "brk-b", "brk.b"),
    "JPM":   ("jpmorgan", "jp morgan", "j.p. morgan"),
    "BAC":   ("bank of america",),
    "TSLA":  ("tesla", "tsla"),
}


def detect_tickers(query: str) -> List[str]:
    """Return the list of tickers whose alias appears in the query.

    Word-boundary regex match (case-insensitive) so that "Meta's" hits
    "meta", "JPMorgan" hits "jpmorgan", and "metadata" / "applet" do NOT
    fire false positives.
    """
    if not query:
        return []
    q = query.lower()
    hits: List[Tuple[int, str]] = []
    seen: set[str] = set()
    for ticker, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            m = re.search(rf"\b{re.escape(alias)}\b", q)
            if m and ticker not in seen:
                hits.append((m.start(), ticker))
                seen.add(ticker)
                break
    hits.sort(key=lambda x: x[0])
    return [t for _, t in hits]


def filter_candidates_by_query(
    query: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Filter candidate chunks to the single ticker mentioned in `query`.

    Returns:
        (filtered, detected_tickers).
        - If exactly ONE ticker is detected AND at least one candidate has
          that ticker, the filtered list contains only those candidates.
        - Otherwise (zero detected, multiple detected, or detected ticker
          absent from candidates), `candidates` is returned unchanged.

    The ambiguity guard matters for comparison questions ("Apple vs Microsoft")
    and for OOD / OOC questions where no ticker is mentioned at all.
    """
    detected = detect_tickers(query)
    if len(detected) != 1:
        return candidates, detected
    ticker = detected[0]
    matched = [c for c in candidates if c.get("ticker") == ticker]
    if not matched:
        # Detected the company by name but no candidate is from that ticker.
        # Keep all candidates so the LLM can at least try, rather than
        # collapsing to an empty context.
        return candidates, detected
    return matched, detected
