from __future__ import annotations

import re
from typing import List, Mapping, Sequence, Tuple

from .schemas import RetrievalResult


COMPANY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "AAPL": ("apple", "aapl"),
    "MSFT": ("microsoft", "msft"),
    "AMZN": ("amazon", "amzn"),
    "GOOGL": ("google", "alphabet", "googl"),
    "META": ("meta platforms", "meta", "facebook", "instagram", "whatsapp"),
    "NVDA": ("nvidia", "nvda"),
    "BRK-B": ("berkshire hathaway", "berkshire", "brk-b", "brk.b"),
    "JPM": ("jpmorgan", "jp morgan", "j.p. morgan", "jpmorgan chase"),
    "BAC": ("bank of america",),
    "TSLA": ("tesla", "tsla"),
}


def detect_tickers(query: str) -> List[str]:
    if not query:
        return []
    lowered = query.lower()
    hits: List[Tuple[int, str]] = []
    seen: set[str] = set()
    for ticker, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            match = re.search(rf"\b{re.escape(alias)}\b", lowered)
            if match and ticker not in seen:
                hits.append((match.start(), ticker))
                seen.add(ticker)
                break
    hits.sort(key=lambda item: item[0])
    return [ticker for _, ticker in hits]


def filter_results_by_query(query: str, candidates: Sequence[RetrievalResult]) -> Tuple[List[RetrievalResult], List[str]]:
    detected = detect_tickers(query)
    candidates = list(candidates)
    if len(detected) != 1:
        return candidates, detected
    ticker = detected[0]
    matched = [candidate for candidate in candidates if candidate.chunk.ticker == ticker]
    return (matched, detected) if matched else (candidates, detected)
