from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9$%.\-]+")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return [tok.lower().strip(".,;:()[]{}") for tok in TOKEN_RE.findall(text or "") if tok.strip(".,;:()[]{}")]


def content_tokens(text: str) -> List[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
    return [tok for tok in tokenize(text) if tok not in stop and len(tok) > 1]


def split_sentences(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return [p.strip() for p in pieces if len(p.strip()) > 10]


def token_f1(prediction: str, gold: str) -> float:
    pred = content_tokens(prediction)
    true = content_tokens(gold)
    if not pred or not true:
        return 0.0
    pred_counts = Counter(pred)
    true_counts = Counter(true)
    common = sum((pred_counts & true_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(true)
    return 2 * precision * recall / (precision + recall)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def chunk_words(words: Sequence[str], size: int, overlap: int) -> List[List[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: List[List[str]] = []
    start = 0
    while start < len(words):
        chunks.append(list(words[start : start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks
