from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    id: str
    text: str
    ticker: str = ""
    cik: str = ""
    company: str = ""
    form_type: str = "10-K"
    filing_date: str = ""
    accession: str = ""
    source: str = "edgar"
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    id: str
    text: str
    source_doc: str
    chunk_index: int
    ticker: str = ""
    cik: str = ""
    company: str = ""
    form_type: str = ""
    filing_date: str = ""
    source: str = ""
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["chunk"] = self.chunk.to_dict()
        return data


@dataclass
class QAExample:
    id: str
    question: str
    answer: str = ""
    evidence: str = ""
    expected_source_doc: str = ""
    dataset: str = "sample"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    method: str
    question: str
    answer: str
    contexts: List[RetrievalResult] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "contexts": [ctx.to_dict() for ctx in self.contexts],
        }
