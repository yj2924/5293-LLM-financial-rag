from __future__ import annotations

from typing import List, Protocol

from .config import RagConfig
from .schemas import AnswerResult, RetrievalResult
from .text_utils import content_tokens, split_sentences, token_f1


SYSTEM_PROMPT = """You are a financial QA assistant. Answer only from the supplied evidence.
If the evidence does not support an answer, say that the filings do not provide enough evidence.
Include short source citations using the provided chunk ids."""


class AnswerGenerator(Protocol):
    name: str

    def answer_without_retrieval(self, question: str) -> AnswerResult:
        ...

    def answer_with_context(self, question: str, contexts: List[RetrievalResult], method: str) -> AnswerResult:
        ...


class ExtractiveAnswerGenerator:
    name = "extractive-grounded-generator"

    def answer_without_retrieval(self, question: str) -> AnswerResult:
        answer = (
            "I do not have enough document evidence to answer this financial question reliably. "
            "This baseline intentionally does not use retrieval context."
        )
        return AnswerResult(method="baseline_llm_no_retrieval", question=question, answer=answer)

    def answer_with_context(self, question: str, contexts: List[RetrievalResult], method: str) -> AnswerResult:
        if not contexts:
            return AnswerResult(
                method=method,
                question=question,
                answer="The retrieved filings do not provide enough evidence to answer this question.",
            )
        best_sentence = self._select_sentence(question, contexts)
        citations = [ctx.chunk.id for ctx in contexts[:2]]
        citation_text = " ".join(f"[{citation}]" for citation in citations)
        answer = f"{best_sentence} {citation_text}".strip()
        return AnswerResult(method=method, question=question, answer=answer, contexts=contexts, citations=citations)

    def _select_sentence(self, question: str, contexts: List[RetrievalResult]) -> str:
        best = ""
        best_score = -1.0
        # Use the highest-ranked evidence as the answer anchor; lower-ranked chunks
        # remain available as citations but should not override the selected source.
        for context_position, ctx in enumerate(contexts[:1]):
            chunk_match = token_f1(ctx.chunk.text, question)
            rank_bonus = 0.03 / (context_position + 1)
            for sentence in split_sentences(ctx.chunk.text):
                score = token_f1(sentence, question) + 0.2 * chunk_match + rank_bonus
                if _contains_numeric_answer_signal(question, sentence):
                    score += 0.1
                if _question_likely_numeric(question) and not any(ch.isdigit() for ch in sentence):
                    score -= 0.25
                if score > best_score:
                    best = sentence
                    best_score = score
        if best:
            return best
        return contexts[0].chunk.text[:350]


def _contains_numeric_answer_signal(question: str, sentence: str) -> bool:
    q = set(content_tokens(question))
    s = set(content_tokens(sentence))
    has_number = any(ch.isdigit() for ch in sentence)
    return has_number and bool(q & s)


def _question_likely_numeric(question: str) -> bool:
    tokens = set(content_tokens(question))
    numeric_cues = {
        "amount",
        "income",
        "net",
        "revenue",
        "sales",
        "percent",
        "percentage",
        "total",
        "billion",
        "million",
        "was",
        "were",
    }
    return bool(tokens & numeric_cues)


def get_generator(config: RagConfig) -> AnswerGenerator:
    backend = config.generator_backend.lower()
    if backend in {"extractive", "local", "offline"}:
        return ExtractiveAnswerGenerator()
    raise ValueError(
        f"Unknown generator backend {config.generator_backend}. "
        "Use FINRAG_GENERATOR_BACKEND=extractive for offline runs."
    )
