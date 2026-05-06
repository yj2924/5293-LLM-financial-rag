from __future__ import annotations

import hashlib
import logging
from typing import Iterable, List, Protocol

import numpy as np

from .config import RagConfig
from .text_utils import tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _token_vector(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 1) & 1 else -1.0
        return index, sign

    def encode(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            for token in tokens:
                index, sign = self._token_vector(token)
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=len(texts) > 32, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


def get_embedder(config: RagConfig) -> Embedder:
    backend = config.embedding_backend.lower()
    if backend in {"hashing", "local", "offline"}:
        return HashingEmbedder(dim=config.embedding_dim)
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(config.embedding_model)
    raise ValueError(f"Unknown embedding backend: {config.embedding_backend}")


def encode_texts(embedder: Embedder, texts: Iterable[str]) -> np.ndarray:
    return embedder.encode(list(texts)).astype(np.float32)
