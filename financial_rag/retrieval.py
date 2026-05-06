from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import RagConfig
from .data import load_saved_chunks, save_chunks
from .embeddings import Embedder, get_embedder
from .schemas import Chunk, RetrievalResult
from .text_utils import content_tokens, jaccard, token_f1

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, chunks: Sequence[Chunk], vectors: np.ndarray, metadata: dict | None = None) -> None:
        self.chunks = list(chunks)
        self.vectors = self._normalize(np.asarray(vectors, dtype=np.float32))
        self.metadata = metadata or {}
        self.chunk_tokens = [content_tokens(chunk.text) for chunk in self.chunks]
        self.chunk_token_sets = [set(tokens) for tokens in self.chunk_tokens]
        self.chunk_token_counts = np.asarray([len(tokens) for tokens in self.chunk_tokens], dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def search(self, query: str, embedder: Embedder, k: int = 5) -> List[RetrievalResult]:
        if not self.chunks:
            return []
        query_vec = embedder.encode([query]).astype(np.float32)
        query_vec = self._normalize(query_vec)
        dense_scores = self.vectors @ query_vec[0]
        query_tokens = content_tokens(query)
        query_set = set(query_tokens)
        query_len = max(1, len(query_tokens))
        lexical_scores = np.asarray(
            [
                _token_overlap_score(query_set, query_len, chunk_set, int(chunk_len))
                for chunk_set, chunk_len in zip(self.chunk_token_sets, self.chunk_token_counts)
            ],
            dtype=np.float32,
        )
        scores = (0.25 * dense_scores) + (0.75 * lexical_scores)
        order = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [
            RetrievalResult(chunk=self.chunks[int(idx)], score=float(scores[int(idx)]), rank=rank + 1)
            for rank, idx in enumerate(order)
        ]


def _token_overlap_score(query_set: set[str], query_len: int, chunk_set: set[str], chunk_len: int) -> float:
    if not query_set or not chunk_set:
        return 0.0
    overlap = len(query_set & chunk_set)
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, chunk_len)
    recall = overlap / query_len
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    union = len(query_set | chunk_set)
    jac = overlap / max(1, union)
    return 0.85 * f1 + 0.15 * jac


def build_vector_store(
    config: RagConfig,
    chunks: Sequence[Chunk],
    embedder: Embedder | None = None,
    metadata: dict | None = None,
) -> VectorStore:
    config.ensure_dirs()
    embedder = embedder or get_embedder(config)
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.encode(texts).astype(np.float32)
    store_metadata = {"embedding_backend": config.embedding_backend, "embedder": embedder.name}
    if metadata:
        store_metadata.update(metadata)
    store = VectorStore(chunks, vectors, metadata=store_metadata)
    save_chunks(config, chunks)
    np.savez_compressed(
        config.vectors_path,
        vectors=store.vectors,
        metadata=json.dumps(store.metadata),
    )
    _try_write_faiss(config.faiss_path, store.vectors)
    logger.info("Saved vector store with %d chunks to %s", len(chunks), config.vectors_path)
    return store


def _try_write_faiss(path: Path, vectors: np.ndarray) -> None:
    try:
        warnings.filterwarnings(
            "ignore",
            message="builtin type .* has no __module__ attribute",
            category=DeprecationWarning,
        )
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype(np.float32))
        faiss.write_index(index, str(path))
    except Exception as exc:  # pragma: no cover - optional backend
        logger.warning("Could not write FAISS index: %s", exc)


def load_vector_store(config: RagConfig) -> VectorStore:
    chunks = load_saved_chunks(config)
    payload = np.load(config.vectors_path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    return VectorStore(chunks, payload["vectors"], metadata=metadata)


def build_or_load_vector_store(config: RagConfig, chunks: Sequence[Chunk] | None = None) -> tuple[Embedder, VectorStore]:
    embedder = get_embedder(config)
    if config.vectors_path.exists() and config.chunks_path.exists() and chunks is None:
        return embedder, load_vector_store(config)
    if chunks is None:
        chunks = load_saved_chunks(config)
    return embedder, build_vector_store(config, chunks, embedder)
