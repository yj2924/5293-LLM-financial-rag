"""Adapter around mem1's FAISS retrieval.

Why this exists:
    mem1's `main.py` builds the FAISS index and `chunks.json` but exposes
    `retrieve_top_k(query, model, index, chunks, k)` which forces the caller
    to also load the embedding model and pass three handles around.

    This adapter persists those once on construction and exposes
        retriever.search(query, k=5) -> List[chunk_dict]

    matching the simplified interface mem2 was promised:
        results = search(question, k=5)

The chunk dicts come straight from `chunks.json`, with one added field:
    `retrieval_score`  -- cosine similarity in [-1, 1], higher = more similar.

Index contract (set by mem1's `main.py`):
    * Embeddings are L2-normalised before being added to the FAISS index.
    * The index is `IndexFlatIP` (inner product), so on unit-norm vectors
      the returned "distance" is exactly cosine similarity.
This adapter mirrors the contract by L2-normalising the query vector before
search. Without normalisation the IP score would conflate query magnitude
with relevance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Defaults match mem1's `main.py` constants so we don't drift if mem1 reruns.
DEFAULT_INDEX_PATH = Path("results/index.faiss")
DEFAULT_CHUNKS_PATH = Path("results/chunks.json")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    """Loads a persisted vector index + chunks and answers `search()` calls.

    In the merged repo this first tries the newer `financial_rag` vector store
    (`results/vectors.npz` + `results/chunks.jsonl`), which supports the
    offline hashing embedder used by the reproducible sample pipeline. If that
    store is unavailable, it falls back to the original mem1 FAISS + MiniLM
    contract from the downloaded member-2 repo.
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.embedding_model_name = embedding_model_name
        self._financial_rag_store = None
        self._financial_rag_embedder = None

        if self._try_load_financial_rag_store():
            logger.info("Retriever ready via financial_rag vector store")
            return

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run mem1's main.py first to build the corpus."
            )
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {self.chunks_path}. "
                "Run mem1's main.py first to build the corpus."
            )

        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

        logger.info("Loading chunks from %s", self.chunks_path)
        with self.chunks_path.open("r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        if self.index.ntotal != len(self.chunks):
            logger.warning(
                "Index size (%d) does not match chunks.json size (%d). "
                "Search will skip out-of-range hits.",
                self.index.ntotal,
                len(self.chunks),
            )

        logger.info("Loading embedding model %s", embedding_model_name)
        self.model = SentenceTransformer(embedding_model_name)

        logger.info(
            "Retriever ready: %d vectors, %d chunks", self.index.ntotal, len(self.chunks)
        )

    def _try_load_financial_rag_store(self) -> bool:
        try:
            from financial_rag.config import RagConfig
            from financial_rag.embeddings import get_embedder
            from financial_rag.retrieval import load_vector_store
        except Exception as exc:
            logger.debug("financial_rag store import unavailable: %s", exc)
            return False

        cfg = RagConfig.from_env()
        if not cfg.vectors_path.exists() or not cfg.chunks_path.exists():
            return False
        try:
            self._financial_rag_embedder = get_embedder(cfg)
            self._financial_rag_store = load_vector_store(cfg)
            return True
        except Exception as exc:
            logger.warning("Could not load financial_rag vector store; falling back to legacy FAISS: %s", exc)
            self._financial_rag_embedder = None
            self._financial_rag_store = None
            return False

    # ------------------------------------------------------------------ API

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Top-k chunks for `query`. Each result has `retrieval_score` attached.

        `retrieval_score` is cosine similarity in [-1, 1]; higher = more similar.
        Consistent with the reranker's score field (also higher = better), so
        downstream code never needs to flip signs.
        """
        if k <= 0:
            return []
        if self._financial_rag_store is not None and self._financial_rag_embedder is not None:
            out: List[Dict[str, Any]] = []
            for result in self._financial_rag_store.search(query, self._financial_rag_embedder, k=k):
                chunk = result.chunk.to_dict()
                chunk["retrieval_score"] = float(result.score)
                out.append(chunk)
            return out
        import faiss

        q_vec = self.model.encode([query]).astype("float32")
        # Match the index contract: unit-norm query against unit-norm chunks.
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, k)
        results: List[Dict[str, Any]] = []
        for i, score in zip(indices[0], scores[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            chunk = dict(self.chunks[i])  # shallow copy so caller can mutate
            chunk["retrieval_score"] = float(score)
            results.append(chunk)
        return results


# ---------------------------------------------------------------------------
# Convenience for the simplified interface mem2 was promised
# ---------------------------------------------------------------------------


_DEFAULT_RETRIEVER: Optional[Retriever] = None


def get_default_retriever() -> Retriever:
    """Lazy singleton — first call loads the index and embedding model."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = Retriever()
    return _DEFAULT_RETRIEVER


def search(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """Module-level shortcut: `from mem2.retriever_adapter import search`."""
    return get_default_retriever().search(question, k=k)
