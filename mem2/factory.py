"""Factory: read configs.yaml -> build a dict of named pipelines.

Single entry point so mem3 can do:

    from mem2 import build_pipelines, run_all_modes
    pipelines = build_pipelines()                # default config, all pipelines
    run_all_modes(questions, pipelines, "out/")

Construction is lazy by resource:
    * LLM is loaded only if at least one requested pipeline needs it (always true today).
    * Retriever (FAISS index + embedding model) is loaded only if a non-baseline
      pipeline is requested.
    * Each reranker is loaded only if its specific pipeline is requested.
This means `build_pipelines(modes=["baseline"])` does not touch FAISS,
sentence-transformers, or any cross-encoder. That keeps smoke tests fast
and lets the demo open with the cheapest mode first.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from mem2.llm_client import LLMClient, load_default_llm
from mem2.pipelines import BaselinePipeline, RerankedRAGPipeline, StandardRAGPipeline
from mem2.reranker import Reranker, load_reranker
from mem2.retriever_adapter import Retriever

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rerank_key(name: str) -> str:
    """Reranker name -> pipeline key. e.g. 'ms-marco' -> 'reranked_msmarco'."""
    return f"reranked_{name.replace('-', '')}"


def build_pipelines(
    config_path: Optional[Path] = None,
    llm: Optional[LLMClient] = None,
    retriever: Optional[Retriever] = None,
    modes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Construct the pipeline set described in `configs.yaml`.

    Args:
        config_path: defaults to mem2/configs.yaml.
        llm: pre-built LLM. If omitted, loaded from config.
        retriever: pre-built retriever. If omitted, loaded from config
            (only when a non-baseline pipeline is requested).
        modes: optional subset of pipeline keys to build, e.g.
            ["baseline"] or ["baseline", "reranked_bge"]. None means all.
            Unknown names raise ValueError.

    Returns:
        {pipeline_key: pipeline_instance}
    """
    cfg = _load_yaml(Path(config_path) if config_path else DEFAULT_CONFIG_PATH)

    rerank_names = list(cfg.get("reranker", {}).get("candidates", ["bge", "ms-marco"]))
    available_keys = ["baseline", "standard_rag"] + [_rerank_key(n) for n in rerank_names]

    if modes is None:
        target_keys = available_keys
    else:
        modes = list(modes)
        unknown = [m for m in modes if m not in available_keys]
        if unknown:
            raise ValueError(
                f"Unknown pipeline names: {unknown}. Available: {available_keys}"
            )
        target_keys = [k for k in available_keys if k in modes]
        if not target_keys:
            raise ValueError("modes resolved to empty pipeline set")

    # Resource needs based on target_keys.
    needs_retriever = any(k != "baseline" for k in target_keys)
    needed_rerank_names = [n for n in rerank_names if _rerank_key(n) in target_keys]

    # ----- LLM (always required: every pipeline calls .generate) -----
    if llm is None:
        llm_cfg = cfg.get("llm", {})
        llm = load_default_llm(
            backend=llm_cfg.get("backend", "qwen_local"),
            model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            cache_dir=Path(llm_cfg["cache_dir"]) if llm_cfg.get("cache_dir") else None,
        )

    # ----- Retriever (only if any non-baseline pipeline is requested) -----
    if needs_retriever and retriever is None:
        rcfg = cfg.get("retriever", {})
        retriever = Retriever(
            index_path=Path(rcfg.get("index_path", "results/index.faiss")),
            chunks_path=Path(rcfg.get("chunks_path", "results/chunks.json")),
            embedding_model_name=rcfg.get("embedding_model_name", "all-MiniLM-L6-v2"),
        )

    pcfg = cfg.get("pipeline", {})
    common = {
        "context_budget": pcfg.get("context_budget_tokens", 3000),
        "max_per_source": pcfg.get("max_per_source", 2),
        "max_new_tokens": pcfg.get("max_new_tokens", 512),
        "temperature": pcfg.get("temperature", 0.0),
    }
    retrieve_k_std = pcfg.get("retrieve_k", 5)
    retrieve_k_rr = pcfg.get("rerank_retrieve_k", 20)
    rerank_top_k = pcfg.get("rerank_top_k", 5)

    pipelines: Dict[str, Any] = {}

    if "baseline" in target_keys:
        pipelines["baseline"] = BaselinePipeline(
            llm,
            max_new_tokens=common["max_new_tokens"],
            temperature=common["temperature"],
        )

    if "standard_rag" in target_keys:
        assert retriever is not None  # implied by needs_retriever
        pipelines["standard_rag"] = StandardRAGPipeline(
            llm,
            retriever,
            retrieve_k=retrieve_k_std,
            **common,
        )

    # Build only the rerankers actually in target_keys.
    for name in needed_rerank_names:
        assert retriever is not None  # implied by needs_retriever
        reranker: Reranker = load_reranker(name)
        pipelines[_rerank_key(name)] = RerankedRAGPipeline(
            llm,
            retriever,
            reranker,
            retrieve_k=retrieve_k_rr,
            rerank_top_k=rerank_top_k,
            **common,
        )

    logger.info("Built %d pipeline(s): %s", len(pipelines), list(pipelines))
    return pipelines
