"""mem2: RAG, Reranking, and System Architecture.

Public surface for downstream consumers (mem3 evaluation):
    from mem2 import build_pipelines, run_pipeline, run_all_modes
"""
from mem2.runner import run_pipeline, run_all_modes
from mem2.factory import build_pipelines

__all__ = ["build_pipelines", "run_pipeline", "run_all_modes"]
