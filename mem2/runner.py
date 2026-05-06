"""Batch runner.

Takes a list of questions and one or more pipelines, runs each pipeline on
each question, and writes JSONL files that mem3's evaluation script can
consume directly. One JSONL per pipeline mode.

Output schema (one JSON object per line):
    {
        "question": str,
        "pipeline_name": str,       # baseline / standard_rag / reranked_bge / reranked_msmarco
        "mode":     str,            # semantic class: baseline / standard_rag / reranked_rag
        "answer":   str,
        "retrieved_chunks": [chunk_dict, ...],
        "prompt":   str,
        "latency_ms": int,
        "retrieve_ms": int,
        "rerank_ms": int,
        "generate_ms": int,
        "llm_model": str,
        "reranker_model": str | None,
        "config":   {...},
    }

`pipeline_name` is the unique key from `build_pipelines()` so mem3 can
distinguish bge vs ms-marco even when both share `mode == "reranked_rag"`.
Aggregate by `pipeline_name` for ablations; aggregate by `mode` for the
top-level retrieval-vs-rerank comparison.

If a pipeline raises, the record is still written with an `error` field set,
so mem3 can see partial failures without losing the rest of the batch.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


def run_pipeline(
    pipeline: Any,
    questions: Sequence[str],
    output_path: Path,
    pipeline_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Run one pipeline over `questions`, writing JSONL incrementally.

    `pipeline_name` is stamped into every record (defaults to `pipeline.mode`).
    Caller (typically `run_all_modes`) should pass the dict key so that
    siblings sharing the same `mode` (e.g. two reranked_rag pipelines) stay
    distinguishable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    name = pipeline_name or getattr(pipeline, "mode", "unknown")

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            logger.info("[%s] %d/%d: %s", name, i, len(questions), q[:80])
            try:
                rec = pipeline.answer(q)
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("Pipeline %s failed on Q%d", name, i)
                rec = {
                    "question": q,
                    "mode": getattr(pipeline, "mode", "unknown"),
                    "error": f"{type(e).__name__}: {e}",
                }
            # Stamp the unique pipeline name so two pipelines with the same
            # `mode` (e.g. reranked_bge vs reranked_msmarco) stay distinct.
            rec.setdefault("pipeline_name", name)
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # crash-safe: partial runs are still readable

    elapsed = time.time() - t_start
    logger.info(
        "[%s] wrote %d records to %s (%.1fs)",
        name,
        len(results),
        output_path,
        elapsed,
    )
    return results


def run_all_modes(
    questions: Sequence[str],
    pipelines: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run every pipeline in `pipelines` over `questions`. Returns a dict
    {pipeline_name: list_of_records}; also writes one JSONL per pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, pipe in pipelines.items():
        path = output_dir / f"{name}.jsonl"
        out[name] = run_pipeline(pipe, questions, path, pipeline_name=name)
    return out


# ---------------------------------------------------------------------------
# CLI: python -m mem2.runner --questions path/to/questions.txt --out results/mem2_runs
# ---------------------------------------------------------------------------


def _read_questions(path: Path) -> List[str]:
    """Accept .txt (one question per line) or .json/.jsonl with `question` field."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.append(json.loads(line)["question"])
        return out
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item if isinstance(item, str) else item["question"] for item in data]
        raise ValueError(f"{path}: expected a JSON list")
    # default: plain text, one question per line
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    import argparse

    from mem2.factory import build_pipelines

    parser = argparse.ArgumentParser(description="Run mem2 pipelines over a batch of questions.")
    parser.add_argument("--questions", required=True, type=Path, help="txt/json/jsonl of questions")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--config", type=Path, default=None, help="path to configs.yaml")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="subset of pipeline names to run (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    questions = _read_questions(args.questions)
    logger.info("Loaded %d questions from %s", len(questions), args.questions)

    # Push `--modes` into the factory so we DON'T spin up the retriever, the
    # cross-encoders, or the LLM weights for pipelines we won't run.
    pipelines = build_pipelines(config_path=args.config, modes=args.modes)

    run_all_modes(questions, pipelines, args.out)


if __name__ == "__main__":
    main()
