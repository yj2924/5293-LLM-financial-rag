from __future__ import annotations

import argparse
import json
import logging
from pprint import pprint

from .config import RagConfig
from .pipeline import answer_question, build_index, run_evaluation


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Financial EDGAR RAG pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build-index", help="Prepare corpus chunks and vector store")
    build.add_argument("--corpus", choices=["auto", "sample", "edgar", "benchmark"], default="auto")
    build.add_argument("--download-edgar", action="store_true", help="Download 10-K filings before indexing")
    build.add_argument("--limit-per-company", type=int, default=1)
    build.add_argument("--after", default="2023-01-01")
    build.add_argument("--before", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate baseline, standard RAG, and reranked RAG")
    evaluate.add_argument("--benchmark", choices=["sample", "financebench", "finder"], default="sample")
    evaluate.add_argument("--limit", type=int, default=None)

    ask = subparsers.add_parser("ask", help="Ask a question against the built vector store")
    ask.add_argument("question")
    ask.add_argument("--standard", action="store_true", help="Use standard RAG instead of reranked RAG")

    run_all = subparsers.add_parser("run-all", help="Build index and run an evaluation")
    run_all.add_argument("--corpus", choices=["auto", "sample", "edgar", "benchmark"], default="auto")
    run_all.add_argument("--benchmark", choices=["sample", "financebench", "finder"], default="sample")
    run_all.add_argument("--download-edgar", action="store_true")
    run_all.add_argument("--limit-per-company", type=int, default=1)
    run_all.add_argument("--limit", type=int, default=None)
    run_all.add_argument("--after", default="2023-01-01")
    run_all.add_argument("--before", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    config = RagConfig.from_env()

    if args.command in {None, "run-all"}:
        corpus = getattr(args, "corpus", "auto")
        benchmark = getattr(args, "benchmark", "sample")
        download_edgar = getattr(args, "download_edgar", False)
        limit_per_company = getattr(args, "limit_per_company", 1)
        limit = getattr(args, "limit", None)
        after = getattr(args, "after", "2023-01-01")
        before = getattr(args, "before", None)
        chunks = build_index(config, corpus=corpus, download_edgar=download_edgar, limit_per_company=limit_per_company, after=after, before=before)
        metrics = run_evaluation(config, benchmark=benchmark, limit=limit)
        print(f"Built {chunks} chunks.")
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "build-index":
        chunks = build_index(
            config,
            corpus=args.corpus,
            download_edgar=args.download_edgar,
            limit_per_company=args.limit_per_company,
            after=args.after,
            before=args.before,
        )
        print(f"Built {chunks} chunks.")
        return 0

    if args.command == "evaluate":
        metrics = run_evaluation(config, benchmark=args.benchmark, limit=args.limit)
        pprint(metrics)
        return 0

    if args.command == "ask":
        print(answer_question(config, args.question, rerank=not args.standard))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
