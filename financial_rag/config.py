from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_EDGAR_DIR = PROJECT_ROOT / "sec_filings" / "sec-edgar-filings"

DEFAULT_COMPANIES: Dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "CBOE": "0001374310",
}


@dataclass(frozen=True)
class RagConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    edgar_dir: Path = LOCAL_EDGAR_DIR if LOCAL_EDGAR_DIR.exists() else Path.home() / "sec-edgar-filings"
    sample_corpus_path: Path = PROJECT_ROOT / "data" / "sample_corpus" / "sample_filings.jsonl"
    sample_benchmark_path: Path = PROJECT_ROOT / "data" / "sample_benchmarks" / "sample_financial_qa.jsonl"
    financebench_cache_path: Path = PROJECT_ROOT / "data" / "benchmarks" / "financebench.jsonl"
    finder_cache_path: Path = PROJECT_ROOT / "data" / "benchmarks" / "finder.jsonl"
    documents_path: Path = PROJECT_ROOT / "results" / "documents.jsonl"
    chunks_path: Path = PROJECT_ROOT / "results" / "chunks.jsonl"
    vectors_path: Path = PROJECT_ROOT / "results" / "vectors.npz"
    faiss_path: Path = PROJECT_ROOT / "results" / "index.faiss"
    metrics_path: Path = PROJECT_ROOT / "results" / "evaluation_metrics.json"
    run_metadata_path: Path = PROJECT_ROOT / "results" / "evaluation_run.json"
    records_path: Path = PROJECT_ROOT / "results" / "evaluation_records.jsonl"
    summary_csv_path: Path = PROJECT_ROOT / "results" / "evaluation_summary.csv"
    data_summary_path: Path = PROJECT_ROOT / "results" / "data_source_summary.csv"
    chunk_size_words: int = 360
    chunk_overlap_words: int = 60
    embedding_backend: str = "hashing"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    vector_backend: str = "numpy"
    reranker_backend: str = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_backend: str = "extractive"
    top_k: int = 5
    candidate_k: int = 20
    random_seed: int = 42
    financebench_dataset: str = "PatronusAI/financebench"
    finder_dataset: str = "Linq-AI-Research/FinDER"
    sec_company_name: str = "Columbia STAT GR5293 financial RAG course project"
    sec_email: str = "yj2924@columbia.edu"

    @classmethod
    def from_env(cls) -> "RagConfig":
        root = Path(os.getenv("FINRAG_PROJECT_ROOT", str(PROJECT_ROOT))).resolve()
        return cls(
            project_root=root,
            data_dir=Path(os.getenv("FINRAG_DATA_DIR", str(root / "data"))),
            results_dir=Path(os.getenv("FINRAG_RESULTS_DIR", str(root / "results"))),
            edgar_dir=Path(
                os.getenv(
                    "FINRAG_EDGAR_DIR",
                    str(LOCAL_EDGAR_DIR if LOCAL_EDGAR_DIR.exists() else Path.home() / "sec-edgar-filings"),
                )
            ),
            sample_corpus_path=Path(
                os.getenv("FINRAG_SAMPLE_CORPUS", str(root / "data" / "sample_corpus" / "sample_filings.jsonl"))
            ),
            sample_benchmark_path=Path(
                os.getenv(
                    "FINRAG_SAMPLE_BENCHMARK",
                    str(root / "data" / "sample_benchmarks" / "sample_financial_qa.jsonl"),
                )
            ),
            financebench_cache_path=Path(
                os.getenv("FINRAG_FINANCEBENCH_CACHE", str(root / "data" / "benchmarks" / "financebench.jsonl"))
            ),
            finder_cache_path=Path(
                os.getenv("FINRAG_FINDER_CACHE", str(root / "data" / "benchmarks" / "finder.jsonl"))
            ),
            documents_path=Path(os.getenv("FINRAG_DOCUMENTS_PATH", str(root / "results" / "documents.jsonl"))),
            chunks_path=Path(os.getenv("FINRAG_CHUNKS_PATH", str(root / "results" / "chunks.jsonl"))),
            vectors_path=Path(os.getenv("FINRAG_VECTORS_PATH", str(root / "results" / "vectors.npz"))),
            faiss_path=Path(os.getenv("FINRAG_FAISS_PATH", str(root / "results" / "index.faiss"))),
            metrics_path=Path(os.getenv("FINRAG_METRICS_PATH", str(root / "results" / "evaluation_metrics.json"))),
            run_metadata_path=Path(
                os.getenv("FINRAG_RUN_METADATA_PATH", str(root / "results" / "evaluation_run.json"))
            ),
            records_path=Path(os.getenv("FINRAG_RECORDS_PATH", str(root / "results" / "evaluation_records.jsonl"))),
            summary_csv_path=Path(os.getenv("FINRAG_SUMMARY_CSV", str(root / "results" / "evaluation_summary.csv"))),
            data_summary_path=Path(
                os.getenv("FINRAG_DATA_SUMMARY", str(root / "results" / "data_source_summary.csv"))
            ),
            chunk_size_words=int(os.getenv("FINRAG_CHUNK_SIZE_WORDS", "360")),
            chunk_overlap_words=int(os.getenv("FINRAG_CHUNK_OVERLAP_WORDS", "60")),
            embedding_backend=os.getenv("FINRAG_EMBEDDING_BACKEND", "hashing"),
            embedding_model=os.getenv("FINRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_dim=int(os.getenv("FINRAG_EMBEDDING_DIM", "384")),
            vector_backend=os.getenv("FINRAG_VECTOR_BACKEND", "numpy"),
            reranker_backend=os.getenv("FINRAG_RERANKER_BACKEND", "lexical"),
            reranker_model=os.getenv("FINRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            generator_backend=os.getenv("FINRAG_GENERATOR_BACKEND", "extractive"),
            top_k=int(os.getenv("FINRAG_TOP_K", "5")),
            candidate_k=int(os.getenv("FINRAG_CANDIDATE_K", "20")),
            random_seed=int(os.getenv("FINRAG_RANDOM_SEED", "42")),
            financebench_dataset=os.getenv("FINRAG_FINANCEBENCH_DATASET", "PatronusAI/financebench"),
            finder_dataset=os.getenv("FINRAG_FINDER_DATASET", "Linq-AI-Research/FinDER"),
            sec_company_name=os.getenv("SEC_COMPANY_NAME", "Columbia STAT GR5293 financial RAG course project"),
            sec_email=os.getenv("SEC_EMAIL", "yj2924@columbia.edu"),
        )

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def retrieval_ks(self) -> Tuple[int, ...]:
        return (1, 3, 5)
