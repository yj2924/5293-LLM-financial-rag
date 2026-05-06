from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .config import DEFAULT_COMPANIES, RagConfig
from .io_utils import read_jsonl, write_jsonl
from .schemas import Chunk, Document
from .text_utils import chunk_words, normalize_space

logger = logging.getLogger(__name__)


SEC_HEADER_RE = re.compile(r"<SEC-HEADER>.*?</SEC-HEADER>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def clean_filing_text(text: str) -> str:
    text = SEC_HEADER_RE.sub(" ", text or "")
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"Table of Contents", " ", text, flags=re.IGNORECASE)
    return normalize_space(text)


def stable_doc_id(*parts: str) -> str:
    raw = "|".join(p for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def load_sample_documents(config: RagConfig) -> List[Document]:
    docs: List[Document] = []
    for row in read_jsonl(config.sample_corpus_path):
        docs.append(
            Document(
                id=row["id"],
                ticker=row.get("ticker", ""),
                cik=row.get("cik", ""),
                company=row.get("company", row.get("ticker", "")),
                form_type=row.get("form_type", "10-K"),
                filing_date=row.get("filing_date", ""),
                accession=row.get("accession", ""),
                source=row.get("source", "sample-edgar-excerpt"),
                source_url=row.get("source_url", ""),
                text=clean_filing_text(row.get("text", "")),
                metadata=row.get("metadata", {}),
            )
        )
    return [doc for doc in docs if len(doc.text) > 80]


def download_edgar_filings(
    config: RagConfig,
    tickers: Sequence[str],
    forms: Sequence[str] = ("10-K",),
    after: Optional[str] = None,
    before: Optional[str] = None,
    limit_per_company: int = 1,
) -> int:
    from sec_edgar_downloader import Downloader

    config.edgar_dir.mkdir(parents=True, exist_ok=True)
    dl = Downloader(config.sec_company_name, config.sec_email, download_folder=config.edgar_dir)
    total = 0
    for ticker in tickers:
        for form in forms:
            logger.info("Downloading %s %s filings from EDGAR", ticker, form)
            total += dl.get(
                form,
                ticker,
                limit=limit_per_company,
                after=after,
                before=before,
                include_amends=False,
                download_details=True,
            )
    return total


def _infer_ticker_from_path(path: Path) -> str:
    parts = path.parts
    if "sec-edgar-filings" in parts:
        idx = parts.index("sec-edgar-filings")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    try:
        return path.parents[2].name
    except IndexError:
        return ""


def load_edgar_documents(config: RagConfig, min_chars: int = 1_000) -> List[Document]:
    docs: List[Document] = []
    for filing_file in config.edgar_dir.glob("**/full-submission.txt"):
        raw = filing_file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_filing_text(raw)
        if len(cleaned) < min_chars:
            continue
        ticker = _infer_ticker_from_path(filing_file)
        form_type = filing_file.parents[1].name if len(filing_file.parents) > 1 else "10-K"
        accession = filing_file.parent.name
        cik = DEFAULT_COMPANIES.get(ticker, "")
        docs.append(
            Document(
                id=f"{ticker}_{accession}" if ticker else stable_doc_id(str(filing_file), raw[:200]),
                ticker=ticker,
                cik=cik,
                company=ticker,
                form_type=form_type,
                filing_date=accession,
                accession=accession,
                source="edgar",
                source_url=f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession.replace('-', '')}/"
                if cik and accession
                else "",
                text=cleaned,
                metadata={"path": str(filing_file), "raw_chars": len(raw), "cleaned_chars": len(cleaned)},
            )
        )
    return docs


def load_benchmark_evidence_documents(config: RagConfig) -> List[Document]:
    """Build a controlled open-book corpus from cached benchmark evidence strings.

    This corpus is useful for FinanceBench/FinDER evaluation because their gold
    evidence often references filings outside the small local EDGAR cache.
    """
    buckets: dict[str, dict] = {}
    for path in (config.financebench_cache_path, config.finder_cache_path):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            evidence = clean_filing_text(row.get("evidence", ""))
            if len(evidence) < 80:
                continue
            dataset = row.get("dataset", path.stem)
            expected_doc = row.get("expected_source_doc") or row.get("id") or stable_doc_id(evidence[:500])
            doc_id = f"{dataset}_{expected_doc}"
            bucket = buckets.setdefault(
                doc_id,
                {
                    "texts": [],
                    "ids": [],
                    "dataset": dataset,
                    "expected_doc": expected_doc,
                    "metadata": row.get("metadata", {}),
                },
            )
            bucket["texts"].append(evidence)
            bucket["ids"].append(row.get("id", ""))
    docs: List[Document] = []
    for doc_id, bucket in buckets.items():
        text = "\n\n".join(bucket["texts"])
        metadata = dict(bucket["metadata"] or {})
        metadata.update(
            {
                "benchmark_dataset": bucket["dataset"],
                "benchmark_ids": ",".join(str(item) for item in bucket["ids"] if item),
                "expected_source_doc": bucket["expected_doc"],
            }
        )
        docs.append(
            Document(
                id=doc_id,
                ticker=str(bucket["expected_doc"]).split("_", 1)[0],
                company=str(metadata.get("company") or bucket["expected_doc"]),
                form_type=str(metadata.get("doc_type") or "benchmark-evidence"),
                filing_date=str(metadata.get("doc_period") or ""),
                source="benchmark-evidence",
                source_url=str(metadata.get("doc_link") or ""),
                text=text,
                metadata=metadata,
            )
        )
    if not docs:
        raise FileNotFoundError(
            "No cached FinanceBench/FinDER evidence found. Run an evaluation once with network access "
            "or create data/benchmarks/financebench.jsonl and data/benchmarks/finder.jsonl."
        )
    return docs


def load_documents(config: RagConfig, corpus: str = "auto") -> List[Document]:
    corpus = corpus.lower()
    if corpus not in {"auto", "sample", "edgar", "benchmark", "benchmarks", "benchmark-evidence"}:
        raise ValueError("corpus must be one of: auto, sample, edgar, benchmark")
    if corpus in {"benchmark", "benchmarks", "benchmark-evidence"}:
        return load_benchmark_evidence_documents(config)
    edgar_docs = load_edgar_documents(config) if corpus in {"auto", "edgar"} else []
    if edgar_docs:
        return edgar_docs
    if corpus == "edgar":
        raise FileNotFoundError(
            f"No EDGAR full-submission.txt files found under {config.edgar_dir}. "
            "Run with --download-edgar or use --corpus sample."
        )
    logger.warning("No local EDGAR filings found; using bundled sample corpus for offline reproducibility.")
    return load_sample_documents(config)


def save_documents(config: RagConfig, documents: Iterable[Document]) -> None:
    write_jsonl(config.documents_path, (doc.to_dict() for doc in documents))


def load_saved_documents(config: RagConfig) -> List[Document]:
    return [Document(**row) for row in read_jsonl(config.documents_path)]


def chunk_documents(config: RagConfig, documents: Sequence[Document]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        words = doc.text.split()
        for idx, word_chunk in enumerate(chunk_words(words, config.chunk_size_words, config.chunk_overlap_words)):
            text = " ".join(word_chunk).strip()
            if len(text) < 80:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc.id}_chunk_{idx}",
                    text=text,
                    source_doc=doc.id,
                    chunk_index=idx,
                    ticker=doc.ticker,
                    cik=doc.cik,
                    company=doc.company,
                    form_type=doc.form_type,
                    filing_date=doc.filing_date,
                    source=doc.source,
                    source_url=doc.source_url,
                    metadata=doc.metadata,
                )
            )
    return chunks


def save_chunks(config: RagConfig, chunks: Iterable[Chunk]) -> None:
    rows = [chunk.to_dict() for chunk in chunks]
    write_jsonl(config.chunks_path, rows)
    legacy_path = config.results_dir / "chunks.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    with legacy_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def load_saved_chunks(config: RagConfig) -> List[Chunk]:
    return [Chunk(**row) for row in read_jsonl(config.chunks_path)]


def write_data_source_summary(config: RagConfig, documents: Sequence[Document], chunks: Sequence[Chunk]) -> None:
    rows = []
    by_doc = {doc.id: doc for doc in documents}
    chunk_counts = {}
    for chunk in chunks:
        chunk_counts[chunk.source_doc] = chunk_counts.get(chunk.source_doc, 0) + 1
    for doc_id, doc in by_doc.items():
        rows.append(
            {
                "document_id": doc_id,
                "ticker": doc.ticker,
                "company": doc.company,
                "form_type": doc.form_type,
                "filing_date": doc.filing_date,
                "source": doc.source,
                "characters": len(doc.text),
                "chunks": chunk_counts.get(doc_id, 0),
                "source_url": doc.source_url,
            }
        )
    config.data_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.data_summary_path, index=False)


def prepare_corpus(config: RagConfig, corpus: str = "auto") -> List[Chunk]:
    config.ensure_dirs()
    documents = load_documents(config, corpus=corpus)
    chunks = chunk_documents(config, documents)
    save_documents(config, documents)
    save_chunks(config, chunks)
    write_data_source_summary(config, documents, chunks)
    logger.info("Prepared %d documents and %d chunks", len(documents), len(chunks))
    return chunks
