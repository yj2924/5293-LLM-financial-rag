

#!/usr/bin/env python3
import os, sys, json, re, logging, hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sec_edgar_downloader import Downloader
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("FinancialRAG")

DATA_DIR = Path.home() / "sec-edgar-filings"
RESULTS_DIR = Path("results")
MIN_TEXT_LENGTH = 200
CHUNK_SIZE = 500
OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = RESULTS_DIR / "index.faiss"
CHUNKS_PATH = RESULTS_DIR / "chunks.json"
K_VALUES = [1, 3, 5]

COMPANIES = {
    "0000320193": "AAPL",
    "0000789019": "MSFT",
    "0001018724": "AMZN",
    "0001652044": "GOOGL",
    "0001326801": "META",
    "0001045810": "NVDA",
    "0001067983": "BRK-B",
    "0000019617": "JPM",
    "0000070858": "BAC",
    "0000093410": "TSLA",
    "0000066740": "MMM",
    "0000796343": "ADBE",
    "0000050863": "INTC",
    "0000200406": "JNJ",
    "0000104169": "WMT",
}
FILING_PERIOD = {"after": "2023-01-01", "before": "2024-01-01"}

_cached_model = None
_cached_index = None
_cached_chunks = None

def ensure_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def clean_raw_text(text: str) -> str:
    if not text or len(text) < MIN_TEXT_LENGTH:
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        # Remove script and style
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Remove XBRL tags
        for tag in soup.find_all(True):
            if tag.name and ("ix:" in tag.name or "xbrli:" in tag.name or "link:" in tag.name):
                tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        # Fallback to simple regex
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_valid_filing(file_path: Path) -> bool:
    try:
        if not file_path.exists():
            return False
        if file_path.stat().st_size < 20 * 1024:  # Minimum 20KB
            return False
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        # Check for SEC document structure
        if "<SEC-DOCUMENT>" not in raw and "<html" not in raw.lower():
            return False
        cleaned = clean_raw_text(raw)
        if len(cleaned) < MIN_TEXT_LENGTH:
            return False
        return True
    except Exception:
        return False

def download_edgar_filings():
    logger.info("Downloading SEC 10-K filings...")
    dl = Downloader(str(DATA_DIR), email_address="yj2924@columbia.edu")
    for cik, ticker in COMPANIES.items():
        logger.info(f"  {ticker} (CIK {cik})...")
        try:
            dl.get("10-K", ticker, after=FILING_PERIOD["after"], before=FILING_PERIOD["before"], download_details=False)
        except Exception as e:
            logger.error(f"  Failed {ticker}: {e}")

def load_and_clean_filings() -> pd.DataFrame:
    logger.info("Loading and cleaning filings...")
    records = []
    skipped_invalid = 0
    skipped_empty = 0
    for ticker_path in DATA_DIR.glob("*"):
        ticker = ticker_path.name
        k10_dir = ticker_path / "10-K"
        if not k10_dir.exists():
            continue
        for filing_dir in k10_dir.glob("*"):
            filing_file = filing_dir / "full-submission.txt"
            if not filing_file.exists():
                continue
            if not is_valid_filing(filing_file):
                skipped_invalid += 1
                continue
            raw = filing_file.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_raw_text(raw)
            if not cleaned:
                skipped_empty += 1
                continue
            cik = next((c for c, t in COMPANIES.items() if t == ticker), ticker)
            doc_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
            records.append({
                "id": f"{ticker}_{filing_dir.name}",
                "ticker": ticker, "cik": cik, "company": ticker,
                "text": cleaned, "raw_length": len(raw),
                "cleaned_length": len(cleaned), "filing_date": filing_dir.name,
                "sha256": doc_hash,
            })
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} valid filings. Skipped {skipped_invalid} invalid, {skipped_empty} empty.")
    return df

def document_data_sources(df: pd.DataFrame):
    logger.info("Documenting data sources...")
    ensure_directories()
    summary = df.groupby("ticker").agg(
        filings=("id", "count"),
        total_chars=("cleaned_length", "sum"),
        avg_chars=("cleaned_length", "mean"),
        max_chars=("cleaned_length", "max"),
        min_chars=("cleaned_length", "min"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "data_source_summary.csv", index=False)
    total_filings = len(df)
    total_chars = df["cleaned_length"].sum()
    with open(RESULTS_DIR / "data_source_stats.txt", "w") as f:
        f.write(f"Total filings: {total_filings}\n")
        f.write(f"Total cleaned characters: {total_chars:,}\n")
        f.write(f"Companies: {', '.join(df['ticker'].unique())}\n")

def chunk_text_by_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def build_chunk_metadata(df: pd.DataFrame) -> List[Dict[str, Any]]:
    logger.info("Building chunks...")
    all_chunks = []
    for _, row in df.iterrows():
        doc_chunks = chunk_text_by_words(row["text"])
        for i, ch in enumerate(doc_chunks):
            if not ch.strip():
                continue
            all_chunks.append({
                "id": f"{row['id']}_chunk_{i}",
                "text": ch,
                "ticker": row["ticker"],
                "company": row["company"],
                "source_doc": row["id"],
                "filing_date": row["filing_date"],
            })
    logger.info(f"Created {len(all_chunks)} chunks from {len(df)} documents.")
    return all_chunks

def generate_embeddings_and_index(chunks: List[Dict[str, Any]]) -> Tuple[SentenceTransformer, faiss.Index, List[Dict[str, Any]]]:
    logger.info("Generating embeddings and building FAISS index...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    ensure_directories()
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"FAISS index saved ({index.ntotal} vectors).")
    return model, index, chunks

def load_index_and_chunks() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {index.ntotal} vectors from index.")
    return index, chunks

def retrieve_top_k(query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    q_vec = model.encode([query]).astype("float32")
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def search(query: str, k: int = 5) -> str:
    global _cached_model, _cached_index, _cached_chunks
    if _cached_model is None or _cached_index is None or _cached_chunks is None:
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            _cached_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            _cached_index, _cached_chunks = load_index_and_chunks()
        else:
            raise RuntimeError("Index not found. Run the full pipeline first.")
    chunks = retrieve_top_k(query, _cached_model, _cached_index, _cached_chunks, k)
    return json.dumps(chunks, indent=2, ensure_ascii=False)

def create_synthetic_testset(chunks: List[Dict[str, Any]], num_questions: int = 30) -> Dataset:
    logger.info("Creating synthetic test set from chunks...")
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(chunks), size=min(num_questions, len(chunks)), replace=False)
    questions = []
    evidence_docs = []
    for idx in sample_indices:
        chunk = chunks[idx]
        question_text = " ".join(chunk["text"].split()[:50])
        questions.append({"question": question_text})
        evidence_docs.append(chunk["source_doc"])
    syn_dataset = Dataset.from_dict({
        "question": [q["question"] for q in questions],
        "evidence": evidence_docs
    })
    logger.info(f"Created synthetic test set with {len(syn_dataset)} questions.")
    return syn_dataset

def evaluate_on_synthetic_dataset(model: SentenceTransformer, index: faiss.Index, chunks: List[Dict[str, Any]], dataset: Dataset) -> Dict[str, float]:
    recalls = {k: [] for k in K_VALUES}
    mrrs = []
    for item in dataset:
        question = item["question"]
        evidence_doc = item["evidence"]
        results = retrieve_top_k(question, model, index, chunks, k=max(K_VALUES))
        retrieved_doc_ids = [r["source_doc"] for r in results]
        for k in K_VALUES:
            hit = evidence_doc in retrieved_doc_ids[:k]
            recalls[k].append(1.0 if hit else 0.0)
        rank_found = False
        for rank, doc_id in enumerate(retrieved_doc_ids, 1):
            if evidence_doc == doc_id:
                mrrs.append(1.0 / rank)
                rank_found = True
                break
        if not rank_found:
            mrrs.append(0.0)
    metrics = {}
    for k in K_VALUES:
        metrics[f"Recall@{k}"] = np.mean(recalls[k])
    metrics["MRR"] = np.mean(mrrs)
    return metrics

def run_full_evaluation(model: SentenceTransformer, index: faiss.Index, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    logger.info("Evaluating retrieval quality (synthetic dataset only)...")
    syn_dataset = create_synthetic_testset(chunks, num_questions=30)
    metrics = evaluate_on_synthetic_dataset(model, index, chunks, syn_dataset)
    ensure_directories()
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def generate_requirements_file():
    packages = [
        "sec-edgar-downloader",
        "sentence-transformers",
        "faiss-cpu",
        "datasets",
        "pandas",
        "numpy",
        "beautifulsoup4",
        "lxml"
    ]
    with open("requirements.txt", "w") as f:
        f.write("\n".join(packages) + "\n")

def generate_readme(metrics: Dict[str, float]):
    ensure_directories()
    metrics_table = "\n".join([f"| {metric} | {value:.4f} |" for metric, value in metrics.items()])
    readme_content = f"""# Financial RAG System
## 1. Overview
Built by the Data, Retrieval, and Reproducibility Lead.
## 2. Evaluation Results (Synthetic)
| Metric | Score |
|--------|-------|
{metrics_table}
## 3. Reproduce
    pip install -r requirements.txt
    python main.py
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

def main():
    global _cached_model, _cached_index, _cached_chunks
    start_time = datetime.now()
    logger.info("Financial RAG Pipeline – START")
    ensure_directories()
    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*/*/full-submission.txt")):
        if INDEX_PATH.exists() and CHUNKS_PATH.exists():
            logger.info("Index found, skipping download.")
        else:
            download_edgar_filings()
    else:
        logger.info("Filings already downloaded – skipping download.")
    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        logger.info("Loading cached index and chunks...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        index, chunks = load_index_and_chunks()
        df = load_and_clean_filings()
    else:
        df = load_and_clean_filings()
        if df.empty:
            logger.error("No valid filings found. Exiting.")
            sys.exit(1)
        document_data_sources(df)
        chunks = build_chunk_metadata(df)
        logger.info(f"Total chunks created: {len(chunks)}")
        model, index, chunks = generate_embeddings_and_index(chunks)
    _cached_model = model
    _cached_index = index
    _cached_chunks = chunks
    metrics = run_full_evaluation(model, index, chunks)
    generate_requirements_file()
    generate_readme(metrics)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Pipeline completed in {elapsed:.1f} seconds.")
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
