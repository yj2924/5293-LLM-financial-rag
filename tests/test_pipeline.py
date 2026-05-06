import tempfile
import unittest
from pathlib import Path

from financial_rag.config import RagConfig
from financial_rag.data import chunk_documents, load_sample_documents
from financial_rag.embeddings import get_embedder
from financial_rag.pipeline import answer_question, build_index, run_evaluation
from financial_rag.retrieval import VectorStore


class PipelineTest(unittest.TestCase):
    def test_sample_retrieval_finds_expected_document(self):
        config = RagConfig.from_env()
        docs = load_sample_documents(config)
        chunks = chunk_documents(config, docs)
        embedder = get_embedder(config)
        vectors = embedder.encode([chunk.text for chunk in chunks])
        store = VectorStore(chunks, vectors)
        results = store.search("What were Apple's 2023 net sales?", embedder, k=3)
        self.assertTrue(any(result.chunk.source_doc == "AAPL_2023_10K" for result in results))

    def test_run_all_on_temp_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = RagConfig.from_env()
            config = RagConfig(
                data_dir=base.data_dir,
                results_dir=Path(tmp),
                sample_corpus_path=base.sample_corpus_path,
                sample_benchmark_path=base.sample_benchmark_path,
                documents_path=Path(tmp) / "documents.jsonl",
                chunks_path=Path(tmp) / "chunks.jsonl",
                vectors_path=Path(tmp) / "vectors.npz",
                faiss_path=Path(tmp) / "index.faiss",
                metrics_path=Path(tmp) / "evaluation_metrics.json",
                records_path=Path(tmp) / "evaluation_records.jsonl",
                summary_csv_path=Path(tmp) / "evaluation_summary.csv",
                data_summary_path=Path(tmp) / "data_source_summary.csv",
            )
            chunks = build_index(config, corpus="sample")
            metrics = run_evaluation(config, benchmark="sample")
            answer = answer_question(config, "What was JPMorgan Chase's 2023 net income?")
            self.assertGreaterEqual(chunks, 5)
            self.assertIn("reranked_rag", metrics)
            self.assertIn("$49.6 billion", answer)


if __name__ == "__main__":
    unittest.main()
