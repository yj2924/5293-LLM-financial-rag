"""Mem2 internal dev evaluation.

SCOPE: PLUMBING SANITY ONLY. This script is NOT mem3's hallucination /
faithfulness evaluation, and the numbers it prints MUST NOT be reported
as final retrieval quality.

What it checks:
    1. All configured pipelines run end-to-end on corpus chunks.
    2. Whether the retriever surfaces chunks from the expected company
       (`top1_ticker_match`, `top5_ticker_match`).
    3. Whether one retrieved chunk satisfies a multi-term evidence signature
       (`top1_evidence_hit`, `top5_evidence_hit`).
    4. Refusal behavior on out-of-domain / out-of-corpus questions.

The dev set is intentionally harder than the first smoke-test set:
    * Some questions omit the company name and ask the retriever to identify
      the issuer from product, segment, or accounting evidence.
    * Some questions contain generic terms shared by many 10-Ks, creating
      realistic distractors.
    * Evidence checks require multiple terms in the SAME chunk, not just any
      one keyword somewhere in top-k.

Outputs: results/mem2_runs/dev/{pipeline}.jsonl + dev_summary.json.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from mem2.factory import build_pipelines
from mem2.runner import run_all_modes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand-curated dev questions covering all 10 corpus companies.
#
# Per-question fields:
#   gold_ticker     the company whose 10-K should answer it (None if OOC/OOD)
#   difficulty      coarse bucket for debugging retrieval failures
#   evidence_terms  list of requirement groups. A chunk is an evidence hit only
#                   if it satisfies ALL groups. Each group is OR logic, e.g.
#                   ["13%", "13 percent"] means either string is acceptable.
#   out_of_domain   True if no SEC filing should answer it
#   out_of_corpus   True if a filing could answer it but is outside our corpus
# ---------------------------------------------------------------------------

DEV_SET: List[Dict[str, Any]] = [
    # Straightforward anchors: still useful for catching broken plumbing.
    {
        "question": "In Apple's product table, what are the Services net sales and Services gross margin percentage?",
        "gold_ticker": "AAPL",
        "difficulty": "anchor",
        "evidence_terms": [["Services"], ["85,200"], ["70.8"]],
    },
    {
        "question": "Which company repurchased $76.6 billion of common stock and paid $15.0 billion of dividends?",
        "gold_ticker": "AAPL",
        "difficulty": "anchor",
        "evidence_terms": [["repurchased"], ["76.6"], ["paid dividends"], ["15.0"]],
    },
    {
        "question": "What did Microsoft disclose about Microsoft Cloud revenue and Office 365 Commercial growth?",
        "gold_ticker": "MSFT",
        "difficulty": "anchor",
        "evidence_terms": [["Microsoft Cloud"], ["111.6"], ["Office 365 Commercial"], ["13%"]],
    },
    {
        "question": "Which filing describes Powerwall and Megapack as lithium-ion battery energy storage products?",
        "gold_ticker": "TSLA",
        "difficulty": "anchor",
        "evidence_terms": [["Powerwall"], ["Megapack"], ["lithium-ion"], ["energy storage"]],
    },

    # Company-identification / needle questions.
    {
        "question": "Which issuer organizes its operations into North America, International, and Amazon Web Services segments?",
        "gold_ticker": "AMZN",
        "difficulty": "needle",
        "evidence_terms": [["North America"], ["International"], ["Amazon Web Services"]],
    },
    {
        "question": "Which filing lists third-party seller commissions, fulfillment and shipping fees, AWS sales, advertising services, and Prime membership fees as revenue sources?",
        "gold_ticker": "AMZN",
        "difficulty": "needle",
        "evidence_terms": [["commissions"], ["fulfillment and shipping fees"], ["AWS sales"], ["Prime membership"]],
    },
    {
        "question": "Which issuer reports two segments called Family of Apps and Reality Labs, with revenue mostly from advertising placements?",
        "gold_ticker": "META",
        "difficulty": "needle",
        "evidence_terms": [["Family of Apps"], ["Reality Labs"], ["advertising placements"]],
    },
    {
        "question": "Which filing says Reality Labs expenses are split across augmented reality, virtual reality, and social platforms initiatives?",
        "gold_ticker": "META",
        "difficulty": "needle",
        "evidence_terms": [["Reality Labs"], ["augmented reality"], ["virtual reality"], ["social platforms"]],
    },
    {
        "question": "Which issuer introduced the Hopper architecture and says H100 includes a Transformer Engine for AI transformer models?",
        "gold_ticker": "NVDA",
        "difficulty": "needle",
        "evidence_terms": [["Hopper"], ["H100"], ["Transformer Engine"], ["transformer models"]],
    },
    {
        "question": "Which filing says the H100 is useful for large language models, deep recommender systems, genomics, and complex digital twins?",
        "gold_ticker": "NVDA",
        "difficulty": "needle",
        "evidence_terms": [["H100"], ["large language models"], ["recommender"], ["genomics"]],
    },
    {
        "question": "Which railroad-related holding company describes BNSF Railway as one of the largest railroad systems in North America?",
        "gold_ticker": "BRK-B",
        "difficulty": "needle",
        "evidence_terms": [["BNSF"], ["Railway"], ["largest railroad systems"], ["North America"]],
    },
    {
        "question": "Which insurer-focused conglomerate says underwriting operations include GEICO, Berkshire Hathaway Primary Group, and Berkshire Hathaway Reinsurance Group?",
        "gold_ticker": "BRK-B",
        "difficulty": "needle",
        "evidence_terms": [["GEICO"], ["Primary Group"], ["Reinsurance Group"], ["underwriting"]],
    },

    # Distractor-heavy finance / platform questions.
    {
        "question": "Which bank filing mentions CET1 capital of $219 billion and Standardized and Advanced CET1 ratios of 13.2% and 13.6%?",
        "gold_ticker": "JPM",
        "difficulty": "distractor",
        "evidence_terms": [["CET1"], ["219"], ["13.2"], ["13.6"]],
    },
    {
        "question": "Which bank filing discusses a standardized approach for operational risk, revised market risk requirements, and a capital floor?",
        "gold_ticker": "BAC",
        "difficulty": "distractor",
        "evidence_terms": [["standardized approach"], ["operational risk"], ["market risk"], ["capital floor"]],
    },
    {
        "question": "Which filing warns that traffic acquisition costs, or TAC, and the associated TAC rate may fluctuate and affect margins?",
        "gold_ticker": "GOOGL",
        "difficulty": "distractor",
        "evidence_terms": [["traffic acquisition costs"], ["TAC"], ["margins"]],
    },
    {
        "question": "Which filing says Google advertising includes Search, YouTube ads, and Google Network?",
        "gold_ticker": "GOOGL",
        "difficulty": "distractor",
        "evidence_terms": [["Google advertising"], ["YouTube"], ["Network"]],
    },
    {
        "question": "Which filing puts Azure and other cloud services inside Server products and cloud services?",
        "gold_ticker": "MSFT",
        "difficulty": "distractor",
        "evidence_terms": [["Azure and other cloud services"], ["Server products"], ["cloud services"]],
    },
    {
        "question": "Which filing describes AI infrastructure investments as increasing operating costs and potentially decreasing operating margins?",
        "gold_ticker": "MSFT",
        "difficulty": "distractor",
        "evidence_terms": [["AI infrastructure"], ["operating costs"], ["operating margins"]],
    },
    {
        "question": "Which filing's net sales category table lists iPhone net sales of $200,583 million?",
        "gold_ticker": "AAPL",
        "difficulty": "distractor",
        "evidence_terms": [["iPhone"], ["200,583"], ["Net sales by category"]],
    },
    {
        "question": "Which filing describes Model 3, Model Y, Model S, and Model X as consumer vehicles?",
        "gold_ticker": "TSLA",
        "difficulty": "distractor",
        "evidence_terms": [["Model 3"], ["Model Y"], ["Model S"], ["Model X"]],
    },

    # Out-of-corpus / out-of-domain sanity checks.
    {
        "question": "What is the population of Mars?",
        "gold_ticker": None,
        "difficulty": "negative",
        "out_of_domain": True,
    },
    {
        "question": "Who won the 2024 FIFA World Cup?",
        "gold_ticker": None,
        "difficulty": "negative",
        "out_of_domain": True,
    },
    {
        "question": "What were Coca-Cola's 2023 revenues?",
        "gold_ticker": None,
        "difficulty": "negative",
        "out_of_corpus": True,
    },
    {
        "question": "What did Pfizer report for oncology revenue in its 2023 10-K?",
        "gold_ticker": None,
        "difficulty": "negative",
        "out_of_corpus": True,
    },
]


# ---------------------------------------------------------------------------
# Lightweight summary metrics (NOT a substitute for mem3's faithfulness eval)
# ---------------------------------------------------------------------------


def _ticker_in_top(retrieved: List[Dict[str, Any]], gold: str, k: int) -> bool:
    return any(c.get("ticker") == gold for c in retrieved[:k])


def _chunk_satisfies_evidence(
    chunk: Dict[str, Any],
    requirements: List[List[str]],
) -> bool:
    """True when one chunk satisfies ALL evidence requirement groups."""
    if not requirements:
        return False
    text = (chunk.get("text") or "").lower()
    for group in requirements:
        terms = group if isinstance(group, list) else [group]
        if not any(str(term).lower() in text for term in terms):
            return False
    return True


def _evidence_in_top(
    retrieved: List[Dict[str, Any]],
    requirements: List[List[str]],
    k: int,
) -> bool:
    return any(_chunk_satisfies_evidence(c, requirements) for c in retrieved[:k])


def _normalise_requirements(q_meta: Dict[str, Any]) -> List[List[str]]:
    """Accept new `evidence_terms`; keep old `evidence_keywords` compatible."""
    if q_meta.get("evidence_terms"):
        return q_meta["evidence_terms"]
    if q_meta.get("evidence_keywords"):
        # Old behavior was ANY keyword. Represent that as one OR group.
        return [q_meta["evidence_keywords"]]
    return []


def _refused(answer: str) -> bool:
    return "insufficient context" in answer.lower()


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _append_difficulty_metric(
    agg: Dict[str, Dict[str, List[float]]],
    difficulty: str,
    metric: str,
    value: float,
) -> None:
    if difficulty not in agg:
        agg[difficulty] = defaultdict(list)
    agg[difficulty][metric].append(value)


def _difficulty_summary(agg: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for difficulty, metrics in sorted(agg.items()):
        d: Dict[str, Any] = {"n": len(metrics.get("seen", []))}
        for metric in (
            "top1_ticker_match",
            "top5_ticker_match",
            "top1_evidence_hit",
            "top5_evidence_hit",
        ):
            if metrics.get(metric):
                d[metric] = _mean(metrics[metric])
        out[difficulty] = d
    return out


def summarize(results_by_mode: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for mode, records in results_by_mode.items():
        agg: Dict[str, Any] = defaultdict(list)
        difficulty_agg: Dict[str, Dict[str, List[float]]] = {}

        for q_meta, rec in zip(DEV_SET, records):
            if rec.get("error"):
                agg["errors"].append(rec.get("question"))
                continue

            gold = q_meta.get("gold_ticker")
            difficulty = q_meta.get("difficulty", "unspecified")
            ood = q_meta.get("out_of_domain") or q_meta.get("out_of_corpus")
            requirements = _normalise_requirements(q_meta)

            agg["latency_ms"].append(rec.get("latency_ms", 0))
            _append_difficulty_metric(difficulty_agg, difficulty, "seen", 1.0)

            refused = _refused(rec.get("answer", ""))
            if ood:
                agg["refusal_on_ood"].append(1.0 if refused else 0.0)
            else:
                agg["refusal_on_in_domain"].append(1.0 if refused else 0.0)

            retrieved = rec.get("retrieved_chunks", [])
            if gold and retrieved:
                top1_ticker = 1.0 if _ticker_in_top(retrieved, gold, 1) else 0.0
                top5_ticker = 1.0 if _ticker_in_top(retrieved, gold, 5) else 0.0
                agg["top1_ticker_match"].append(top1_ticker)
                agg["top5_ticker_match"].append(top5_ticker)
                _append_difficulty_metric(
                    difficulty_agg, difficulty, "top1_ticker_match", top1_ticker
                )
                _append_difficulty_metric(
                    difficulty_agg, difficulty, "top5_ticker_match", top5_ticker
                )

                if requirements:
                    top1_evidence = 1.0 if _evidence_in_top(retrieved, requirements, 1) else 0.0
                    top5_evidence = 1.0 if _evidence_in_top(retrieved, requirements, 5) else 0.0
                    agg["top1_evidence_hit"].append(top1_evidence)
                    agg["top5_evidence_hit"].append(top5_evidence)
                    _append_difficulty_metric(
                        difficulty_agg, difficulty, "top1_evidence_hit", top1_evidence
                    )
                    _append_difficulty_metric(
                        difficulty_agg, difficulty, "top5_evidence_hit", top5_evidence
                    )

        out: Dict[str, Any] = {
            "n": len(records),
            "errors": len(agg["errors"]),
            "avg_latency_ms": _mean(agg["latency_ms"]),
        }
        for metric in (
            "top1_ticker_match",
            "top5_ticker_match",
            "top1_evidence_hit",
            "top5_evidence_hit",
            "refusal_on_ood",
            "refusal_on_in_domain",
        ):
            if agg[metric]:
                out[metric] = _mean(agg[metric])
        out["by_difficulty"] = _difficulty_summary(difficulty_agg)
        summary[mode] = out
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_dir = Path("results/mem2_runs/dev")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = [item["question"] for item in DEV_SET]
    pipelines = build_pipelines()

    results = run_all_modes(questions, pipelines, out_dir)

    summary = summarize(results)
    summary_path = out_dir / "dev_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("MEM2 DEV EVAL SUMMARY  (plumbing sanity only -- not for the report)")
    print("=" * 72)
    for mode, m in summary.items():
        print(f"\n[{mode}]")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
    print("\nFull summary -> ", summary_path)
    print("Per-mode JSONL -> ", out_dir)


if __name__ == "__main__":
    main()
