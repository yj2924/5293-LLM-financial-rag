from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from financial_rag.config import RagConfig
from financial_rag.io_utils import read_json
from financial_rag.pipeline import build_index, compare_methods, run_evaluation


st.set_page_config(page_title="Financial RAG Evidence Dashboard", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --bg: #f6f8fb;
        --panel: rgba(255, 255, 255, 0.86);
        --sidebar: #f1f5f9;
        --border: #e2e8f0;
        --text: #0f172a;
        --muted: #64748b;
        --accent: #3157d5;
        --accent-strong: #2748b8;
        --accent-soft: #eef2ff;
        --success: #059669;
        --success-soft: #dcfce7;
        --row-win: #f7faff;
        --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        --shadow-soft: 0 10px 26px rgba(15, 23, 42, 0.06);
        --radius: 8px;
    }

    html, body, [class*="css"] {
        font-family: Inter, Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    .stDeployButton,
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
    }

    .block-container {
        max-width: 1440px;
        padding: 2.45rem 4rem 3.2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, var(--sidebar) 100%);
        border-right: 1px solid var(--border);
        box-shadow: 1px 0 0 rgba(255, 255, 255, 0.7) inset;
    }

    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.45rem;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 760;
        letter-spacing: 0;
        margin-bottom: 1.15rem;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #334155;
        font-size: 0.82rem;
        font-weight: 680;
        letter-spacing: 0;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: rgba(255,255,255,0.92) !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        color: var(--text) !important;
    }

    div[data-testid="stTextInput"] input {
        height: 3.35rem !important;
        line-height: 3.35rem !important;
        font-size: 0.98rem;
        font-weight: 560;
        padding: 0 1rem !important;
        display: flex;
        align-items: center;
    }

    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        align-items: center;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: var(--radius);
        height: 3rem;
        font-weight: 720;
        border: 1px solid #cbd5e1;
        background: rgba(255,255,255,0.82);
        color: #334155;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(180deg, #3c63df 0%, var(--accent) 100%);
        color: #ffffff;
        border: 1px solid var(--accent);
        border-radius: var(--radius);
        box-shadow: 0 10px 22px rgba(49, 87, 213, 0.20);
        font-weight: 740;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--border);
        margin: 2rem 0 1.2rem;
    }

    .hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 2rem;
        margin-bottom: 1.35rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.75rem;
        line-height: 1;
        font-weight: 780;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: #111827;
        font-size: clamp(2.15rem, 2.8vw, 3.25rem);
        line-height: 1.02;
        font-weight: 820;
        letter-spacing: 0;
        margin: 0;
    }

    .hero-subtitle {
        max-width: 520px;
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.5;
        margin: 0.15rem 0 0;
        text-align: right;
    }

    .query-shell {
        margin: 0.95rem 0 0.95rem;
    }

    .query-caption {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
        margin-top: 0.25rem;
    }

    .section-title {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 760;
        letter-spacing: 0;
        margin: 1.7rem 0 0.8rem;
    }

    .section-kicker {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 780;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .run-meta {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.34rem 0.62rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255,255,255,0.74);
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 650;
        margin-bottom: 0.2rem;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1.05rem 0 1.65rem;
    }

    .kpi-card {
        position: relative;
        overflow: hidden;
        min-height: 136px;
        padding: 1.08rem 1.12rem;
        border: 1px solid rgba(226, 232, 240, 0.92);
        border-radius: var(--radius);
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.78));
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(12px);
    }

    .kpi-card::after {
        content: "";
        position: absolute;
        inset: 0;
        border-top: 3px solid rgba(49, 87, 213, 0.26);
        pointer-events: none;
    }

    .kpi-label {
        position: relative;
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 720;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        white-space: nowrap;
        margin-bottom: 1rem;
        z-index: 1;
    }

    .kpi-value {
        position: relative;
        color: #0f172a;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0;
        z-index: 1;
    }

    .kpi-badge {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.85rem;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        color: #047857;
        background: var(--success-soft);
        border: 1px solid rgba(5, 150, 105, 0.17);
        font-size: 0.74rem;
        font-weight: 760;
        z-index: 1;
    }

    .trend-arrow {
        font-size: 0.72rem;
    }

    .table-card {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--panel);
        box-shadow: var(--shadow);
        overflow-x: auto;
        margin-top: 0.6rem;
    }

    .rag-table {
        width: 100%;
        min-width: 980px;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 0.92rem;
    }

    .rag-table thead th {
        background: #f8fafc;
        color: #64748b;
        font-size: 0.74rem;
        font-weight: 780;
        letter-spacing: 0.045em;
        text-transform: uppercase;
        padding: 0.95rem 1rem;
        border-bottom: 1px solid var(--border);
        text-align: right;
        white-space: nowrap;
    }

    .rag-table thead th:first-child,
    .rag-table tbody td:first-child {
        width: 25%;
        text-align: left;
    }

    .rag-table tbody tr {
        transition: background 160ms ease;
    }

    .rag-table tbody tr:hover {
        background: #f8fbff;
    }

    .rag-table tbody tr.winner {
        background: var(--row-win);
        box-shadow: inset 3px 0 0 var(--accent);
    }

    .rag-table tbody td {
        padding: 1rem;
        color: #1e293b;
        border-bottom: 1px solid #edf2f7;
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .rag-table tbody tr:last-child td {
        border-bottom: 0;
    }

    .method-cell {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-weight: 760;
        color: #0f172a;
    }

    .method-dot {
        width: 0.58rem;
        height: 0.58rem;
        border-radius: 999px;
        background: #cbd5e1;
        flex: 0 0 auto;
    }

    .winner .method-dot {
        background: var(--accent);
        box-shadow: 0 0 0 4px rgba(49, 87, 213, 0.10);
    }

    .winner-pill {
        display: inline-flex;
        margin-left: 0.45rem;
        padding: 0.12rem 0.45rem;
        border-radius: 999px;
        color: #2748b8;
        background: var(--accent-soft);
        font-size: 0.68rem;
        font-weight: 760;
        vertical-align: middle;
    }

    .note {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        margin: 0.75rem 0 0.2rem;
    }

    .answer-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 0.75rem;
    }

    .answer-card {
        min-height: 168px;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: rgba(255,255,255,0.78);
        padding: 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
    }

    .answer-title {
        color: #0f172a;
        font-size: 0.9rem;
        font-weight: 780;
        margin-bottom: 0.65rem;
    }

    .answer-body {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .citation-line {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
        margin-top: 0.8rem;
    }

    .artifact-list {
        display: grid;
        gap: 0.45rem;
        margin: 0.35rem 0 0.2rem;
    }

    .artifact-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.62rem 0.75rem;
        border: 1px solid #e5edf5;
        border-radius: var(--radius);
        background: rgba(248,250,252,0.72);
        color: #334155;
        font-size: 0.84rem;
    }

    .artifact-status {
        color: var(--success);
        font-size: 0.74rem;
        font-weight: 760;
        white-space: nowrap;
    }

    .artifact-item span:first-child {
        overflow-wrap: anywhere;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: rgba(255,255,255,0.72);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stExpander"] details summary p {
        font-weight: 760;
        color: #0f172a;
    }

    @media (max-width: 1100px) {
        .block-container {
            padding: 2rem 1.5rem 2.5rem;
        }
        .hero {
            display: block;
        }
        .hero-subtitle {
            text-align: left;
            margin-top: 0.75rem;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .answer-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 720px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
        .rag-table {
            table-layout: auto;
            min-width: 860px;
        }
        .table-card {
            overflow-x: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_config() -> RagConfig:
    return RagConfig.from_env()


METHOD_LABELS = {
    "baseline_llm_no_retrieval": "Baseline LLM",
    "standard_rag": "Standard RAG",
    "reranked_rag": "Reranked RAG",
}


METRIC_LABELS = {
    "answer_token_f1": "Answer F1",
    "faithfulness": "Faithfulness",
    "hallucination_proxy_rate": "Hallucination",
    "citation_rate": "Citation",
    "retrieval_recall_at_1": "Recall@1",
    "retrieval_mrr": "MRR",
}


def metric_value(metrics: dict, method: str, metric: str) -> float | None:
    value = metrics.get(method, {}).get(metric)
    return float(value) if value is not None else None


def format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def format_delta(value: float | None) -> str:
    return f"{value:+.3f}" if value is not None else "n/a"


def trend_badge(delta: float | None) -> str:
    if delta is None:
        return ""
    return f'<div class="kpi-badge"><span class="trend-arrow">&uarr;</span>{escape(format_delta(delta))}</div>'


def build_kpi_cards(metrics: dict) -> str:
    standard_recall = metric_value(metrics, "standard_rag", "retrieval_recall_at_1")
    reranked_recall = metric_value(metrics, "reranked_rag", "retrieval_recall_at_1")
    standard_mrr = metric_value(metrics, "standard_rag", "retrieval_mrr")
    reranked_mrr = metric_value(metrics, "reranked_rag", "retrieval_mrr")
    standard_hallucination = metric_value(metrics, "standard_rag", "hallucination_proxy_rate")
    reranked_hallucination = metric_value(metrics, "reranked_rag", "hallucination_proxy_rate")
    hallucination_delta = (
        standard_hallucination - reranked_hallucination
        if standard_hallucination is not None and reranked_hallucination is not None
        else None
    )
    cards = [
        ("Standard Recall@1", standard_recall, standard_recall),
        (
            "Reranked Recall@1",
            reranked_recall,
            reranked_recall - standard_recall if None not in (reranked_recall, standard_recall) else None,
        ),
        (
            "Reranked MRR",
            reranked_mrr,
            reranked_mrr - standard_mrr if None not in (reranked_mrr, standard_mrr) else None,
        ),
        ("Hallucination Reduction", reranked_hallucination, hallucination_delta),
    ]
    html = ['<div class="kpi-grid">']
    for label, value, delta in cards:
        html.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(format_metric(value))}</div>'
            f"{trend_badge(delta)}"
            f"</div>"
        )
    html.append("</div>")
    return "\n".join(html)


def build_comparison_table(metrics: dict) -> str:
    columns = [
        ("Method", None),
        ("Answer F1", "answer_token_f1"),
        ("Faithfulness", "faithfulness"),
        ("Hallucination", "hallucination_proxy_rate"),
        ("Citation", "citation_rate"),
        ("Recall@1", "retrieval_recall_at_1"),
        ("MRR", "retrieval_mrr"),
    ]
    methods = ["baseline_llm_no_retrieval", "standard_rag", "reranked_rag"]
    html = ['<div class="table-card"><table class="rag-table"><thead><tr>']
    for title, _ in columns:
        html.append(f"<th>{escape(title)}</th>")
    html.append("</tr></thead><tbody>")
    for method in methods:
        values = metrics.get(method, {})
        winner = method == "reranked_rag"
        row_class = ' class="winner"' if winner else ""
        method_label = METHOD_LABELS.get(method, method)
        win_pill = '<span class="winner-pill">best</span>' if winner else ""
        html.append(f"<tr{row_class}>")
        html.append(
            f'<td><div class="method-cell">'
            f'<span class="method-dot"></span>'
            f"<span>{escape(method_label)}{win_pill}</span>"
            f"</div></td>"
        )
        for _, metric_name in columns[1:]:
            value = values.get(metric_name)
            html.append(f"<td>{escape(format_metric(float(value)) if value is not None else 'n/a')}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "\n".join(html)


def build_answer_cards(answers: dict) -> str:
    order = [
        ("Baseline LLM", "baseline_llm_no_retrieval"),
        ("Standard RAG", "standard_rag"),
        ("Reranked RAG", "reranked_rag"),
    ]
    html = ['<div class="answer-grid">']
    for label, key in order:
        result = answers[key]
        citations = ", ".join(result.citations) if result.citations else "None"
        html.append(
            f'<div class="answer-card">'
            f'<div class="answer-title">{escape(label)}</div>'
            f'<div class="answer-body">{escape(result.answer)}</div>'
            f'<div class="citation-line">Citations: {escape(citations)}</div>'
            f"</div>"
        )
    html.append("</div>")
    return "\n".join(html)


def build_evidence_summary(answers: dict) -> str:
    sections = []
    for label, key in (("Standard RAG Evidence", "standard_rag"), ("Reranked RAG Evidence", "reranked_rag")):
        result = answers[key]
        rows = []
        for ctx in result.contexts[:3]:
            score = f"{ctx.score:.3f}"
            if ctx.rerank_score is not None:
                score = f"{score} / rerank {ctx.rerank_score:.3f}"
            snippet = ctx.chunk.text[:260].strip()
            rows.append(
                f'<div class="artifact-item">'
                f"<span>{escape(ctx.chunk.ticker)} | {escape(ctx.chunk.source_doc)} | {escape(snippet)}</span>"
                f'<span class="artifact-status">rank {ctx.rank} | {escape(score)}</span>'
                f"</div>"
            )
        content = "\n".join(rows) if rows else '<div class="note">No retrieved chunks.</div>'
        sections.append(
            f'<div class="section-title">{escape(label)}</div>'
            f'<div class="artifact-list">{content}</div>'
        )
    return "\n".join(sections)


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_artifact_list(paths: list[Path], root: Path) -> str:
    rows = []
    for path in paths:
        exists = path.exists()
        status = "Ready" if exists else "Missing"
        rows.append(
            f'<div class="artifact-item">'
            f"<span>{escape(display_path(path, root))}</span>"
            f'<span class="artifact-status">{escape(status)}</span>'
            f"</div>"
        )
    return f'<div class="artifact-list">{"".join(rows)}</div>'


config = get_config()

st.markdown(
    '<div class="hero">'
    '<div><div class="eyebrow">Evidence-Grounded Finance QA</div>'
    '<h1 class="hero-title">Financial RAG Evaluation</h1></div>'
    '<div class="hero-subtitle">'
    "Compare baseline LLM answers, standard retrieval, and reranked retrieval against cited SEC filing evidence."
    "</div></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run Controls")
    corpus = st.selectbox(
        "Corpus",
        ["auto", "edgar", "benchmark", "sample"],
        index=0,
        format_func={"auto": "Auto", "edgar": "EDGAR", "benchmark": "Benchmark Evidence", "sample": "Sample"}.get,
    )
    benchmark = st.selectbox(
        "Benchmark",
        ["financebench", "finder", "sample"],
        index=0,
        format_func={"financebench": "FinanceBench", "finder": "FinDER", "sample": "Sample"}.get,
    )
    limit = st.number_input("Benchmark limit", min_value=1, max_value=500, value=25, step=5)

    if st.button("Build / Refresh Index", use_container_width=True):
        try:
            with st.spinner("Preparing corpus and vector store..."):
                chunks = build_index(config, corpus=corpus)
            st.success(f"Indexed {chunks} chunks")
        except Exception as exc:  # pragma: no cover - surfaced in the UI for demo resilience.
            st.error(f"Index build failed: {exc}")

    if st.button("Run Evaluation", type="primary", use_container_width=True):
        try:
            with st.spinner("Running baseline, standard RAG, and reranked RAG..."):
                metrics = run_evaluation(config, benchmark=benchmark, limit=None if benchmark == "sample" else int(limit))
            st.success("Evaluation complete")
            st.session_state["metrics"] = metrics
            st.session_state["metrics_meta"] = {
                "corpus": corpus,
                "benchmark": benchmark,
                "limit": "all sample questions" if benchmark == "sample" else int(limit),
            }
        except Exception as exc:  # pragma: no cover - surfaced in the UI for demo resilience.
            st.error(f"Evaluation failed: {exc}")

    st.divider()
    st.caption("Recording flow: build index, compare one question, show evidence, run evaluation.")

default_question = "What were Apple's 2023 net sales?"
st.markdown('<div class="query-shell">', unsafe_allow_html=True)
question = st.text_input("Question", default_question, label_visibility="visible")
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1, 3])
with left:
    run_question = st.button("Compare Methods", type="primary", use_container_width=True)
with right:
    st.markdown(
        '<div class="query-caption">Runs no retrieval, standard RAG, and reranked RAG on the same query.</div>',
        unsafe_allow_html=True,
    )

if run_question:
    try:
        with st.spinner("Retrieving evidence and generating grounded answers..."):
            st.session_state["answers"] = compare_methods(config, question)
    except Exception as exc:  # pragma: no cover - surfaced in the UI for demo resilience.
        st.error(f"Question comparison failed: {exc}")

metrics = st.session_state.get("metrics")
metrics_meta = st.session_state.get("metrics_meta")
if metrics is None and config.metrics_path.exists():
    metrics = read_json(config.metrics_path)
    metrics_meta = (
        read_json(config.run_metadata_path)
        if config.run_metadata_path.exists()
        else {"corpus": "saved index", "benchmark": "saved run", "limit": "saved run"}
    )

st.markdown('<div class="section-kicker">Evaluation Metrics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Method Performance</div>', unsafe_allow_html=True)

if metrics:
    meta = metrics_meta or {"corpus": "unknown", "benchmark": "unknown", "limit": "unknown"}
    st.markdown(
        f'<div class="run-meta">'
        f'corpus={escape(str(meta["corpus"]))} | benchmark={escape(str(meta["benchmark"]))} | '
        f'limit={escape(str(meta["limit"]))}'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(build_kpi_cards(metrics), unsafe_allow_html=True)
    st.markdown(build_comparison_table(metrics), unsafe_allow_html=True)
    if meta["benchmark"] in {"sample", "last saved"}:
        st.markdown(
            '<div class="note">'
            "Demo metrics are useful for showing the pipeline behavior. Final research claims should come from the larger "
            "EDGAR + FinanceBench run, with FinDER as an optional stress test."
            "</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Run evaluation to populate the comparison table.")

answers = st.session_state.get("answers")
if answers:
    with st.expander("Evidence Trace", expanded=False):
        st.markdown(build_answer_cards(answers), unsafe_allow_html=True)
        st.markdown(build_evidence_summary(answers), unsafe_allow_html=True)

artifact_paths = [
    config.data_summary_path,
    config.metrics_path,
    config.summary_csv_path,
    config.records_path,
    config.project_root / "results" / "mem2_runs",
]
with st.expander("Reproducibility Outputs", expanded=False):
    st.markdown(build_artifact_list(artifact_paths, config.project_root), unsafe_allow_html=True)
