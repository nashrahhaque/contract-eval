"""Streamlit dashboard for contract clause extraction evaluation."""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Contract Eval — Crosby AI", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.stApp { background: #0f1117; }
.metric-box { background:#1c1f26; border-radius:10px; padding:18px; text-align:center; }
.metric-val { font-size:2rem; font-weight:700; color:#4ade80; }
.metric-lbl { font-size:0.8rem; color:#9ca3af; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    p = Path("data/results.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

data = load()
if not data:
    st.error("Run `python pipeline.py` first to generate data/results.json")
    st.stop()

metrics = data["metrics"]
MODELS = list(metrics.keys())
PALETTE = ["#4ade80", "#60a5fa", "#f472b6"]
RISK_COLOR = {"high": "#ef4444", "medium": "#f97316", "low": "#22c55e"}

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚖️ Contract Eval")
    st.markdown("LLM clause extraction benchmark for legal AI systems.")
    st.divider()
    model = st.selectbox("Model", MODELS)
    st.divider()
    page = st.radio("View", ["Overview", "Per-Clause Breakdown", "Red Flags & Calibration", "Error Analysis"])

m = metrics[model]

# ── Overview ──
if page == "Overview":
    st.markdown("## Model Benchmark Overview")
    st.caption("Evaluating precision, recall, and risk-weighted F1 across 3 LLM extractors on 20 synthetic contracts.")

    cols = st.columns(4)
    for col, (label, val) in zip(cols, [
        ("Macro F1", f"{m['macro_f1']:.3f}"),
        ("Risk-Weighted F1", f"{m['risk_weighted_f1']:.3f}"),
        ("Red Flag F1", f"{m['red_flag_f1']:.3f}"),
        ("Calibration ECE ↓", f"{m['ece']:.3f}"),
    ]):
        col.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

    st.divider()

    # Model comparison bar chart
    df = pd.DataFrame([
        {"Model": mod, "Metric": k, "Value": metrics[mod][k]}
        for mod in MODELS
        for k in ["macro_f1", "risk_weighted_f1", "red_flag_f1"]
    ])
    df["Metric"] = df["Metric"].map({
        "macro_f1": "Macro F1",
        "risk_weighted_f1": "Risk-Weighted F1",
        "red_flag_f1": "Red Flag F1",
    })
    fig = px.bar(df, x="Model", y="Value", color="Metric", barmode="group",
                 color_discrete_sequence=PALETTE, template="plotly_dark",
                 title="Model Comparison — Key Metrics")
    fig.update_layout(paper_bgcolor="#1c1f26", plot_bgcolor="#1c1f26", yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)

    # Recall vs Precision scatter
    scatter_df = pd.DataFrame([
        {"Model": mod, "Precision": metrics[mod]["macro_precision"],
         "Recall": metrics[mod]["macro_recall"],
         "RW-F1": metrics[mod]["risk_weighted_f1"]}
        for mod in MODELS
    ])
    fig2 = px.scatter(scatter_df, x="Precision", y="Recall", text="Model",
                      size="RW-F1", color="Model",
                      color_discrete_sequence=PALETTE, template="plotly_dark",
                      title="Precision vs Recall (bubble = Risk-Weighted F1)")
    fig2.update_traces(textposition="top center")
    fig2.update_layout(paper_bgcolor="#1c1f26", xaxis_range=[0.6,1], yaxis_range=[0.6,1])
    st.plotly_chart(fig2, use_container_width=True)

    st.info("**Note:** In legal contract review, recall is weighted more heavily than precision — "
            "a missed high-risk clause (false negative) is riskier than a spurious flag (false positive). "
            "Risk-weighted F1 penalises high-risk errors 3× more.")

# ── Per-Clause Breakdown ──
elif page == "Per-Clause Breakdown":
    st.markdown(f"## Per-Clause Performance — `{model}`")

    pc = m["per_clause"]
    df = pd.DataFrame([
        {"Clause": v["name"], "Risk": v["risk"], "Precision": v["precision"],
         "Recall": v["recall"], "F1": v["f1"], "Support": v["support"],
         "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]}
        for v in pc.values()
    ]).sort_values("F1", ascending=False)

    fig = px.bar(df, x="Clause", y="F1", color="Risk",
                 color_discrete_map=RISK_COLOR, template="plotly_dark",
                 title="F1 Score by Clause Type")
    fig.update_layout(paper_bgcolor="#1c1f26", xaxis_tickangle=-35, yaxis_range=[0,1])
    fig.add_hline(y=m["macro_f1"], line_dash="dash", line_color="white",
                  annotation_text="Macro avg")
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: model × clause F1
    st.markdown("#### All Models — Clause F1 Heatmap")
    clause_names = [v["name"] for v in list(metrics[MODELS[0]]["per_clause"].values())]
    heat = np.array([
        [metrics[mod]["per_clause"][cid]["f1"] for cid in metrics[mod]["per_clause"]]
        for mod in MODELS
    ])
    fig2 = px.imshow(heat, x=clause_names, y=MODELS,
                     color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                     text_auto=".2f", template="plotly_dark",
                     title="F1 Heatmap: Models × Clause Types")
    fig2.update_layout(paper_bgcolor="#1c1f26", xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.style.background_gradient(subset=["F1"], cmap="RdYlGn"),
                 use_container_width=True, hide_index=True)

# ── Red Flags & Calibration ──
elif page == "Red Flags & Calibration":
    st.markdown(f"## Red Flag Detection & Confidence Calibration — `{model}`")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Red Flag Detection")
        rf_df = pd.DataFrame([
            {"Model": mod, "Metric": k, "Value": metrics[mod][k]}
            for mod in MODELS
            for k in ["red_flag_precision", "red_flag_recall", "red_flag_f1"]
        ])
        rf_df["Metric"] = rf_df["Metric"].str.replace("red_flag_", "").str.title()
        fig = px.bar(rf_df, x="Metric", y="Value", color="Model", barmode="group",
                     color_discrete_sequence=PALETTE, template="plotly_dark",
                     title="Red Flag Detection across Models")
        fig.update_layout(paper_bgcolor="#1c1f26", yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red flags = specific risky patterns (e.g. 'uncapped indemnification', "
                   "'no DPA', 'AI model trained on client data').")

    with col2:
        st.markdown("### Confidence Calibration (ECE)")
        ece_df = pd.DataFrame([
            {"Model": mod, "ECE": metrics[mod]["ece"]}
            for mod in MODELS
        ])
        fig2 = px.bar(ece_df, x="Model", y="ECE", color="Model",
                      color_discrete_sequence=PALETTE, template="plotly_dark",
                      title="Expected Calibration Error (lower = better)")
        fig2.update_layout(paper_bgcolor="#1c1f26")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("ECE measures how well model confidence scores match actual accuracy. "
                   "Well-calibrated models allow legal teams to trust confidence thresholds "
                   "for human review routing.")

# ── Error Analysis ──
elif page == "Error Analysis":
    st.markdown(f"## Error Analysis — `{model}`")

    preds = data["predictions"][model]
    rows = [r for r in preds if r["outcome"] in ("fn", "fp", "miscat")]

    df_err = pd.DataFrame([
        {
            "Contract": r["contract_id"],
            "Type": r["type"].upper(),
            "Error": r["outcome"].upper(),
            "True Clause": r["true"] or "—",
            "Predicted": r["pred"] or "—",
            "Confidence": r["conf"],
        }
        for r in rows
    ])

    col1, col2, col3 = st.columns(3)
    col1.metric("False Negatives (missed)", m["total_fn"], delta=None)
    col2.metric("False Positives (hallucinated)", m["total_fp"], delta=None)
    col3.metric("Miscategorizations", sum(1 for r in preds if r["outcome"]=="miscat"))

    st.markdown("#### Error Breakdown by Clause")
    from pipeline import CLAUSES
    err_by_clause = {}
    for r in preds:
        cid = r["true"] or r["pred"]
        if not cid or cid not in CLAUSES: continue
        err_by_clause.setdefault(cid, {"fn":0,"fp":0,"miscat":0})
        if r["outcome"] in err_by_clause[cid]:
            err_by_clause[cid][r["outcome"]] += 1

    ec_df = pd.DataFrame([
        {"Clause": CLAUSES[cid]["name"], "Risk": CLAUSES[cid]["risk"], **counts}
        for cid, counts in err_by_clause.items() if any(counts.values())
    ]).sort_values("fn", ascending=False)

    fig = px.bar(ec_df, x="Clause", y=["fn","fp","miscat"],
                 color_discrete_map={"fn": "#ef4444","fp":"#f97316","miscat":"#a78bfa"},
                 template="plotly_dark", title="Errors by Clause Type",
                 labels={"value":"Count","variable":"Error Type"})
    fig.update_layout(paper_bgcolor="#1c1f26", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Error Log")
    st.dataframe(df_err, use_container_width=True, hide_index=True)

    st.markdown("#### Active Learning Candidates")
    st.caption("High-error clauses where human review would most improve the model:")
    top_err = ec_df.nlargest(5, "fn")[["Clause","Risk","fn","fp","miscat"]]
    st.dataframe(top_err, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#6b7280;font-size:0.78rem;">contract-eval · LLM evaluation framework for legal AI</div>', unsafe_allow_html=True)
