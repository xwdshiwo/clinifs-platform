"""
Page 4 — Custom RRA Builder

Compose your own RRA from 2–3 scoring methods.
If you have run MEL locally, upload its feature scores to form a
three-method RRA (ANOVA + MI + MEL) — the paper's strongest pipeline.
"""
import sys, os
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

st.title("⚙️ Custom RRA Builder")
st.markdown(
    "Build a custom **Robust Rank Aggregation** ensemble from 2–3 scoring methods. "
    "Optionally upload MEL scores from a local run to form the strongest three-method pipeline."
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("RRA Configuration")

    available_methods = ["anova", "mi", "mrmr", "relieff"]
    selected_methods = st.multiselect(
        "Scoring methods (select 2–3)",
        options=available_methods,
        default=["anova", "mi"],
        format_func=lambda x: {"anova":"ANOVA","mi":"Mutual Info",
                                "mrmr":"mRMR","relieff":"ReliefF"}[x],
    )
    k = st.slider("Panel size (k)", 3, 100, 20)
    run_cv = st.checkbox("5-fold CV AUC estimate", value=True)

    st.divider()
    st.subheader("Upload MEL scores (optional)")
    mel_file = st.file_uploader(
        "MEL score CSV — one column named 'score', same feature order as data",
        type=["csv"],
        key="mel_upload",
    )
    if mel_file:
        st.success("MEL scores loaded ✓")

# ─── Data upload ─────────────────────────────────────────────────────────────

st.subheader("1 · Upload expression data")
data_file = st.file_uploader(
    "Expression CSV — rows=samples, last col=label (0/1)",
    type=["csv"],
    key="data_upload",
)

if data_file is None:
    st.info("Upload a CSV to enable the custom RRA.")
    st.stop()

@st.cache_data(show_spinner="Parsing…")
def parse_csv(b):
    df = pd.read_csv(io.BytesIO(b))
    feat = df.columns[:-1].tolist()
    X = df[feat].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    if y_raw.dtype.kind not in ("i","u","f"):
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(int)
    return X, y, feat

X, y, feature_names = parse_csv(data_file.read())
n_s, n_f = X.shape
st.success(f"Loaded: **{n_s} samples × {n_f} features**")

# ─── MEL scores ──────────────────────────────────────────────────────────────

mel_scores = None
if mel_file:
    try:
        mel_df = pd.read_csv(mel_file)
        col = mel_df.columns[0]
        mel_scores = mel_df[col].values.astype(float)
        if len(mel_scores) != n_f:
            st.error(f"MEL scores length ({len(mel_scores)}) ≠ n_features ({n_f}). Ignoring.")
            mel_scores = None
        else:
            st.info(f"Using three-method RRA: {selected_methods} + MEL")
    except Exception as e:
        st.error(f"Could not parse MEL file: {e}")
        mel_scores = None

selected_methods_run = selected_methods + (["mel"] if mel_scores is not None else [])

# ─── Build & run ─────────────────────────────────────────────────────────────

if len(selected_methods) < 2:
    st.warning("Please select at least 2 methods.")
    st.stop()

st.subheader("2 · Run")

if st.button("▶ Run Custom RRA", type="primary"):
    from clinifs import RankAggregateFilter

    with st.spinner(f"Running RRA({', '.join(selected_methods_run)}, k={k})…"):
        try:
            extra = {"mel": mel_scores} if mel_scores is not None else None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rra = RankAggregateFilter(
                    methods=selected_methods_run, k=k
                )
                rra.fit(X, y, extra_scores=extra)
            for w in caught:
                st.warning(str(w.message))
        except Exception as e:
            st.error(f"RRA failed: {e}")
            st.stop()

    sel_idx   = rra.selected_indices_
    sel_names = [feature_names[i] for i in sel_idx]
    # rho_scores_ is in pre-filtered space; selected features' rho values
    # are simply the k smallest in ascending order
    rho_sorted = np.sort(rra.rho_scores_)[:len(sel_names)]

    st.success(f"Selected **{len(sel_names)} features** via RRA({', '.join(selected_methods_run)})")

    # CV estimate
    if run_cv:
        X_sel = X[:, sel_idx]
        clf = LogisticRegression(C=1, solver="liblinear", max_iter=500, random_state=42)
        with st.spinner("5-fold CV…"):
            try:
                aucs = cross_val_score(clf, X_sel, y, cv=5, scoring="roc_auc")
                st.metric("Discovery AUC (5-fold CV)", f"{aucs.mean():.4f}",
                          delta=f"± {aucs.std():.4f}")
            except Exception as e:
                st.warning(f"CV failed: {e}")

    # ── Panel table ──────────────────────────────────────────────
    st.subheader("3 · Selected panel")

    result_df = pd.DataFrame({
        "rank"           : range(1, len(sel_names)+1),
        "feature"        : sel_names,
        "original_index" : sel_idx,
        "rho_score"      : [f"{v:.4e}" for v in rho_sorted],
    })
    st.dataframe(result_df, use_container_width=True, height=400)

    csv_bytes = result_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download panel CSV",
        data=csv_bytes,
        file_name=f"clinifs_rra_k{k}.csv",
        mime="text/csv",
    )

    # ── ρ-score distribution ────────────────────────────────────────────
    st.subheader("RRA ρ-score distribution (lower = consistently top-ranked)")
    rho_all = rra.rho_scores_
    plot_rho = pd.DataFrame({
        "feature_rank": range(1, len(rho_all)+1),
        "rho": np.sort(rho_all),
    })
    fig = px.line(
        plot_rho.head(min(200, len(plot_rho))),
        x="feature_rank", y="rho",
        labels={"feature_rank": "Feature rank", "rho": "ρ-score (Bonferroni)"},
        title="ρ-score vs feature rank (top 200 features)",
    )
    fig.add_vline(x=k, line_dash="dash", line_color="red",
                  annotation_text=f"k={k}", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

# ─── Method comparison from benchmark ──────────────────────────────────────
st.divider()
st.subheader("Benchmark reference — single methods on Hard tier (k=20)")
st.caption(
    "Source: 5×5 CV across Bladder_GSE31189 + Prostate_GSE6919_U95Av2 at k=20. "
    "RRA configurations were not benchmarked at fixed k; the manuscript reports "
    "RRA(ANOVA+MI) gains of +3.1 pp over single ANOVA (macro-averaged across k=3–50) "
    "and +1.5–1.7 pp additional gain when MEL is added (k=3–15 only)."
)
ref = pd.DataFrame({
    "Method":          ["MEL",   "ReliefF", "RFECV",  "ElasticNet", "ANOVA", "MI"],
    "Hard AUC k=20":   [0.6900,  0.6812,    0.6758,   0.6689,       0.6422,  0.6452],
    "Stability (Φ)":   [0.068,   0.257,     0.029,    0.203,        0.308,   0.163],
    "Source for RRA":  ["✓ (upload)", "—",    "—",     "—",          "built-in","built-in"],
})
st.dataframe(ref, use_container_width=True, hide_index=True)
st.caption(
    "For RRA, the manuscript only reports tier-level macro-averages "
    "(see Discussion §3.6); fixed-k RRA values are not in the benchmark."
)
