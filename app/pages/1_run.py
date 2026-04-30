"""
Page 1 — Run Analysis

Upload a gene expression CSV (rows=samples, cols=features/genes).
Select a method and panel size k, then run feature selection.
Results are downloadable as a CSV feature list.

Runtime methods (all < 5 s on microarray data after ANOVA pre-filter):
  ANOVA, MI, mRMR, ReliefF, L1-Logistic, ElasticNet, LinSVC-L1,
  ExtraTrees, RRA(ANOVA+MI) [auto / explicit]

EA methods (30–120 s, run locally — not available here in deployed mode):
  GA, BPSO  →  download clinifs and run locally.
"""
import sys, os
# app/pages/ -> app/ -> pkg_root (where clinifs/ lives)
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
# app/ for _utils
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.title("🔬 Run Feature Selection")
st.markdown(
    "Upload your gene expression matrix (CSV) and select a method. "
    "Feature selection runs directly in the browser — no data leaves your machine."
)

# ─── Sidebar controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    method = st.selectbox(
        "Method",
        options=[
            "auto (adaptive RRA)",
            "RRA (ANOVA + MI)",
            "ANOVA",
            "Mutual Information",
            "mRMR",
            "ReliefF",
            "L1-Logistic",
            "ElasticNet",
            "LinSVC-L1",
            "ExtraTrees",
            "BorutaPy",
            "RFECV",
        ],
        index=0,
        help="'auto' estimates dataset difficulty and applies the benchmark-recommended method.",
    )
    k = st.slider("Panel size (k)", min_value=3, max_value=100, value=20, step=1)
    run_cv = st.checkbox("Estimate AUC via 5-fold CV", value=True,
                         help="Adds ~5 s. Fits a LogisticRegression on the selected panel to estimate discovery AUC.")

    st.divider()
    st.caption(
        "⚠️ GA / BPSO require 30–120 s per fit and are not available here. "
        "Install [clinifs](https://github.com/xwdshiwo/clinifs) and run locally."
    )

# ─── File upload ─────────────────────────────────────────────────────────────

st.subheader("1 · Upload data")
uploaded = st.file_uploader(
    "Expression matrix CSV — rows = samples, last column = label (0/1)",
    type=["csv"],
    help="The last column must be the binary class label (0/1). "
         "Feature columns should be gene probe IDs or gene symbols.",
)

if uploaded is None:
    st.info("Upload a CSV to get started. "
            "Expected format: N samples × P features + 1 label column (last).")
    st.stop()

# ─── Parse data ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Parsing data…")
def parse_csv(data_bytes):
    df = pd.read_csv(io.BytesIO(data_bytes))
    feature_cols = df.columns[:-1].tolist()
    label_col    = df.columns[-1]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_col].values
    # Encode string labels
    if y_raw.dtype.kind not in ("i", "u", "f"):
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = y_raw.astype(int)
    return X, y, feature_cols

try:
    X, y, feature_names = parse_csv(uploaded.read())
except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

n_samples, n_features = X.shape
st.success(f"Loaded: **{n_samples} samples × {n_features} features** "
           f"| Classes: {np.unique(y).tolist()}")

# ─── Method mapping ──────────────────────────────────────────────────────────

_METHOD_KEY = {
    "auto (adaptive RRA)"  : "auto",
    "RRA (ANOVA + MI)"     : "rra",
    "ANOVA"                : "anova",
    "Mutual Information"   : "mi",
    "mRMR"                 : "mrmr",
    "ReliefF"              : "relieff",
    "L1-Logistic"          : "l1",
    "ElasticNet"           : "elasticnet",
    "LinSVC-L1"            : "linsvc",
    "ExtraTrees"           : "extratrees",
    "BorutaPy"             : "boruta",
    "RFECV"                : "rfecv",
}

# ─── Run ─────────────────────────────────────────────────────────────────────

st.subheader("2 · Run")
if st.button("▶ Run Feature Selection", type="primary"):
    from clinifs import FeatureSelector

    method_key = _METHOD_KEY[method]
    with st.spinner(f"Running {method} (k={k})…"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fs = FeatureSelector(method=method_key, k=k)
                fs.fit(X, y)
            for w in caught:
                st.warning(str(w.message))
        except Exception as e:
            st.error(f"Feature selection failed: {e}")
            st.stop()

    sel_idx = fs.selected_indices_
    sel_names = [feature_names[i] for i in sel_idx]

    # Tier info (auto only)
    if hasattr(fs, "tier_") and fs.tier_:
        tier_color = {"easy_medium": "🟢", "hard": "🔴"}.get(fs.tier_, "")
        st.info(
            f"{tier_color} Auto-detected tier: **{fs.tier_.replace('_', '/')}** "
            f"(random-baseline AUC = {fs.random_baseline_auc_:.3f})"
        )

    st.success(f"Selected **{len(sel_names)} features** out of {n_features}")

    # ── Optional CV estimate ──────────────────────────────────────────────
    if run_cv:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        X_sel = X[:, sel_idx]
        clf = LogisticRegression(C=1, solver="liblinear", max_iter=500, random_state=42)
        with st.spinner("5-fold CV AUC estimate…"):
            try:
                aucs = cross_val_score(clf, X_sel, y, cv=5, scoring="roc_auc")
                st.metric("Discovery AUC (5-fold CV)", f"{aucs.mean():.4f}",
                          delta=f"± {aucs.std():.4f}")
            except Exception as e:
                st.warning(f"CV failed: {e}")

    # ── Display table ─────────────────────────────────────────────────────
    st.subheader("3 · Selected panel")
    result_df = pd.DataFrame({
        "rank": range(1, len(sel_names) + 1),
        "feature": sel_names,
        "original_index": sel_idx,
    })
    st.dataframe(result_df, use_container_width=True, height=400)

    # ── Download ─────────────────────────────────────────────────────────
    csv_bytes = result_df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download panel CSV",
        data=csv_bytes,
        file_name=f"clinifs_panel_k{k}_{method_key}.csv",
        mime="text/csv",
    )
