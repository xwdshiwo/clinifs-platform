"""
Page 2 — Browse Pre-computed Benchmark Results

Shows the 1 155-cell benchmark (15 methods × 11 datasets × 5×5 CV)
aggregated by difficulty tier.  All numbers are from real experiments.

Key insight: method rankings depend heavily on k (panel size).
  k=3–5:   ReliefF dominant
  k=10–20: MEL dominant (clinically relevant range)
  k=30–50: ReliefF / mRMR competitive
The "Multi-k average" tab averages all k values; the "AUC vs k" tab
shows the k-dependent story.

MEL and SFE results shown for reference only (not available in the
online Run tool due to compute constraints).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from _utils import (
    load_tier_auc, load_tier_stability, load_four_dim,
    load_dataset_difficulty, load_summary_all,
    FAMILY_COLOR, TIER_COLOR, METHOD_DISPLAY, METHOD_FAMILY,
    _path,
)

@st.cache_data
def load_e2():
    df = pd.read_csv(_path("e2_long_enriched.csv"))
    df["display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["family"]  = df["method"].map(METHOD_FAMILY).fillna("Other")
    return df

st.title("📊 Benchmark Results Browser")
st.markdown(
    "All results are from a **5×5 nested cross-validation** (5 repetitions × 5 folds) "
    "on **11 cancer microarray datasets** across **15 feature selection methods**. "
    "Datasets are grouped into three difficulty tiers based on cross-method mean AUC."
)

# ─── Data ────────────────────────────────────────────────────────────────────
auc_df       = load_tier_auc()
stab_df      = load_tier_stability()
four_df      = load_four_dim()
diff_df      = load_dataset_difficulty()
summary_df   = load_summary_all()
e2_df        = load_e2()

# ─── Tab layout ──────────────────────────────────────────────────────────────
tab1, tab_k, tab2, tab3, tab4 = st.tabs([
    "AUC (Multi-k avg)", "AUC vs k", "Stability by Tier", "4-D Profile", "Dataset Details"
])

# ════════════════════════════════════════════════════════════════════════════
# Tab 1: AUC heatmap (multi-k average)
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Mean AUC by Method × Difficulty Tier")
    st.caption(
        "⚠️ Numbers are **averaged across k = 3, 5, 10, 15, 20, 30, 50**. "
        "This inflates methods that perform well at small k. "
        "See **AUC vs k** tab for the k-dependent story."
    )

    pivot = (
        auc_df
        .sort_values("Hard", ascending=False)
        .set_index("display")[["Easy", "Medium", "Hard", "grand_mean"]]
    )
    pivot.columns = ["Easy", "Medium", "Hard", "Grand mean"]

    fig = px.imshow(
        pivot,
        text_auto=".4f",
        color_continuous_scale="RdYlGn",
        zmin=0.55, zmax=1.0,
        aspect="auto",
        labels={"color": "AUC"},
        title="Mean AUC averaged over k=3,5,10,15,20,30,50 (5×5 CV)",
    )
    fig.update_layout(height=520, font_size=13,
                      coloraxis_colorbar=dict(thickness=12))
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart: Hard-tier comparison
    st.subheader("Hard-Tier AUC Comparison (multi-k average)")
    bar_df = (
        auc_df
        .assign(color=auc_df["family"].map(FAMILY_COLOR).fillna("#999"))
        .sort_values("Hard", ascending=True)
    )
    fig2 = px.bar(
        bar_df, x="Hard", y="display",
        orientation="h",
        color="family",
        color_discrete_map=FAMILY_COLOR,
        labels={"Hard": "Mean AUC (Hard tier, avg k)", "display": "Method", "family": "Family"},
        title="Hard-Tier AUC — averaged over k=3–50",
    )
    fig2.update_layout(height=460, showlegend=True,
                       xaxis_range=[0.55, 0.72])
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# Tab_k: AUC vs k (new — shows k-dependent story)
# ════════════════════════════════════════════════════════════════════════════
with tab_k:
    st.subheader("Hard-Tier AUC vs Panel Size k")
    st.markdown(
        "**Key finding**: MEL leads at k=10–20 (clinical panel range), "
        "ReliefF leads at small (k=3–5) and large (k=50) panels. "
        "Averaged across all k, ReliefF appears marginally higher — but "
        "for typical clinical usage (k=10–20), MEL is the strongest single method."
    )

    hard_ds   = ["Bladder_GSE31189", "Prostate_GSE6919_U95Av2"]
    hard_e2   = e2_df[e2_df["dataset"].isin(hard_ds)]
    hard_by_k = (
        hard_e2
        .groupby(["method", "display", "family", "k"])["auc_mean"]
        .mean().reset_index()
        .rename(columns={"auc_mean": "AUC"})
    )

    col_left, col_right = st.columns([2, 1])
    with col_right:
        highlight = st.multiselect(
            "Highlight methods",
            options=sorted(hard_by_k["display"].unique()),
            default=["MEL", "ReliefF", "ANOVA", "Mutual Info", "mRMR"],
        )

    with col_left:
        plot_df = hard_by_k.copy()
        plot_df["opacity"] = plot_df["display"].apply(
            lambda x: 1.0 if x in highlight else 0.15
        )
        fig_k = px.line(
            plot_df, x="k", y="AUC",
            color="display",
            color_discrete_map={d: FAMILY_COLOR.get(METHOD_FAMILY.get(m,""),"#ccc")
                                 for m, d in zip(hard_by_k["method"], hard_by_k["display"])},
            markers=True,
            labels={"AUC": "Mean AUC (Hard tier)", "k": "Panel size k", "display": "Method"},
            title="Hard-tier AUC vs k  (avg over Bladder + Prostate datasets)",
        )
        # Make non-highlighted traces faint
        for trace in fig_k.data:
            if trace.name not in highlight:
                trace.opacity = 0.15
                trace.line.width = 1
            else:
                trace.line.width = 3
        fig_k.update_layout(height=500, xaxis=dict(tickvals=[3,5,10,15,20,30,50]))
        st.plotly_chart(fig_k, use_container_width=True)

    # Pivot table: Hard AUC at each k
    st.subheader("Hard-tier AUC at each k (fixed panel budget comparison)")
    pivot_k = hard_by_k.pivot_table(
        index="display", columns="k", values="AUC"
    ).round(4)
    pivot_k.columns = [f"k={c}" for c in pivot_k.columns]
    pivot_k["Multi-k avg"] = pivot_k.mean(axis=1).round(4)
    pivot_k = pivot_k.sort_values("k=20", ascending=False)
    st.dataframe(
        pivot_k.style.highlight_max(axis=0, color="#c6efce"),
        use_container_width=True,
    )
    st.caption(
        "Green cells = best method at that k. "
        "Sort by k=20 (clinically common panel size): "
        "MEL and ReliefF lead the Hard tier at this budget."
    )

# ════════════════════════════════════════════════════════════════════════════
# Tab 2: Stability heatmap
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Stability (Nogueira Φ) by Method × Tier")
    st.caption(
        "Φ ∈ [−1, 1]; higher = more reproducible panel across CV folds. "
        "Filter methods are consistently more stable than wrappers."
    )

    stab_pivot = (
        stab_df
        .sort_values("grand_mean", ascending=False)
        .set_index("display")[["Easy", "Medium", "Hard", "grand_mean"]]
    )
    stab_pivot.columns = ["Easy", "Medium", "Hard", "Grand mean"]

    fig3 = px.imshow(
        stab_pivot,
        text_auto=".3f",
        color_continuous_scale="Blues",
        zmin=0.0, zmax=0.80,
        aspect="auto",
        labels={"color": "Φ"},
        title="Nogueira Φ Stability (5×5 CV, k=50)",
    )
    fig3.update_layout(height=520, font_size=13,
                       coloraxis_colorbar=dict(thickness=12))
    st.plotly_chart(fig3, use_container_width=True)

    # AUC vs Stability scatter — hover labels only (no text overlap)
    st.subheader("AUC–Stability Trade-off (Hard Tier, multi-k avg)")
    merged = auc_df[["method","display","family","Hard"]].merge(
        stab_df[["method","Hard"]].rename(columns={"Hard": "Stability"}),
        on="method"
    )
    fig4 = px.scatter(
        merged, x="Stability", y="Hard",
        hover_name="display",
        color="family",
        color_discrete_map=FAMILY_COLOR,
        labels={"Hard": "AUC (Hard, avg k)", "Stability": "Φ Stability (Hard, avg k)"},
        title="Hard-tier AUC vs Φ Stability — hover over points for method names",
        size_max=14,
    )
    fig4.update_traces(marker_size=12)
    fig4.update_layout(height=480)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Hover over any point to see the method name. ⚠️ Both axes are multi-k averages; for k=20 specifically, MEL leads Hard AUC.")

# ════════════════════════════════════════════════════════════════════════════
# Tab 3: 4-D profile
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Four-Dimensional Method Profile")
    st.markdown(
        """
        | Dimension | Description |
        |---|---|
        | **AUC (Hard)** | Mean AUC on Hard-tier datasets |
        | **Stability (Φ)** | Nogueira Φ averaged across all tiers |
        | **CGC hit rate** | Fraction of selected genes in COSMIC Cancer Gene Census |
        | **External val. AUC** | Mean AUC on held-out validation cohorts |
        """
    )

    # Radar / spider chart
    dims = ["AUC_Hard", "Stability", "A3_CGC_rate", "Ext_val_AUC"]
    dim_labels = ["AUC (Hard)", "Stability", "CGC hit rate", "External val. AUC"]

    # Normalise to [0, 1] per dimension
    norm_df = four_df.copy()
    for d in dims:
        mn, mx = norm_df[d].min(), norm_df[d].max()
        norm_df[d] = (norm_df[d] - mn) / (mx - mn + 1e-9)

    family_filter = st.multiselect(
        "Filter by family",
        options=list(FAMILY_COLOR.keys()),
        default=list(FAMILY_COLOR.keys()),
    )
    plot_df = norm_df[norm_df["family"].isin(family_filter)]

    fig5 = go.Figure()
    for _, row in plot_df.iterrows():
        vals = [row[d] for d in dims]
        vals.append(vals[0])  # close polygon
        fig5.add_trace(go.Scatterpolar(
            r=vals,
            theta=dim_labels + [dim_labels[0]],
            fill="toself",
            opacity=0.35,
            name=row["display"],
            line=dict(color=FAMILY_COLOR.get(row["family"], "#888")),
        ))
    fig5.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=550,
        title="Normalised 4-D profile (each axis 0–1 within-dimension)",
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Table
    st.subheader("Raw 4-D Values")
    disp_4d = (
        four_df[["display", "family", "AUC_Hard", "Stability", "A3_CGC_rate", "Ext_val_AUC"]]
        .rename(columns={
            "display": "Method", "family": "Family",
            "AUC_Hard": "AUC (Hard)", "Stability": "Stability (Φ)",
            "A3_CGC_rate": "CGC hit rate", "Ext_val_AUC": "Ext. val. AUC",
        })
        .sort_values("AUC (Hard)", ascending=False)
    )
    st.dataframe(
        disp_4d.style.format({
            "AUC (Hard)": "{:.4f}",
            "Stability (Φ)": "{:.4f}",
            "CGC hit rate": "{:.4f}",
            "Ext. val. AUC": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# Tab 4: Dataset details
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Dataset Characteristics")

    diff_disp = diff_df.rename(columns={
        "dataset": "Dataset", "cancer": "Cancer type",
        "mean_auc": "Cross-method AUC",
        "difficulty_level": "Tier",
        "n_samples": "N samples",
        "n_features": "N features",
        "class_ratio": "Class ratio",
    })
    diff_disp["Tier"] = diff_disp["Tier"].str.capitalize()

    st.dataframe(
        diff_disp.style.format({
            "Cross-method AUC": "{:.4f}",
            "Class ratio": "{:.2f}",
        }).applymap(
            lambda v: f"background-color: {TIER_COLOR.get(v.lower(), '')}"
            if isinstance(v, str) and v.lower() in TIER_COLOR else "",
            subset=["Tier"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Per-Dataset AUC (all methods, k=50)")
    pivot2 = (
        summary_df
        .assign(
            method_disp=summary_df["method"].map(METHOD_DISPLAY).fillna(summary_df["method"])
        )
        .pivot_table(index="dataset_short", columns="method_disp", values="auc_mean")
        .round(4)
    )
    st.dataframe(pivot2, use_container_width=True)
