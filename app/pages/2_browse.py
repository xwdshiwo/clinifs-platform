import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _i18n import init_language, tr
from _utils import (
    FAMILY_COLOR,
    METHOD_DISPLAY,
    METHOD_FAMILY,
    TIER_COLOR,
    _path,
    load_dataset_difficulty,
    load_four_dim,
    load_tier_auc,
    load_tier_stability,
)

lang = init_language()


@st.cache_data
def load_e2():
    df = pd.read_csv(_path("e2_long_enriched.csv"))
    df["display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["family"] = df["method"].map(METHOD_FAMILY).fillna("Other")
    return df


auc_df = load_tier_auc()
stab_df = load_tier_stability()
four_df = load_four_dim()
diff_df = load_dataset_difficulty()
e2_df = load_e2()

st.title(tr("📊 Benchmark Results Explorer", "📊 基准结果浏览"))
st.markdown(
    tr(
        "Explore benchmark results across datasets, panel sizes, and evaluation criteria. Use the filters and plots below to compare methods under different task settings instead of relying on a single overall score.",
        "浏览不同数据集、基因面板规模和评价指标下的基准结果。你可以按任务设置比较方法，而不是只依赖单一总分。",
    )
)

st.info(
    tr(
        "Some published methods are included as benchmark references. Methods that require separate repositories or specialized workflows can be run locally and integrated through uploaded score files.",
        "部分已发表方法作为基准参考展示。依赖独立仓库或专项流程的方法可在本地运行，并通过上传分数文件参与融合分析。",
    )
)

# Key result cards
hard_auc = auc_df.sort_values("Hard", ascending=False).iloc[0]
easy_stab = stab_df.sort_values("Easy", ascending=False).iloc[0]
mel_k20 = (
    e2_df[(e2_df["difficulty_tier"] == "Hard") & (e2_df["method"] == "mel") & (e2_df["k"] == 20)]["auc_mean"].mean()
)
relieff_k20 = (
    e2_df[(e2_df["difficulty_tier"] == "Hard") & (e2_df["method"] == "relieff") & (e2_df["k"] == 20)]["auc_mean"].mean()
)
hard_k20 = (
    e2_df[(e2_df["difficulty_tier"] == "Hard") & (e2_df["k"] == 20)]
    .groupby(["method", "display"], as_index=False)["auc_mean"]
    .mean()
    .sort_values("auc_mean", ascending=False)
    .iloc[0]
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(tr("Hard-tier top multi-k AUC", "高难度多 k 平均 AUC 最高"), METHOD_DISPLAY.get(hard_auc["method"], hard_auc["method"]), f"{hard_auc['Hard']:.3f}")
c2.metric(tr("Hard-tier k=20 reference", "高难度 k=20 参考"), hard_k20["display"], f"AUC={hard_k20['auc_mean']:.3f}")
c3.metric(tr("Most stable Easy-tier method", "Easy 档最稳定方法"), METHOD_DISPLAY.get(easy_stab["method"], easy_stab["method"]), f"Φ={easy_stab['Easy']:.3f}")
c4.metric(tr("Consensus option", "共识融合选项"), "RRA", tr("combine rankings", "融合排序"))

story, auc_k, rra_tab, tradeoff, details = st.tabs(
    [
        tr("1 Overview", "1 概览"),
        tr("2 AUC vs k", "2 AUC 与 k"),
        tr("3 RRA consensus", "3 RRA 共识"),
        tr("4 Trade-offs", "4 多维权衡"),
        tr("5 Detail tables", "5 详细表格"),
    ]
)

with story:
    st.subheader(tr("Compare methods within the right task context", "在合适任务背景下比较方法"))
    cols = st.columns([1.15, 0.85])
    with cols[0]:
        diff_show = diff_df.copy()
        diff_show["tier"] = diff_show["difficulty_level"].str.capitalize()
        fig = px.scatter(
            diff_show,
            x="n_samples",
            y="mean_auc",
            size="n_features",
            color="tier",
            color_discrete_map={"Easy": "#67B99A", "Medium": "#F0A500", "Hard": "#E07070"},
            hover_name="dataset",
            labels={"n_samples": tr("Number of samples", "样本量"), "mean_auc": tr("Cross-method mean AUC", "跨方法平均 AUC"), "tier": tr("Difficulty", "难度")},
            title=tr("Dataset difficulty defines the interpretation space", "数据集难度决定方法比较的解释空间"),
        )
        fig.update_layout(height=470)
        st.plotly_chart(fig, width="stretch")
    with cols[1]:
        st.markdown(
            tr(
                """
**How to read the benchmark**

1. Easy datasets are nearly saturated, so methods look similar.
2. Hard datasets expose method differences, but absolute AUC remains limited.
3. Small panels should be compared at the target k rather than by a global average.
4. Consensus methods can combine complementary rankings when no single method is clearly dominant.
5. Stability, biological plausibility and external validation provide complementary evidence.
""",
                """
**如何阅读基准结果**

1. Easy 数据集接近饱和，因此方法差异被压缩。
2. Hard 数据集能暴露方法差异，但绝对 AUC 仍有限。
3. 小面板任务应在目标 k 下比较，而不是只看全局均值。
4. 当没有单一方法明显占优时，共识方法可以融合互补排序。
5. 稳定性、生物学合理性和外部验证提供互补证据。
""",
            )
        )

with auc_k:
    st.subheader(tr("AUC depends on panel size k", "AUC 随面板规模 k 改变"))
    tier_choice = st.radio(
        tr("Difficulty tier", "难度档"),
        ["Hard", "Medium", "Easy"],
        horizontal=True,
    )
    default_methods = ["MEL", "ReliefF", "ANOVA", "Mutual Info", "mRMR", "SFE"] if tier_choice == "Hard" else ["ANOVA", "mRMR", "ReliefF", "MEL", "Mutual Info"]
    tier_df = e2_df[e2_df["difficulty_tier"] == tier_choice]
    by_k = tier_df.groupby(["method", "display", "family", "k"], as_index=False)["auc_mean"].mean().rename(columns={"auc_mean": "AUC"})
    selected = st.multiselect(
        tr("Methods to highlight", "选择要突出显示的方法"),
        sorted(by_k["display"].unique()),
        default=[m for m in default_methods if m in set(by_k["display"])],
    )
    fig = px.line(
        by_k,
        x="k",
        y="AUC",
        color="display",
        markers=True,
        color_discrete_map={row["display"]: FAMILY_COLOR.get(row["family"], "#999999") for _, row in by_k.iterrows()},
        labels={"k": tr("Panel size k", "面板规模 k"), "display": tr("Method", "方法")},
        title=tr(f"{tier_choice}-tier AUC across panel sizes", f"{tier_choice} 档不同 k 下的 AUC"),
    )
    for trace in fig.data:
        if trace.name in selected:
            trace.line.width = 3.5
            trace.opacity = 1.0
        else:
            trace.line.width = 1.0
            trace.opacity = 0.15
    fig.update_layout(height=540, xaxis=dict(tickvals=[3, 5, 10, 15, 20, 30, 50]))
    st.plotly_chart(fig, width="stretch")

    pivot = by_k.pivot_table(index="display", columns="k", values="AUC").round(4)
    pivot.columns = [f"k={c}" for c in pivot.columns]
    sort_col = "k=20" if "k=20" in pivot.columns else pivot.columns[0]
    st.dataframe(pivot.sort_values(sort_col, ascending=False).style.highlight_max(axis=0, color="#d8f3dc"), width="stretch")

with rra_tab:
    st.subheader(tr("Using RRA to combine complementary rankings", "使用 RRA 融合互补排序"))
    st.markdown(
        tr(
            """
RRA is useful when several ranking methods provide complementary signals. The web app supports built-in rankings such as Welch t-test, ANOVA, Mutual Information, mRMR, ReliefF and model-based scores, and can also accept external score files from published methods.

- Use RRA when you want a consensus panel rather than a single-method list.
- Use external scores when a method is available from another repository or local workflow.
- Treat the result as an exploratory panel unless it is validated on an independent cohort.
""",
            """
当多个排序方法提供互补信号时，RRA 可用于生成共识面板。网页端支持 Welch t 检验、ANOVA、互信息、mRMR、ReliefF 和模型重要性等内置排序，也可以接收来自已发表方法的外部分数文件。

- 如果希望获得共识面板，而不是单一方法列表，可以使用 RRA。
- 如果某个方法来自其他仓库或本地流程，可以上传其分数作为外部排序。
- 除非经过独立队列验证，否则结果应视为探索性候选面板。
""",
        )
    )
    rra_summary = pd.DataFrame(
        {
            tr("Scenario", "场景"): [tr("General consensus panel", "通用共识面板"), tr("External method available", "已有外部方法结果"), tr("Stability-first panel", "稳定性优先面板"), tr("Single-method online baseline", "在线单方法基线")],
            tr("Candidate strategy", "候选策略"): ["RRA(t-test+ANOVA+MI)", "RRA built-in + external score", "ANOVA / t-test", "ReliefF / model-based scores"],
            tr("Reason", "理由"): [
                tr("Combines simple and interpretable statistical rankings", "融合简单且可解释的统计排序"),
                tr("Allows published or local methods to be compared without re-implementing them", "无需重写外部方法，也能比较其排序结果"),
                tr("ANOVA provides a high-stability statistical baseline", "ANOVA 可作为高稳定性的统计基线"),
                tr("Useful when a single ranking is preferred for interpretation", "适合需要单一排序用于解释的场景"),
            ],
        }
    )
    st.dataframe(rra_summary, width="stretch", hide_index=True)

with tradeoff:
    st.subheader(tr("Performance is only one dimension", "预测性能只是一个维度"))
    merged = auc_df[["method", "display", "family", "Hard"]].merge(
        stab_df[["method", "Hard"]].rename(columns={"Hard": "Stability"}), on="method"
    )
    fig = px.scatter(
        merged,
        x="Stability",
        y="Hard",
        color="family",
        hover_name="display",
        text="display",
        color_discrete_map=FAMILY_COLOR,
        labels={"Hard": tr("Hard-tier AUC, multi-k average", "Hard 档 AUC，多 k 平均"), "Stability": tr("Hard-tier stability Φ", "Hard 档稳定性 Φ")},
        title=tr("Hard-tier AUC–stability trade-off", "Hard 档 AUC–稳定性权衡"),
    )
    fig.update_traces(textposition="top center", marker_size=11)
    fig.update_layout(height=540)
    st.plotly_chart(fig, width="stretch")

    dims = ["AUC_Hard", "Stability", "A3_CGC_rate", "Ext_val_AUC"]
    dim_labels = [tr("Hard AUC", "Hard AUC"), tr("Stability", "稳定性"), tr("CGC hit", "CGC 命中"), tr("External AUC", "外部 AUC")]
    compare_methods = st.multiselect(
        tr("Methods for 4-D comparison", "选择四维对比方法"),
        sorted(four_df["display"].unique()),
        default=[m for m in ["ANOVA", "Mutual Info", "ReliefF", "MEL", "SFE", "BPSO"] if m in set(four_df["display"])],
    )
    norm = four_df.copy()
    for d in dims:
        norm[d] = (norm[d] - norm[d].min()) / (norm[d].max() - norm[d].min() + 1e-9)
    fig_radar = go.Figure()
    for _, row in norm[norm["display"].isin(compare_methods)].iterrows():
        vals = [row[d] for d in dims]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=dim_labels + [dim_labels[0]],
                fill="toself",
                name=row["display"],
                line=dict(color=FAMILY_COLOR.get(row["family"], "#888888")),
                opacity=0.45,
            )
        )
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=560)
    st.plotly_chart(fig_radar, width="stretch")

with details:
    st.subheader(tr("Dataset difficulty table", "数据集难度表"))
    table = diff_df.rename(
        columns={
            "dataset": tr("Dataset", "数据集"),
            "cancer": tr("Cancer", "癌种"),
            "mean_auc": tr("Cross-method mean AUC", "跨方法平均 AUC"),
            "difficulty_level": tr("Tier", "难度档"),
            "n_samples": tr("Samples", "样本数"),
            "n_features": tr("Features", "特征数"),
            "class_ratio": tr("Class ratio", "类别比例"),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    st.subheader(tr("Full benchmark table", "完整基准表"))
    show_cols = ["display", "family", "dataset", "difficulty_tier", "k", "auc_mean", "stability_nogueira"]
    detail = e2_df[show_cols].rename(
        columns={
            "display": tr("Method", "方法"),
            "family": tr("Family", "家族"),
            "dataset": tr("Dataset", "数据集"),
            "difficulty_tier": tr("Tier", "难度"),
            "k": "k",
            "auc_mean": "AUC",
            "stability_nogueira": "Φ",
        }
    )
    method_filter = st.multiselect(tr("Filter methods", "筛选方法"), sorted(detail[tr("Method", "方法")].unique()), default=[])
    if method_filter:
        detail = detail[detail[tr("Method", "方法")].isin(method_filter)]
    st.dataframe(detail.round(4), width="stretch", hide_index=True, height=520)
