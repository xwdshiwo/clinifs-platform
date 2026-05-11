
import os
import sys

_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from _analysis import make_demo_expression, parse_expression_csv, rra_aggregate, score_features, top_indices_from_scores
from _i18n import init_language, tr
from _panel_viz import make_rho_profile_figure, render_gene_panel_summary, render_plotly

lang = init_language()

RECOMMENDED_METHODS = {
    "rra_default": "Consensus RRA (t-test + ANOVA + MI)",
    "ttest": "Welch t-test",
    "anova": "ANOVA F-test",
    "mi": "Mutual Information",
    "l1": "L1 Logistic",
    "elasticnet": "ElasticNet Logistic",
    "extratrees": "ExtraTrees importance",
}
ADVANCED_METHODS = {
    "mrmr": "mRMR",
    "relieff": "ReliefF",
}
METHODS = {**RECOMMENDED_METHODS, **ADVANCED_METHODS}
ENRICHMENT_MODES = {
    "built_in": "Built-in lightweight cancer gene sets",
    "gprofiler": "g:Profiler online API",
    "custom_gmt": "Custom GMT upload",
}
HEATMAP_ORDERINGS = {
    "label_panel_mean": "By label + panel mean z-score",
    "label": "By label",
    "cluster": "Hierarchical clustering",
    "original": "Original sample order",
}


def enrichment_mode_label(mode):
    labels = {
        "built_in": tr("Built-in lightweight cancer gene sets", "内置轻量癌症基因集"),
        "gprofiler": tr("g:Profiler online API", "g:Profiler 在线 API"),
        "custom_gmt": tr("Custom GMT upload", "上传自定义 GMT"),
    }
    return labels[mode]


def heatmap_ordering_label(ordering):
    labels = {
        "label_panel_mean": tr("By label + panel mean z-score", "按标签 + 面板平均 z-score"),
        "label": tr("By label", "按标签"),
        "cluster": tr("Hierarchical clustering", "层次聚类"),
        "original": tr("Original sample order", "原始样本顺序"),
    }
    return labels[ordering]

st.title(tr("🔬 Online Feature Selection", "🔬 在线特征选择"))
st.markdown(
    tr(
        "Upload a binary expression matrix or use the built-in example to generate an exploratory ranked gene panel. The web app focuses on lightweight methods with stable Python implementations; external methods can be compared through score uploads in the RRA explorer.",
        "上传二分类表达矩阵，或使用内置示例数据，生成探索性的候选基因面板。网页端优先提供具有稳定 Python 实现的轻量方法；外部方法可在 RRA 探索页通过上传分数进行比较。",
    )
)

cards = st.columns(3)
cards[0].metric(tr("Input", "输入"), tr("CSV or demo", "CSV 或示例"), tr("last column = label", "末列为标签"))
cards[1].metric(tr("Methods", "方法"), tr("7 recommended", "7 个推荐方法"), tr("advanced methods separated", "高级方法单独列出"))
cards[2].metric(tr("Output", "输出"), tr("panel + biology preview", "面板 + 生物学预览"), tr("downloadable tables and figures", "表格与图表可下载"))

with st.expander(tr("Method availability", "方法可用性"), expanded=False):
    st.markdown(
        tr(
            "Recommended online methods are RRA, Welch t-test, ANOVA, Mutual Information, L1/ElasticNet logistic models and ExtraTrees. mRMR and ReliefF are advanced options that may be slower or unavailable in some deployments. Complex third-party methods should be run locally and integrated as external score files in **RRA & Gene Panel**.",
            "推荐在线方法包括 RRA、Welch t 检验、ANOVA、互信息、L1/ElasticNet 逻辑回归和 ExtraTrees。mRMR 与 ReliefF 属于高级选项，在部分部署环境中可能较慢或不可用。复杂第三方方法建议本地运行后，将特征分数上传到 **RRA 与基因面板** 页面进行融合。",
        )
    )

with st.sidebar:
    st.header(tr("Analysis settings", "分析设置"))
    data_source = st.radio(
        tr("Data source", "数据来源"),
        options=["demo", "upload"],
        format_func=lambda x: tr("Built-in demo dataset", "内置示例数据") if x == "demo" else tr("Upload my own CSV", "上传我的 CSV"),
        index=0,
    )
    method_group = st.radio(
        tr("Method tier", "方法层级"),
        options=["recommended", "advanced"],
        format_func=lambda x: tr("Recommended online methods", "推荐在线方法") if x == "recommended" else tr("Advanced online methods", "高级在线方法"),
        index=0,
    )
    method_options = list(RECOMMENDED_METHODS.keys()) if method_group == "recommended" else list(ADVANCED_METHODS.keys())
    method_key = st.selectbox(
        tr("Feature selection method", "特征选择方法"),
        options=method_options,
        index=0,
        format_func=lambda x: METHODS[x],
    )
    if method_group == "advanced":
        st.warning(tr("Advanced methods may be slower or unavailable on some online deployments.", "高级方法在部分线上部署环境中可能较慢或不可用。"))
    k = st.slider(tr("Panel size k", "面板基因数 k"), min_value=3, max_value=100, value=20, step=1)
    run_cv = st.checkbox(
        tr("Estimate discovery AUC by 5-fold CV", "用 5 折 CV 估计发现集 AUC"),
        value=True,
        help=tr("Exploratory discovery-set estimate; not external validation.", "探索性的发现集估计，不是外部验证。"),
    )
    st.divider()
    st.subheader(tr("Biological preview", "生物学预览"))
    enrichment_mode = st.selectbox(
        tr("Enrichment mode", "富集模式"),
        options=list(ENRICHMENT_MODES.keys()),
        format_func=enrichment_mode_label,
    )
    gprofiler_confirmed = False
    custom_gmt_bytes = None
    if enrichment_mode == "gprofiler":
        st.caption(tr("Only selected gene symbols will be sent to g:Profiler; the expression matrix is not sent.", "只会向 g:Profiler 发送入选基因名，不发送表达矩阵。"))
        gprofiler_confirmed = st.checkbox(tr("I agree to use the external g:Profiler API", "我同意使用外部 g:Profiler API"), value=False)
    elif enrichment_mode == "custom_gmt":
        gmt_file = st.file_uploader(tr("Upload gene-set GMT", "上传基因集 GMT"), type=["gmt", "txt"], key="online_gmt")
        custom_gmt_bytes = gmt_file.getvalue() if gmt_file is not None else None
    heatmap_ordering = st.selectbox(
        tr("Heatmap sample ordering", "热图样本排序"),
        options=list(HEATMAP_ORDERINGS.keys()),
        format_func=heatmap_ordering_label,
        index=0,
    )

if data_source == "demo":
    demo_df = make_demo_expression()
    X, y, feature_names, label_col, raw_df = parse_expression_csv(demo_df.to_csv(index=False).encode())
    st.success(tr("Loaded built-in demo dataset with recognizable gene symbols.", "已载入带可识别基因名的内置示例数据。"))
else:
    st.subheader(tr("1 · Upload expression matrix", "1 · 上传表达矩阵"))
    st.caption(
        tr(
            "Expected format: rows are samples, feature columns are numeric genes/probes, and the final column is a binary label.",
            "格式要求：行为样本，特征列为数值型基因/探针，最后一列为二分类标签。",
        )
    )
    uploaded = st.file_uploader(
        tr("CSV file", "CSV 文件"),
        type=["csv"],
        help=tr("String labels are automatically encoded.", "字符串标签会自动编码。"),
    )
    if uploaded is None:
        st.info(tr("Upload a CSV or enable the built-in demo dataset.", "请上传 CSV，或启用内置示例数据。"))
        st.stop()
    try:
        X, y, feature_names, label_col, raw_df = parse_expression_csv(uploaded.read())
    except Exception as exc:
        st.error(tr(f"Could not parse CSV: {exc}", f"CSV 解析失败：{exc}"))
        st.stop()

n_samples, n_features = X.shape
st.subheader(tr("1 · Dataset profile", "1 · 数据概览"))
class_counts = pd.Series(y).value_counts().sort_index()
profile_cols = st.columns(4)
profile_cols[0].metric(tr("Samples", "样本数"), n_samples)
profile_cols[1].metric(tr("Features", "特征数"), n_features)
profile_cols[2].metric(tr("Label", "标签列"), label_col)
profile_cols[3].metric(tr("Class balance", "类别比例"), " / ".join(str(int(v)) for v in class_counts.values))

with st.expander(tr("Preview input data", "预览输入数据"), expanded=False):
    st.dataframe(raw_df.head(10), width="stretch")

st.subheader(tr("2 · Run feature selection", "2 · 运行特征选择"))
if st.button(tr("▶ Run analysis", "▶ 开始分析"), type="primary"):
    effective_k = min(k, n_features)
    with st.spinner(tr(f"Running {METHODS[method_key]} with k={effective_k}...", f"正在运行 {METHODS[method_key]}，k={effective_k}...")):
        try:
            if method_key == "rra_default":
                score_map = {
                    "Welch t-test": score_features("ttest", X, y),
                    "ANOVA F-test": score_features("anova", X, y),
                    "Mutual Information": score_features("mi", X, y),
                }
                sel_idx, rho_scores, input_names = rra_aggregate(score_map, effective_k)
                result_df = pd.DataFrame(
                    {
                        "rank": range(1, len(sel_idx) + 1),
                        "feature": [feature_names[i] for i in sel_idx],
                        "original_index": sel_idx,
                        "rho_score": rho_scores[sel_idx],
                    }
                )
            else:
                scores = score_features(method_key, X, y)
                sel_idx = top_indices_from_scores(scores, effective_k)
                result_df = pd.DataFrame(
                    {
                        "rank": range(1, len(sel_idx) + 1),
                        "feature": [feature_names[i] for i in sel_idx],
                        "original_index": sel_idx,
                        "score": scores[sel_idx],
                    }
                )
        except Exception as exc:
            st.error(tr(f"Feature selection failed: {exc}", f"特征选择失败：{exc}"))
            st.stop()

    auc_text = "NA"
    if run_cv:
        X_sel = X[:, sel_idx]
        clf = LogisticRegression(C=1, solver="liblinear", max_iter=500, random_state=42)
        n_splits = min(5, int(np.bincount(y.astype(int)).min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            try:
                aucs = cross_val_score(clf, X_sel, y, cv=cv, scoring="roc_auc")
                auc_text = f"{aucs.mean():.4f} ± {aucs.std():.4f}"
                st.metric(tr("Discovery AUC estimate", "发现集 AUC 估计"), f"{aucs.mean():.4f}", f"± {aucs.std():.4f}")
            except Exception as exc:
                st.warning(tr(f"CV estimate failed: {exc}", f"CV 估计失败：{exc}"))

    st.subheader(tr("3 · Selected feature panel", "3 · 选中特征面板"))
    st.dataframe(result_df, width="stretch", height=390)

    fig = px.bar(
        result_df.head(min(30, len(result_df))),
        x="rank",
        y="feature",
        orientation="h",
        title=tr("Top selected features", "Top 入选特征"),
        labels={"rank": tr("Rank", "排序"), "feature": tr("Feature", "特征")},
    )
    fig.update_layout(height=520, yaxis={"categoryorder": "array", "categoryarray": result_df.head(min(30, len(result_df)))["feature"][::-1]})
    render_plotly(fig, f"clinifs_top_features_{method_key}_k{effective_k}.html")
    if method_key == "rra_default":
        render_plotly(make_rho_profile_figure(rho_scores, len(sel_idx)), f"clinifs_rra_rho_profile_k{effective_k}.html")

    summary = pd.DataFrame(
        {
            "item": ["method", "k", "n_samples", "n_features", "cv_auc"],
            "value": [METHODS[method_key], effective_k, n_samples, n_features, auc_text],
        }
    )
    st.download_button(
        tr("⬇ Download selected panel CSV", "⬇ 下载选中面板 CSV"),
        data=result_df.to_csv(index=False).encode(),
        file_name=f"clinifs_panel_{method_key}_k{effective_k}.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.download_button(
        tr("⬇ Download run summary CSV", "⬇ 下载运行摘要 CSV"),
        data=summary.to_csv(index=False).encode(),
        file_name=f"clinifs_summary_{method_key}_k{effective_k}.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.subheader(tr("4 · Biological interpretation preview", "4 · 生物学解释预览"))
    render_gene_panel_summary(
        raw_df,
        [feature_names[i] for i in sel_idx],
        feature_names,
        label_col=label_col,
        prefix=f"clinifs_online_{method_key}_k{effective_k}",
        enrichment_mode=enrichment_mode,
        custom_gmt_bytes=custom_gmt_bytes,
        gprofiler_confirmed=gprofiler_confirmed,
        heatmap_ordering=heatmap_ordering,
    )
