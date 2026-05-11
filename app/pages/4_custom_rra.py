
import io
import os
import sys
import warnings

_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from _analysis import (
    make_demo_expression,
    parse_external_rank_csv,
    parse_expression_csv,
    rra_aggregate,
    score_features,
)
from _i18n import init_language, tr
from _panel_viz import make_rho_profile_figure, render_gene_panel_summary, render_plotly

lang = init_language()

BUILTIN_METHODS = {
    "ttest": "Welch t-test",
    "anova": "ANOVA F-test",
    "mi": "Mutual Information",
    "mrmr": "mRMR",
    "relieff": "ReliefF",
    "l1": "L1 Logistic",
    "elasticnet": "ElasticNet Logistic",
    "extratrees": "ExtraTrees importance",
}
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
EXTERNAL_VALUE_MODES = {
    "auto": "Auto-detect ranks from column names",
    "score": "Scores: higher values are better",
    "rank": "Ranks: smaller values are better",
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


def external_value_mode_label(mode):
    labels = {
        "auto": tr("Auto-detect ranks from column names", "根据列名自动识别排名列"),
        "score": tr("Scores: higher values are better", "分数：数值越大越好"),
        "rank": tr("Ranks: smaller values are better", "排名：数值越小越好"),
    }
    return labels[mode]

st.title(tr("⚙️ Custom RRA & Gene Panel Explorer", "⚙️ 自定义 RRA 与基因面板探索"))
st.markdown(
    tr(
        "Combine rankings from built-in feature-selection methods and optional external score/rank files, then inspect the selected panel with exploratory biological summaries.",
        "融合内置特征选择方法和可选外部分数/排名文件，并对选中的候选基因面板进行探索性生物学概览。",
    )
)

st.info(
    tr(
        "This page is intended for exploratory analysis. Online enrichment uses a small built-in pathway dictionary for visualization and should not replace database-backed biological interpretation.",
        "本页用于探索分析。在线富集使用小型内置通路词典生成可视化，不替代基于专业数据库的生物学解释。",
    )
)

with st.sidebar:
    st.header(tr("Data and ranking inputs", "数据与排序输入"))
    data_source = st.radio(
        tr("Data source", "数据来源"),
        options=["demo", "upload"],
        format_func=lambda x: tr("Built-in demo dataset", "内置示例数据") if x == "demo" else tr("Upload my own CSV", "上传我的 CSV"),
        index=0,
    )
    k = st.slider(tr("Panel size k", "面板基因数 k"), 3, 100, 20)
    selected_methods = st.multiselect(
        tr("Built-in ranking methods", "内置排序方法"),
        options=list(BUILTIN_METHODS.keys()),
        default=["ttest", "anova", "mi"],
        format_func=lambda x: BUILTIN_METHODS[x],
    )
    if any(m in selected_methods for m in ["mrmr", "relieff"]):
        st.warning(tr("mRMR and ReliefF are advanced methods and may be slower or unavailable in some online deployments.", "mRMR 和 ReliefF 属于高级方法，在部分线上部署中可能较慢或不可用。"))
    run_cv = st.checkbox(tr("Estimate discovery AUC by 5-fold CV", "用 5 折 CV 估计发现集 AUC"), value=True)
    st.divider()
    st.subheader(tr("Biological preview", "生物学预览"))
    enrichment_mode = st.selectbox(
        tr("Enrichment mode", "富集模式"),
        options=list(ENRICHMENT_MODES.keys()),
        format_func=enrichment_mode_label,
        key="rra_enrichment_mode",
    )
    gprofiler_confirmed = False
    custom_gmt_bytes = None
    if enrichment_mode == "gprofiler":
        st.caption(tr("Only selected gene symbols will be sent to g:Profiler; the expression matrix is not sent.", "只会向 g:Profiler 发送入选基因名，不发送表达矩阵。"))
        gprofiler_confirmed = st.checkbox(tr("I agree to use the external g:Profiler API", "我同意使用外部 g:Profiler API"), value=False, key="rra_gprofiler_confirm")
    elif enrichment_mode == "custom_gmt":
        gmt_file = st.file_uploader(tr("Upload gene-set GMT", "上传基因集 GMT"), type=["gmt", "txt"], key="rra_gmt")
        custom_gmt_bytes = gmt_file.getvalue() if gmt_file is not None else None
    heatmap_ordering = st.selectbox(
        tr("Heatmap sample ordering", "热图样本排序"),
        options=list(HEATMAP_ORDERINGS.keys()),
        format_func=heatmap_ordering_label,
        index=0,
        key="rra_heatmap_ordering",
    )
    st.divider()
    st.subheader(tr("External score/ranking files", "外部分数/排名文件"))
    st.caption(
        tr(
            "Upload one or more CSV files. Each file can contain one or multiple numeric ranking/score columns. If a feature/gene/id column is present, values are aligned by feature name; otherwise rows must follow the same feature order as the expression matrix.",
            "可上传一个或多个 CSV 文件；每个文件可包含一列或多列数值排名/分数。若包含 feature/gene/id 列，则按特征名对齐；否则必须与表达矩阵特征顺序一致。",
        )
    )
    external_value_mode = st.radio(
        tr("External value interpretation", "外部数值解释方式"),
        options=list(EXTERNAL_VALUE_MODES.keys()),
        format_func=external_value_mode_label,
        index=0,
        horizontal=False,
    )
    with st.expander(tr("Supported external ranking formats", "支持的外部排名格式"), expanded=False):
        st.markdown(
            tr(
                "- **Same-order scores/ranks**: a CSV with one or more numeric columns and exactly the same number/order of rows as the expression features.\n- **Feature-aligned scores/ranks**: a CSV with a `feature`, `gene`, `gene_symbol`, `probe`, `id`, or `name` column plus one or more numeric columns.\n- In auto mode, columns whose names contain `rank`, `ranking`, `order`, or `position` are interpreted as ranks where smaller is better; other numeric columns are interpreted as scores where larger is better.\n- Missing feature-aligned values are assigned the worst score so partial top-ranked lists can still be aggregated.",
                "- **同顺序分数/排名**：CSV 含一列或多列数值列，行数和顺序必须与表达矩阵特征完全一致。\n- **按特征名对齐分数/排名**：CSV 含 `feature`、`gene`、`gene_symbol`、`probe`、`id` 或 `name` 列，以及一列或多列数值列。\n- 自动模式下，列名包含 `rank`、`ranking`、`order` 或 `position` 会被视为排名（数值越小越好）；其他数值列视为分数（数值越大越好）。\n- 按特征名对齐时，缺失特征会被赋予最差分数，因此只包含 top 特征的部分排名也可以参与融合。",
            )
        )
    external_files = st.file_uploader(
        tr(
            "Optional CSV files with external scores or ranks",
            "可选上传外部分数或排名 CSV",
        ),
        type=["csv"],
        accept_multiple_files=True,
        key="external_scores",
    )
    external_files = external_files or []

if data_source == "demo":
    df = make_demo_expression()
    X, y, feature_names, label_col, raw_df = parse_expression_csv(df.to_csv(index=False).encode())
    st.success(tr("Loaded built-in demo dataset with gene symbols.", "已载入带基因名的内置示例数据。"))
else:
    st.subheader(tr("1 · Upload expression data", "1 · 上传表达数据"))
    data_file = st.file_uploader(
        tr(
            "Expression CSV: rows are samples, feature columns are genes/probes, final column is the binary label",
            "表达矩阵 CSV：行为样本，特征列为基因/探针，最后一列为二分类标签",
        ),
        type=["csv"],
        key="data_upload",
    )
    if data_file is None:
        st.info(tr("Upload a CSV or enable the built-in demo dataset.", "请上传 CSV，或启用内置示例数据。"))
        st.stop()
    try:
        X, y, feature_names, label_col, raw_df = parse_expression_csv(data_file.read())
    except Exception as exc:
        st.error(tr(f"Could not parse expression CSV: {exc}", f"表达矩阵解析失败：{exc}"))
        st.stop()

n_samples, n_features = X.shape
st.subheader(tr("1 · Dataset", "1 · 数据集"))
cols = st.columns(4)
cols[0].metric(tr("Samples", "样本数"), n_samples)
cols[1].metric(tr("Features", "特征数"), n_features)
cols[2].metric(tr("Label column", "标签列"), label_col)
cols[3].metric(tr("External files", "外部文件"), len(external_files))

with st.expander(tr("Preview data", "预览数据"), expanded=False):
    st.dataframe(raw_df.head(10), width="stretch")

if len(selected_methods) == 0 and len(external_files) == 0:
    st.warning(tr("Select built-in methods, upload external rankings, or both.", "请选择内置方法、上传外部排名，或两者同时使用。"))
    st.stop()

score_map = {}
score_errors = []
external_meta_tables = []
with st.spinner(tr("Computing feature rankings...", "正在计算特征排序...")):
    for method in selected_methods:
        try:
            score_map[BUILTIN_METHODS[method]] = score_features(method, X, y)
        except Exception as exc:
            score_errors.append(f"{BUILTIN_METHODS[method]}: {exc}")

for file in external_files:
    try:
        parsed_scores, parsed_meta = parse_external_rank_csv(
            file.getvalue(),
            n_features=n_features,
            feature_names=feature_names,
            value_mode=external_value_mode,
        )
        base_name = os.path.splitext(file.name)[0]
        for col_name, score in parsed_scores.items():
            score_map[f"External: {base_name} / {col_name}"] = score
        parsed_meta.insert(0, "file", file.name)
        external_meta_tables.append(parsed_meta)
    except Exception as exc:
        score_errors.append(f"{file.name}: {exc}")

for err in score_errors:
    st.warning(err)

if len(score_map) < 2:
    st.error(tr("Fewer than two valid ranking inputs are available after parsing.", "解析后可用排序输入少于两个。"))
    st.stop()

if external_meta_tables:
    with st.expander(tr("Parsed external ranking inputs", "已解析的外部排序输入"), expanded=True):
        st.dataframe(pd.concat(external_meta_tables, ignore_index=True), width="stretch", hide_index=True)

st.subheader(tr("2 · Ranking inputs", "2 · 排序输入"))
rank_cols = st.columns(min(4, len(score_map)))
for idx, (name, scores) in enumerate(score_map.items()):
    top = [feature_names[i] for i in np.argsort(-np.asarray(scores))[:5]]
    with rank_cols[idx % len(rank_cols)]:
        st.metric(name, tr("Top genes", "Top 基因"))
        st.caption(", ".join(top))

st.subheader(tr("3 · RRA consensus panel", "3 · RRA 共识面板"))
try:
    sel_idx, rho_scores, input_names = rra_aggregate(score_map, k=k)
except Exception as exc:
    st.error(tr(f"RRA failed: {exc}", f"RRA 失败：{exc}"))
    st.stop()

sel_names = [feature_names[i] for i in sel_idx]
result_df = pd.DataFrame(
    {
        "rank": np.arange(1, len(sel_idx) + 1),
        "feature": sel_names,
        "original_index": sel_idx,
        "rho_score": rho_scores[sel_idx],
    }
)

metric_cols = st.columns(3)
metric_cols[0].metric(tr("Selected features", "选中特征"), len(sel_names))
metric_cols[1].metric(tr("RRA inputs", "RRA 输入"), len(input_names))
metric_cols[2].metric(tr("Best ρ-score", "最佳 ρ 分数"), f"{result_df['rho_score'].min():.2e}")

if run_cv:
    X_sel = X[:, sel_idx]
    cv = StratifiedKFold(n_splits=min(5, np.bincount(y.astype(int)).min()), shuffle=True, random_state=42)
    clf = LogisticRegression(C=1, solver="liblinear", max_iter=500, random_state=42)
    try:
        aucs = cross_val_score(clf, X_sel, y, cv=cv, scoring="roc_auc")
        st.metric(tr("Discovery AUC estimate", "发现集 AUC 估计"), f"{aucs.mean():.4f}", f"± {aucs.std():.4f}")
    except Exception as exc:
        st.warning(tr(f"CV estimate failed: {exc}", f"CV 估计失败：{exc}"))

left, right = st.columns([1.05, 0.95])
with left:
    st.dataframe(result_df, width="stretch", height=420)
    st.download_button(
        tr("⬇ Download RRA panel CSV", "⬇ 下载 RRA 面板 CSV"),
        data=result_df.to_csv(index=False).encode(),
        file_name=f"clinifs_rra_panel_k{len(sel_names)}.csv",
        mime="text/csv",
        on_click="ignore",
    )
with right:
    render_plotly(make_rho_profile_figure(rho_scores, len(sel_names)), f"clinifs_rra_rho_profile_k{len(sel_names)}.html")

st.subheader(tr("4 · Biological interpretation preview", "4 · 生物学解释预览"))
render_gene_panel_summary(
    raw_df,
    sel_names,
    feature_names,
    label_col=label_col,
    prefix=f"clinifs_rra_k{len(sel_names)}",
    enrichment_mode=enrichment_mode,
    custom_gmt_bytes=custom_gmt_bytes,
    gprofiler_confirmed=gprofiler_confirmed,
    heatmap_ordering=heatmap_ordering,
)

st.subheader(tr("5 · External methods and integration", "5 · 外部方法与集成"))
st.markdown(
    tr(
        "Published methods that require separate repositories or specialized workflows can be used by exporting their feature scores or ranks and uploading them above as external ranking inputs. A single uploaded CSV may contain several method columns, so method authors can share one compact file for RRA aggregation.",
        "依赖独立仓库或专项流程的已发表方法，可先在本地导出特征分数或排名，再在本页作为外部排序输入上传。单个上传 CSV 可以包含多个方法列，因此方法作者可以用一个紧凑文件参与 RRA 融合。",
    )
)
