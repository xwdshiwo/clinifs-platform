import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from _i18n import init_language, tr

lang = init_language()

st.title(tr("❓ Help & Usage Guide", "❓ 帮助与使用说明"))
st.markdown(
    tr(
        "This page explains how to prepare input files, run online feature selection, interpret RRA outputs, and understand the exploratory biological visualizations.",
        "本页说明如何准备输入文件、运行在线特征选择、解释 RRA 输出，以及理解探索性生物学可视化。",
    )
)

st.subheader(tr("1 · Input data format", "1 · 输入数据格式"))
st.markdown(
    tr(
        "Upload a CSV file where rows are samples, feature columns are numeric gene/probe measurements, and the final column is a binary class label. String labels are automatically encoded. For biological interpretation, gene symbols are preferred over anonymous probe IDs.",
        "上传 CSV 文件：行为样本，特征列为数值型基因/探针表达，最后一列为二分类标签。字符串标签会自动编码。若要进行生物学解释，建议使用基因名而不是匿名探针 ID。",
    )
)

st.subheader(tr("2 · Online Analysis", "2 · 在线分析"))
st.markdown(
    tr(
        "Use Online Analysis when you want a quick feature panel from one dataset. Choose either the built-in demo or Upload my own CSV in the sidebar. Recommended online methods are separated from advanced methods such as mRMR and ReliefF. Results include selected features, discovery-set AUC estimate, enrichment preview, expression heatmap, network view, and downloadable tables/figures.",
        "当你希望基于一个数据集快速得到候选特征面板时，使用在线分析页面。可在侧边栏选择内置示例或上传自己的 CSV。推荐在线方法与 mRMR、ReliefF 等高级方法已分层展示。结果包括入选特征、发现集 AUC 估计、富集预览、表达热图、网络视图，以及可下载表格/图表。",
    )
)

st.subheader(tr("3 · RRA & Gene Panel", "3 · RRA 与基因面板"))
st.markdown(
    tr(
        "Use RRA & Gene Panel when you want to combine two or more ranking inputs. Built-in ranking methods can be mixed with user-provided score or rank files. You may upload multiple CSV files, and each file may contain multiple numeric method columns. Files can either follow the same feature order as the expression matrix, or include a feature/gene/id column for name-based alignment. In auto mode, columns named like rank/order/position are treated as ranks where smaller is better; other numeric columns are treated as scores where larger is better.",
        "当你希望融合两个或更多排序输入时，使用 RRA 与基因面板页面。内置排序方法可以与用户自有分数或排名文件混合。你可以上传多个 CSV 文件，每个文件也可以包含多个数值方法列。文件可以与表达矩阵特征顺序一致，也可以包含 feature/gene/id 列按名称对齐。自动模式下，列名类似 rank/order/position 的列会被视为排名（数值越小越好）；其他数值列视为分数（数值越大越好）。",
    )
)

st.subheader(tr("4 · Enrichment method", "4 · 富集分析方法"))
st.markdown(
    tr(
        "The default enrichment preview does not download an external enrichment package or offline database bundle. It uses a small built-in dictionary of cancer-related gene sets and applies a hypergeometric over-representation test with Benjamini-Hochberg FDR correction. Optional modes include g:Profiler online API and custom GMT upload. g:Profiler sends only selected gene symbols to an external service, not the full expression matrix. Custom GMT mode uses your uploaded gene-set file, so you are responsible for its license.",
        "默认富集预览不会下载外部富集软件包或离线数据库包。它使用平台内置的小型癌症相关基因集词典，并进行超几何过表达检验和 Benjamini-Hochberg FDR 校正。可选模式包括 g:Profiler 在线 API 和自定义 GMT 上传。g:Profiler 只会向外部服务发送入选基因名，不发送完整表达矩阵。自定义 GMT 模式使用你上传的基因集文件，因此需自行确保其许可合规。",
    )
)

st.subheader(tr("5 · Heatmap ordering", "5 · 热图排序"))
st.markdown(
    tr(
        "For the expression heatmap, each selected gene is z-scored across samples. The default sample order is class label followed by mean selected-panel z-score. Optional orders include label only, hierarchical clustering, and original sample order. The default order is not hierarchical clustering.",
        "表达热图会对每个入选基因在样本间做 z-score。默认样本顺序为先按类别标签，再按入选面板平均 z-score。可选排序包括仅按标签、层次聚类和原始样本顺序。默认排序不是层次聚类。",
    )
)

st.subheader(tr("6 · Figure downloads", "6 · 图片下载"))
st.markdown(
    tr(
        "Interactive figures can be downloaded as HTML using the download buttons below each chart. The Plotly toolbar also provides image export for PNG snapshots. Tables are downloadable as CSV files.",
        "交互式图表可通过每张图下方按钮下载为 HTML。Plotly 图表工具栏也提供 PNG 快照导出。结果表格可下载为 CSV 文件。",
    )
)

st.subheader(tr("7 · Interpreting limitations", "7 · 结果解释限制"))
st.markdown(
    tr(
        "Discovery AUC is an internal cross-validation estimate and should not be interpreted as external validation. Candidate panels should be validated in an independent cohort before being used for biological or translational conclusions.",
        "发现集 AUC 是内部交叉验证估计，不等同于外部验证。候选面板在用于生物学或转化结论前，应在独立队列中验证。",
    )
)

st.subheader(tr("8 · Method Guide", "8 · 方法指南"))
st.markdown(
    tr(
        "Method Guide is an educational page. It explains method tiers and external-score workflows but does not inspect uploaded data or estimate dataset difficulty. Use Online Analysis for data-specific results.",
        "方法指南是说明性页面，用于解释方法分层和外部分数流程；它不会读取上传数据，也不会估计数据难度。针对具体数据的结果请使用在线分析页面。",
    )
)
