
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from _i18n import init_language, tr

lang = init_language()

st.title(tr("🎯 Method Guide", "🎯 方法指南"))
st.markdown(
    tr(
        "This page explains how to choose between online methods, advanced methods, RRA aggregation and external score uploads. It is an educational guide, not a data-driven recommendation engine because it does not inspect an uploaded dataset.",
        "本页说明如何在在线方法、高级方法、RRA 融合和外部分数上传之间选择。它是方法说明页，不是数据驱动推荐引擎，因为本页不会读取用户上传数据。",
    )
)

st.info(
    tr(
        "For data-specific assessment, use Online Analysis first. A future data-driven guide can compute sample size, feature count, class balance, ranking agreement, stability and difficulty from an uploaded matrix.",
        "如果需要针对具体数据评估，请先使用在线分析页面。未来的数据驱动指南可以从上传矩阵计算样本量、特征数、类别比例、排序一致性、稳定性和数据难度。",
    )
)

st.subheader(tr("1 · Recommended online methods", "1 · 推荐在线方法"))
recommended = pd.DataFrame(
    {
        tr("Method", "方法"): ["Consensus RRA", "Welch t-test", "ANOVA F-test", "Mutual Information", "L1 Logistic", "ElasticNet Logistic", "ExtraTrees"],
        tr("Best use", "适用场景"): [
            tr("Default robust first pass from several simple rankings", "多个简单排序的默认稳健初筛"),
            tr("Transparent two-class differential screening", "透明的二分类差异筛选"),
            tr("Fast statistical baseline", "快速统计基线"),
            tr("Nonlinear dependency screening", "非线性依赖筛选"),
            tr("Sparse embedded linear model", "稀疏嵌入式线性模型"),
            tr("Regularized embedded linear model", "正则化嵌入式线性模型"),
            tr("Tree-based nonlinear feature importance", "树模型非线性特征重要性"),
        ],
        tr("Online status", "在线状态"): [tr("Recommended", "推荐")] * 7,
    }
)
st.dataframe(recommended, width="stretch", hide_index=True)

st.subheader(tr("2 · Advanced online methods", "2 · 高级在线方法"))
advanced = pd.DataFrame(
    {
        tr("Method", "方法"): ["mRMR", "ReliefF"],
        tr("Why separated", "为何单独列出"): [
            tr("Heavier dependency chain and may be slower on high-dimensional matrices", "依赖链更重，高维矩阵上可能较慢"),
            tr("Older package and may be sensitive to deployment environment compatibility", "包较旧，可能受部署环境兼容性影响"),
        ],
        tr("Fallback", "替代方案"): [
            tr("Use RRA or upload local mRMR scores", "使用 RRA 或上传本地 mRMR 分数"),
            tr("Use ExtraTrees/RRA or upload local ReliefF scores", "使用 ExtraTrees/RRA 或上传本地 ReliefF 分数"),
        ],
    }
)
st.dataframe(advanced, width="stretch", hide_index=True)

st.subheader(tr("3 · External score upload", "3 · 外部分数上传"))
st.markdown(
    tr(
        "Complex methods such as Boruta, RFECV, GA, BPSO, MEL, SFE or specialized third-party workflows are better run locally. Export one score per feature in the same feature order as the expression matrix, then upload the score file in RRA & Gene Panel.",
        "Boruta、RFECV、GA、BPSO、MEL、SFE 或复杂第三方流程更适合本地运行。请按表达矩阵相同特征顺序导出每个特征一个分数，再在 RRA 与基因面板页面上传。",
    )
)

st.subheader(tr("4 · Enrichment choices", "4 · 富集选择"))
enrichment = pd.DataFrame(
    {
        tr("Mode", "模式"): ["Built-in lightweight", "g:Profiler API", "Custom GMT", "Reactome offline later"],
        tr("Use when", "何时使用"): [
            tr("Fast preview without external requests", "无需外部请求的快速预览"),
            tr("You accept sending selected gene symbols to an external service", "可接受向外部服务发送入选基因名"),
            tr("You have a licensed or custom gene-set file", "已有合规或自定义基因集文件"),
            tr("Future safer embedded pathway database option", "未来更安全的内置通路数据库选项"),
        ],
        tr("License note", "许可说明"): [
            tr("Small platform dictionary", "平台小型词典"),
            tr("External service; cite g:Profiler", "外部服务；引用 g:Profiler"),
            tr("User is responsible for gene-set license", "用户负责基因集许可"),
            tr("Reactome data are suitable for later CC0-based embedding", "Reactome 数据适合后续基于 CC0 内置"),
        ],
    }
)
st.dataframe(enrichment, width="stretch", hide_index=True)

st.subheader(tr("5 · Practical workflow", "5 · 实用流程"))
st.markdown(
    tr(
        "Start with Online Analysis using a recommended method or default RRA. If you already have results from specialized algorithms, move to RRA & Gene Panel and upload external scores. Treat discovery AUC and enrichment as exploratory outputs until independent validation is available.",
        "建议先在在线分析页面使用推荐方法或默认 RRA。如果已有专项算法结果，则进入 RRA 与基因面板页面上传外部分数。发现集 AUC 和富集都应视为探索结果，直到获得独立验证。",
    )
)
