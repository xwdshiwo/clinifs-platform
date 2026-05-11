import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from _i18n import init_language, tr

lang = init_language()

st.title(tr("📬 Contact & Contribute", "📬 联系我们与参与共建"))
st.markdown(
    tr(
        "clinifs is designed as an extensible feature-selection and benchmark exploration platform. Researchers and method authors can contribute additional algorithms, score files, datasets, or validation results.",
        "clinifs 被设计为可扩展的特征选择与基准结果探索平台。研究者和方法作者可以贡献更多算法、分数文件、数据集或验证结果。",
    )
)

st.subheader(tr("How to contribute a method", "如何贡献算法结果"))
st.markdown(
    tr(
        "If your method can output one score per feature, you can integrate it immediately through the RRA & Gene Panel page. For inclusion in benchmark result tables, please provide a reproducible script, environment requirements, method citation, and score/result files on the agreed datasets.",
        "如果你的方法能为每个特征输出一个分数，就可以直接通过 RRA 与基因面板页面集成。若希望加入基准结果表，请提供可复现实验脚本、环境依赖、方法引用，以及约定数据集上的分数/结果文件。",
    )
)

requirements = pd.DataFrame(
    {
        tr("Contribution type", "贡献类型"): [
            tr("External score file", "外部分数文件"),
            tr("New algorithm", "新算法"),
            tr("New dataset", "新数据集"),
            tr("External validation", "外部验证结果"),
        ],
        tr("Minimum information", "最低信息要求"): [
            tr("One numeric score column matching expression feature order", "一列数值分数，特征顺序与表达矩阵一致"),
            tr("Runnable code, dependency list, citation, expected input/output", "可运行代码、依赖列表、引用、输入输出格式"),
            tr("Expression matrix, binary labels, feature annotation, accession or source", "表达矩阵、二分类标签、特征注释、编号或来源"),
            tr("Selected panel, validation cohort, metric definition, evaluation script", "候选面板、验证队列、指标定义、评估脚本"),
        ],
        tr("Use in platform", "平台用途"): [
            tr("Immediate RRA integration", "立即参与 RRA 融合"),
            tr("Future online or offline benchmark module", "未来在线或离线基准模块"),
            tr("Expand benchmark coverage", "扩展基准覆盖范围"),
            tr("Assess generalization", "评估泛化能力"),
        ],
    }
)
st.dataframe(requirements, width="stretch", hide_index=True)

st.subheader(tr("Suggested contact channels", "建议联系渠道"))
st.markdown(
    tr(
        "Use the GitHub repository issue tracker for reproducible bug reports, feature requests, method integration proposals, and dataset contributions. Include a small example file whenever possible.",
        "建议通过 GitHub 仓库 issue 提交可复现 bug、功能请求、方法集成提案和数据集贡献。请尽量附带小型示例文件。",
    )
)

st.link_button("clinifs package", "https://github.com/xwdshiwo/clinifs")
st.link_button("clinifs platform", "https://github.com/xwdshiwo/clinifs-platform")
st.link_button("clinifs benchmark", "https://github.com/xwdshiwo/clinifs-benchmark")

st.subheader(tr("What we can add next", "后续可扩展方向"))
st.markdown(
    tr(
        "Potential extensions include more published feature-selection algorithms, richer external validation datasets, database-backed enrichment, survival endpoints, multi-class tasks, and downloadable report generation.",
        "后续可扩展更多已发表特征选择算法、更丰富的外部验证数据、数据库支持的富集分析、生存结局、多分类任务和可下载报告生成。",
    )
)
