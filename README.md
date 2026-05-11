# clinifs Web Platform / clinifs 在线平台

Interactive Streamlit application for feature selection, benchmark exploration, and lightweight gene-panel interpretation.

It provides online analysis for user-uploaded CSV files, pre-computed benchmark result exploration, RRA-based ranking aggregation, and contribution guidance for additional methods or datasets.

**Public URL**: https://clinifs-platform.streamlit.app/  
**Source**: `platform/app/`

本平台用于在线特征选择、基准结果浏览、RRA 排序融合和轻量候选基因面板解释。平台不保存上传数据；公开部署版的上传数据会在 Streamlit 会话中临时处理，如需完全本地隐私保护，请按下方命令本地运行。

---

## Pages

| Page | Description |
|---|---|
| 🔬 **Online Analysis** | Choose demo or CSV upload, select a method tier and panel size k, and run lightweight feature selection with biological preview. |
| 📊 **Benchmark Results** | Explore 1 155 pre-computed benchmark evaluation units (AUC, Nogueira Φ) by method, dataset, and difficulty tier. |
| 🎯 **Method Guide** | Understand online method tiers, external-score workflows, enrichment choices and limitations. |
| ⚙️ **RRA & Gene Panel** | Combine built-in rankings and user-provided score/rank files, then inspect the selected panel with exploratory gene-set summaries, heatmaps and network views. |
| ❓ **Help** | Learn input format, RRA usage, enrichment modes, heatmap ordering and downloads. |
| 📬 **Contact** | Find contribution routes for new algorithms, score files, datasets and validation results. |

Several published methods are displayed as benchmark reference methods. Methods that rely on separate repositories, licenses, or specialized workflows can be run locally and integrated by uploading feature-score files to the RRA & Gene Panel page.

若干已发表方法作为基准参考展示。依赖独立仓库、许可或专项流程的方法可在本地运行，再将特征分数上传到 RRA 与基因面板页面参与融合分析。

RRA & Gene Panel accepts multiple external CSV files, and each file can contain multiple numeric method columns. External files can either use the same feature order as the uploaded expression matrix or include a `feature`/`gene`/`gene_symbol`/`probe`/`id`/`name` column for name-based alignment. Auto mode treats columns named like `rank`, `ranking`, `order`, or `position` as ranks where smaller is better; other numeric columns are treated as scores where larger is better.

RRA 与基因面板页面支持上传多个外部 CSV 文件，每个文件也可以包含多个数值方法列。外部文件可以与表达矩阵特征顺序一致，也可以包含 `feature`/`gene`/`gene_symbol`/`probe`/`id`/`name` 列按名称对齐。自动模式会将列名包含 `rank`、`ranking`、`order` 或 `position` 的列视为排名（数值越小越好）；其他数值列视为分数（数值越大越好）。

Online methods are separated into recommended methods and advanced methods. Recommended methods include RRA, Welch t-test, ANOVA, Mutual Information, L1/ElasticNet Logistic and ExtraTrees. mRMR and ReliefF are advanced options and may be slower or unavailable in some online deployments.

在线方法分为推荐方法和高级方法。推荐方法包括 RRA、Welch t 检验、ANOVA、互信息、L1/ElasticNet 逻辑回归和 ExtraTrees。mRMR 和 ReliefF 属于高级选项，在部分线上部署中可能较慢或不可用。

Enrichment modes include a default built-in lightweight cancer gene-set dictionary, optional g:Profiler API use, and custom GMT upload. g:Profiler mode sends only selected gene symbols to the external service, not the expression matrix. Custom GMT users are responsible for gene-set license compliance.

富集模式包括默认内置轻量癌症基因集词典、可选 g:Profiler API 和自定义 GMT 上传。g:Profiler 模式只会把入选基因名发送到外部服务，不发送表达矩阵。自定义 GMT 用户需自行确保基因集许可合规。

---

## Local run

```bash
cd platform
pip install -r requirements.txt
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Method colours (Q1 – strict three-family)

| Family | Colour |
|---|---|
| Filter | `#4C9BE8` (blue) |
| Embedded | `#67B99A` (green) |
| Wrapper | `#E07070` (red) |

---

## Data

`app/data/` contains pre-computed CSV summaries from the benchmark. These files are read-only and do not require the original GEO datasets.

| File | Description |
|---|---|
| `summary_all.csv` | Per-method × per-dataset × per-k AUC and Φ summary |
| `method_by_tier_auc.csv` | Method × tier AUC table |
| `method_by_tier_stability.csv` | Method × tier Φ table |
| `four_dim_raw.csv` | Four-dimensional evaluation raw values (used for radar chart) |
| `dataset_difficulty.csv` | 11 datasets with tier label and median AUC |
| `e2_long_enriched.csv` | Enriched long-form results |

## Example upload files / 示例上传文件

`examples/example_upload_small_bilingual.csv` is a synthetic test file for the Online Analysis page. Rows are samples, columns are genes, and the final column is `label`.

`examples/example_mel_scores_same_order.csv` is a synthetic MEL-score-like one-column file with the same feature order as the example expression matrix. It is only for testing the upload workflow, not for biological interpretation.

`examples/example_gene_symbol_expression.csv` is a synthetic expression matrix with recognizable cancer-related gene symbols. It is used by the built-in demo in the RRA & Gene Panel page and can also be uploaded manually.

`examples/example_external_scores_extratrees.csv` is a one-column external-score example with the same feature order as `example_gene_symbol_expression.csv`.

`examples/example_external_rankings_multi.csv` is a multi-column external ranking example with a `gene` column and two method columns.

`examples/example_upload_small_bilingual.csv` 是用于测试在线上传流程的合成数据；行为样本，列为基因，最后一列为 `label`。`examples/example_mel_scores_same_order.csv` 是同顺序的一列分数文件，仅用于测试 MEL 分数上传流程，不代表真实 MEL 输出或生物学结论。`examples/example_gene_symbol_expression.csv` 是带可识别癌症相关基因名的合成表达矩阵，可用于 RRA 与基因面板页面的演示；`examples/example_external_scores_extratrees.csv` 是同特征顺序的一列外部分数示例；`examples/example_external_rankings_multi.csv` 是带 `gene` 列和两个方法列的多列外部排名示例。

---

## Deployment

See [`deploy_streamlit_cloud.md`](deploy_streamlit_cloud.md) for step-by-step Streamlit Community Cloud deployment instructions.
