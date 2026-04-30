# clinifs Web Platform

Interactive Streamlit application accompanying the paper:

> *Multi-dimensional benchmarking of feature selection methods for clinical small-panel cancer gene-expression classification*, Briefings in Bioinformatics, 2026.

**Public URL**: https://clinifs.streamlit.app *(deployed; replace with actual URL after deployment)*  
**Source**: `platform/app/`

---

## Pages

| Page | Description |
|---|---|
| 🔬 **Run Analysis** | Upload your gene expression CSV, select a method and panel size k, run feature selection in-browser. No data leaves your machine. |
| 📊 **Browse Results** | Explore the paper's 1 155 pre-computed benchmark evaluation units (AUC, Nogueira Φ) by method, dataset, and difficulty tier. |
| 🎯 **Get Recommendation** | Answer 2–3 questions about your clinical scenario and receive an evidence-based method recommendation (backed by §3.6 of the paper). |
| ⚙️ **Custom RRA** | Interactively configure RRA(ANOVA+MI) or RRA(ANOVA+MI+MEL) and inspect the ρ-score distribution on your own data. |

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

`app/data/` contains pre-computed CSV summaries from the paper benchmark. These files are read-only and do not require the original GEO datasets.

| File | Description |
|---|---|
| `summary_all.csv` | Per-method × per-dataset × per-k AUC and Φ summary |
| `method_by_tier_auc.csv` | Method × tier AUC table |
| `method_by_tier_stability.csv` | Method × tier Φ table |
| `four_dim_raw.csv` | Four-dimensional evaluation raw values (used for radar chart) |
| `dataset_difficulty.csv` | 11 datasets with tier label and median AUC |
| `e2_long_enriched.csv` | Enriched long-form results |

---

## Deployment

See [`deploy_streamlit_cloud.md`](deploy_streamlit_cloud.md) for step-by-step Streamlit Community Cloud deployment instructions.
