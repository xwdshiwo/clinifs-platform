"""
Page 3 — Method Recommendation Engine

Rule-based recommendation following the paper's two-tier scheme.
No model training needed — logic is derived from benchmark findings.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

st.title("🎯 Method Recommendation")
st.markdown(
    "Answer a few questions about your dataset to get a personalised "
    "feature selection recommendation based on **benchmark findings** "
    "across 11 cancer microarray cohorts."
)

# ─── Input form ──────────────────────────────────────────────────────────────

with st.form("recommend_form"):
    col1, col2 = st.columns(2)

    with col1:
        cancer_type = st.selectbox(
            "Cancer type (if known)",
            [
                "Not specified / other",
                "Bladder (urothelial)",
                "Breast",
                "Colorectal",
                "Leukemia / AML",
                "Liver (HCC)",
                "Lung (NSCLC)",
                "Pancreatic",
                "Prostate",
                "Renal cell carcinoma",
            ],
        )
        n_samples = st.number_input(
            "Approximate number of samples (N)",
            min_value=10, max_value=10000, value=100, step=10,
        )

    with col2:
        k_target = st.number_input(
            "Desired panel size (k)",
            min_value=3, max_value=200, value=20, step=1,
        )
        priority = st.radio(
            "Primary optimisation target",
            ["Balanced (AUC + stability)", "Max AUC (Hard datasets)", "Max stability / reproducibility", "Biological interpretability"],
            index=0,
        )
        has_mel = st.checkbox("I have MEL scores available (local run)", value=False)

    submitted = st.form_submit_button("Get Recommendation", type="primary")

if not submitted:
    st.info("Fill in the form above and click **Get Recommendation**.")
    st.stop()

# ─── Rule engine ────────────────────────────────────────────────────────────

# Known difficulty from benchmark
HARD_CANCERS   = {"Bladder (urothelial)", "Prostate"}
MEDIUM_CANCERS = {"Colorectal", "Pancreatic", "Breast", "Renal cell carcinoma"}
EASY_CANCERS   = {"Leukemia / AML", "Liver (HCC)", "Lung (NSCLC)"}

def infer_tier(cancer, n_samples):
    if cancer in HARD_CANCERS:
        return "hard"
    if cancer in EASY_CANCERS and n_samples >= 80:
        return "easy"
    if cancer in MEDIUM_CANCERS:
        return "medium"
    # Unknown cancer: infer from sample size heuristic
    if n_samples < 60:
        return "medium"
    return "unknown"

tier = infer_tier(cancer_type, n_samples)

# Priority-based recommendations
RECS = {
    ("easy",   "Balanced (AUC + stability)")                    : ("RRA (ANOVA + MI)", "rra"),
    ("easy",   "Max AUC (Hard datasets)")                       : ("ANOVA",            "anova"),
    ("easy",   "Max stability / reproducibility")               : ("ANOVA",            "anova"),
    ("easy",   "Biological interpretability")                   : ("RRA (ANOVA + MI)", "rra"),
    ("medium", "Balanced (AUC + stability)")                    : ("RRA (ANOVA + MI)", "rra"),
    ("medium", "Max AUC (Hard datasets)")                       : ("ReliefF",          "relieff"),
    ("medium", "Max stability / reproducibility")               : ("ANOVA",            "anova"),
    ("medium", "Biological interpretability")                   : ("RRA (ANOVA + MI)", "rra"),
    ("hard",   "Balanced (AUC + stability)")                    : ("RRA (ANOVA + MI)", "rra"),
    ("hard",   "Max AUC (Hard datasets)")                       : ("RRA (ANOVA + MI + MEL)" if has_mel else "RRA (ANOVA + MI)", "rra"),
    ("hard",   "Max stability / reproducibility")               : ("ANOVA",            "anova"),
    ("hard",   "Biological interpretability")                   : ("RRA (ANOVA + MI)", "rra"),
    ("unknown","Balanced (AUC + stability)")                    : ("RRA (ANOVA + MI)", "rra"),
    ("unknown","Max AUC (Hard datasets)")                       : ("ReliefF",          "relieff"),
    ("unknown","Max stability / reproducibility")               : ("ANOVA",            "anova"),
    ("unknown","Biological interpretability")                   : ("RRA (ANOVA + MI)", "rra"),
}

rec_name, rec_key = RECS.get((tier, priority), ("RRA (ANOVA + MI)", "rra"))

# ─── Warnings ────────────────────────────────────────────────────────────────

caveats = []

if n_samples < 80:
    caveats.append(
        "⚠️ **Small sample size (< 80)**: Avoid RFECV — high overfitting risk documented "
        "in Hard-tier benchmarks (val_AUC = 0.556 on Prostate). "
        "Prefer ANOVA or RRA."
    )
if tier == "hard":
    caveats.append(
        "⚠️ **Hard tier**: All 15 methods show AUC 0.61–0.70. "
        "No single method dominates — validate on an independent cohort."
    )
if rec_name == "RFECV" and n_samples < 80:
    rec_name, rec_key = "RRA (ANOVA + MI)", "rra"
    caveats.append("RFECV replaced with RRA due to small sample size.")

# ─── Output ──────────────────────────────────────────────────────────────────

st.divider()

tier_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴", "unknown": "⚪"}

col_a, col_b = st.columns([1, 2])
with col_a:
    st.metric("Inferred difficulty tier",
              f"{tier_emoji.get(tier,'')} {tier.capitalize()}")
    st.metric("Recommended method", rec_name)

with col_b:
    st.subheader("Rationale")
    rationale_map = {
        "rra":      "RRA (ANOVA + MI) is the most consistent all-rounder in this benchmark. "
                    "It combines two complementary Filter scores; in the manuscript it improves "
                    "Hard-tier AUC by +3.1 pp over single ANOVA when macro-averaged across k=3–50 "
                    "(paired Wilcoxon p<0.01 at k=3–15). Runtime is < 1 s after ANOVA pre-filter.",
        "anova":    "ANOVA achieves the highest stability (Φ = 0.776 Easy, 0.548 Medium, 0.308 Hard) "
                    "of all 15 methods, making it the top choice when panel reproducibility is paramount. "
                    "Hard-tier AUC is modest (0.6422 at k=20), trading peak performance for consistency.",
        "relieff":  "ReliefF tops the Hard-tier multi-k average AUC (0.6846 averaged over k=3–50). "
                    "At k=20 specifically, MEL leads (0.6900); ReliefF is second (0.6812). "
                    "Use ReliefF when MEL is unavailable. Runtime: ~1–6 s after ANOVA pre-filter.",
        "mrmr":     "mRMR explicitly reduces feature redundancy, producing more orthogonal "
                    "panels than ANOVA. Best for downstream multi-gene assay design.",
    }
    st.markdown(rationale_map.get(rec_key,
                "Recommended based on benchmark performance profile."))

for c in caveats:
    st.warning(c)

# ─── Quick-start code snippet ────────────────────────────────────────────────

st.divider()
st.subheader("Quick-start code")

extra_arg = ""
extra_comment = ""
if rec_name.endswith("MEL)"):
    extra_arg = ', extra_scores={"mel": mel_scores}'
    extra_comment = "# mel_scores = your local MEL score array (length = n_features)\n"

snippet = f"""
{extra_comment}from clinifs import FeatureSelector

fs = FeatureSelector(method="{rec_key}", k={k_target}{extra_arg})
fs.fit(X_train, y_train)
X_selected = fs.transform(X_test)

print("Selected features:", fs.selected_indices_)
"""
st.code(snippet.strip(), language="python")

# ─── Benchmark reference table ───────────────────────────────────────────────

st.divider()
st.subheader("Benchmark reference — Hard tier, k=20 (fixed budget, single methods)")
st.caption(
    "Source: 5×5 CV across Bladder_GSE31189 + Prostate_GSE6919_U95Av2 at k=20. "
    "RRA(ANOVA+MI) was not benchmarked at individual k points; "
    "see Browse → AUC vs k for multi-k single-method results."
)
ref_data = {
    "Method":          ["MEL",   "ReliefF", "RFECV",  "ElasticNet", "ExtraTrees","ANOVA"],
    "AUC Hard k=20":   [0.6900,  0.6812,    0.6758,   0.6689,       0.6674,      0.6422],
    "Stability (Φ)":   [0.068,   0.257,     0.029,    0.203,        0.048,       0.308],
    "Runtime (s/fit)": ["30–90", "1–6",     "1–5",    "< 1",        "< 1",       "< 1"],
    "Available online":["No",    "Yes",     "Yes",    "Yes",        "Yes",       "Yes"],
}
st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)
st.info(
    "**Note on MEL**: MEL leads Hard tier at k=10–20 but is computationally expensive "
    "(requires a separate MEL fit) and not available in the online Run tool. "
    "For online use, **ReliefF** is the strongest available single method; "
    "**RRA(ANOVA+MI)** offers the best stability-AUC balance."
)
