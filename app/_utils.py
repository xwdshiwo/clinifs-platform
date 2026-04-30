"""
Shared utilities for clinifs Streamlit app.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Display names ────────────────────────────────────────────────────────────

METHOD_DISPLAY = {
    "anova":        "ANOVA",
    "mi":           "Mutual Info",
    "mrmr":         "mRMR",
    "relieff":      "ReliefF",
    "variance":     "Variance",
    "l1_logistic":  "L1-Logistic",
    "elasticnet":   "ElasticNet",
    "linearsvc_l1": "LinSVC-L1",
    "boruta":       "BorutaPy",
    "extratrees":   "ExtraTrees",
    "rfecv":        "RFECV",
    "ga":           "GA",
    "bpso":         "BPSO",
    "mel":          "MEL",
    "sfe":          "SFE",
}

METHOD_FAMILY = {
    "anova":        "Filter",
    "mi":           "Filter",
    "mrmr":         "Filter",
    "relieff":      "Filter",
    "variance":     "Filter",
    "l1_logistic":  "Embedded",
    "elasticnet":   "Embedded",
    "linearsvc_l1": "Embedded",
    "boruta":       "Wrapper",
    "extratrees":   "Wrapper",
    "rfecv":        "Wrapper",
    "ga":           "Wrapper",
    "bpso":         "Wrapper",
    "mel":          "Wrapper",
    "sfe":          "Wrapper",
}

FAMILY_COLOR = {
    "Filter":   "#4C9BE8",
    "Embedded": "#67B99A",
    "Wrapper":  "#E07070",
}

TIER_COLOR = {
    "easy":   "#4CAF50",
    "medium": "#FF9800",
    "hard":   "#F44336",
}

DATASET_CANCER = {
    "Bladder_GSE31189":        "Bladder",
    "Breast_GSE70947":         "Breast",
    "Colorectal_GSE44076":     "Colorectal",
    "Colorectal_GSE44861":     "Colorectal (val)",
    "Leukemia_GSE63270":       "Leukemia",
    "Liver_GSE14520_U133A":    "Liver",
    "Liver_GSE76427":          "Liver (val)",
    "Lung_GSE19804":           "Lung",
    "Pancreatic_GSE16515":     "Pancreatic",
    "Prostate_GSE6919_U95Av2": "Prostate",
    "Renal_GSE53757":          "Renal",
}

# ─── Data loaders ────────────────────────────────────────────────────────────

def _path(fname):
    return os.path.join(DATA_DIR, fname)


def load_tier_auc():
    df = pd.read_csv(_path("method_by_tier_auc.csv"))
    df["display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["family"]  = df["method"].map(METHOD_FAMILY).fillna("Other")
    return df


def load_tier_stability():
    df = pd.read_csv(_path("method_by_tier_stability.csv"))
    df["display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["family"]  = df["method"].map(METHOD_FAMILY).fillna("Other")
    return df


def load_four_dim():
    df = pd.read_csv(_path("four_dim_raw.csv"), index_col=0)
    df.index.name = "method"
    df = df.reset_index()
    df["display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["family"]  = df["method"].map(METHOD_FAMILY).fillna("Other")
    return df


def load_dataset_difficulty():
    df = pd.read_csv(_path("dataset_difficulty.csv"))
    df["cancer"] = df["dataset"].map(DATASET_CANCER).fillna(df["dataset"])
    return df


def load_summary_all():
    df = pd.read_csv(_path("summary_all.csv"))
    df["display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["dataset_short"] = (
        df["dataset"]
        .str.replace(".csv", "", regex=False)
        .map(DATASET_CANCER)
        .fillna(df["dataset"].str.replace(".csv", "", regex=False))
    )
    return df
