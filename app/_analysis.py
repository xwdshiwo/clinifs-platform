import io
import json
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import hypergeom, ttest_ind
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

GENE_SETS = {
    "Cell-cycle and proliferation": ["MKI67", "TOP2A", "PCNA", "CCNB1", "CCND1", "CDK1", "CDK2", "AURKA", "BIRC5", "E2F1"],
    "DNA damage and genome stability": ["TP53", "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51", "PARP1", "MDM2"],
    "EGFR/ERBB signaling": ["EGFR", "ERBB2", "ERBB3", "KRAS", "NRAS", "BRAF", "MAPK1", "MAPK3", "PIK3CA", "AKT1"],
    "PI3K-AKT-mTOR signaling": ["PIK3CA", "PIK3R1", "AKT1", "AKT2", "MTOR", "PTEN", "TSC1", "TSC2", "RPTOR", "FOXO3"],
    "Apoptosis regulation": ["BAX", "BCL2", "BCL2L1", "CASP3", "CASP8", "CASP9", "FAS", "FASLG", "TNF", "TP53"],
    "EMT and invasion": ["VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "MMP2", "MMP9"],
    "Immune checkpoint and inflammation": ["CD274", "PDCD1", "CTLA4", "LAG3", "TIGIT", "IFNG", "CXCL9", "CXCL10", "IL6", "TNF"],
    "Angiogenesis and hypoxia": ["VEGFA", "KDR", "FLT1", "HIF1A", "EPAS1", "ANGPT1", "ANGPT2", "PECAM1", "VWF", "NOS3"],
    "Hormone and nuclear receptor signaling": ["ESR1", "PGR", "AR", "FOXA1", "GATA3", "NCOA1", "NCOA2", "NR3C1", "PPARG", "RXRA"],
    "Cancer metabolism": ["MYC", "LDHA", "HK2", "PKM", "SLC2A1", "IDH1", "IDH2", "GLS", "ACLY", "FASN"],
}

DEMO_SIGNAL_GENES = ["TP53", "BRCA1", "EGFR", "ERBB2", "PIK3CA", "PTEN", "MKI67", "TOP2A", "CD274", "VEGFA", "VIM", "CDH1"]
DEMO_BACKGROUND_GENES = sorted({gene for genes in GENE_SETS.values() for gene in genes})


def make_demo_expression(random_state=7, n_per_class=36):
    rng = np.random.default_rng(random_state)
    genes = DEMO_BACKGROUND_GENES + [f"GENE_{i:03d}" for i in range(1, 61)]
    genes = list(dict.fromkeys(genes))
    n_features = len(genes)
    X0 = rng.normal(0, 1, size=(n_per_class, n_features))
    X1 = rng.normal(0, 1, size=(n_per_class, n_features))
    effects = {
        "TP53": 1.35,
        "BRCA1": 1.1,
        "EGFR": 1.25,
        "ERBB2": 1.4,
        "PIK3CA": 1.0,
        "PTEN": -1.1,
        "MKI67": 1.55,
        "TOP2A": 1.35,
        "CD274": 0.95,
        "VEGFA": 0.9,
        "VIM": 0.8,
        "CDH1": -0.85,
    }
    for gene, effect in effects.items():
        if gene in genes:
            X1[:, genes.index(gene)] += effect
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    df = pd.DataFrame(X, columns=genes)
    df["label"] = y
    return df


def parse_expression_csv(data_bytes):
    df = pd.read_csv(io.BytesIO(data_bytes))
    if df.shape[1] < 3:
        raise ValueError("CSV must contain at least two feature columns and one label column.")
    feature_cols = df.columns[:-1].tolist()
    label_col = df.columns[-1]
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_df.isna().any().any():
        bad = X_df.columns[X_df.isna().any()].tolist()[:5]
        raise ValueError(f"Non-numeric or missing feature values detected, e.g. {bad}")
    y_raw = df[label_col].values
    if y_raw.dtype.kind not in ("i", "u", "f"):
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("Only binary classification labels are supported.")
    return X_df.values.astype(np.float32), y, feature_cols, label_col, df


FEATURE_ID_COLUMN_NAMES = {"feature", "feature_id", "gene", "gene_symbol", "symbol", "probe", "probe_id", "id", "name"}
RANK_COLUMN_HINTS = ("rank", "ranking", "order", "position")


def _external_column_is_rank(column_name, value_mode):
    value_mode = value_mode.lower().strip()
    if value_mode == "rank":
        return True
    if value_mode == "score":
        return False
    name = str(column_name).lower()
    return any(hint in name for hint in RANK_COLUMN_HINTS)


def _external_values_to_scores(values, is_rank):
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No numeric values were found in the external ranking column.")
    if is_rank:
        max_rank = np.nanmax(values[finite])
        scores = max_rank + 1.0 - values
        scores[~finite] = 0.0
    else:
        min_value = np.nanmin(values[finite])
        max_value = np.nanmax(values[finite])
        fill_value = min_value - max(1.0, abs(max_value - min_value))
        scores = values.copy()
        scores[~finite] = fill_value
    return clean_scores(scores)


def parse_external_rank_csv(data_bytes, n_features, feature_names, value_mode="auto"):
    df = pd.read_csv(io.BytesIO(data_bytes))
    if df.empty:
        raise ValueError("External ranking CSV is empty.")
    normalized = {str(col).strip().lower(): col for col in df.columns}
    id_col = next((normalized[name] for name in FEATURE_ID_COLUMN_NAMES if name in normalized), None)
    candidate_cols = [col for col in df.columns if col != id_col]
    numeric_cols = []
    for col in candidate_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError("External ranking CSV must contain at least one numeric score or rank column.")

    score_map = {}
    meta_rows = []
    if id_col is not None:
        feature_ids = df[id_col].astype(str).str.strip().str.upper()
        target_ids = [str(feature).strip().upper() for feature in feature_names]
        for col in numeric_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            temp = pd.DataFrame({"feature": feature_ids, "value": numeric})
            temp = temp[(temp["feature"] != "") & temp["value"].notna()]
            duplicate_count = int(temp["feature"].duplicated().sum())
            temp = temp.drop_duplicates("feature", keep="first")
            value_by_feature = dict(zip(temp["feature"], temp["value"]))
            aligned = np.array([value_by_feature.get(feature, np.nan) for feature in target_ids], dtype=float)
            matched = int(np.isfinite(aligned).sum())
            if matched == 0:
                raise ValueError(f"{col}: no feature IDs matched the expression matrix.")
            is_rank = _external_column_is_rank(col, value_mode)
            score_map[str(col)] = _external_values_to_scores(aligned, is_rank)
            meta_rows.append(
                {
                    "column": str(col),
                    "value_type": "rank_low_is_best" if is_rank else "score_high_is_best",
                    "alignment": "feature_id",
                    "matched_features": matched,
                    "missing_features": int(n_features - matched),
                    "duplicate_feature_ids": duplicate_count,
                }
            )
    else:
        if len(df) != n_features:
            raise ValueError(
                f"No feature/gene/id column was found, so row-order alignment is required; got {len(df)} rows but expected {n_features}."
            )
        for col in numeric_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            is_rank = _external_column_is_rank(col, value_mode)
            score_map[str(col)] = _external_values_to_scores(numeric, is_rank)
            meta_rows.append(
                {
                    "column": str(col),
                    "value_type": "rank_low_is_best" if is_rank else "score_high_is_best",
                    "alignment": "row_order",
                    "matched_features": int(numeric.notna().sum()),
                    "missing_features": int(numeric.isna().sum()),
                    "duplicate_feature_ids": 0,
                }
            )
    return score_map, pd.DataFrame(meta_rows)


def clean_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return np.nan_to_num(scores, nan=0.0, posinf=np.nanmax(scores[np.isfinite(scores)]) if np.isfinite(scores).any() else 0.0, neginf=0.0)


def score_features(method, X, y, random_state=42):
    method = method.lower().strip()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if method == "ttest":
        classes = np.unique(y)
        stat, _ = ttest_ind(X[y == classes[0]], X[y == classes[1]], axis=0, equal_var=False, nan_policy="omit")
        return clean_scores(np.abs(stat))
    if method == "anova":
        scores, _ = f_classif(X, y)
        return clean_scores(scores)
    if method == "mi":
        return clean_scores(mutual_info_classif(X, y, random_state=random_state))
    if method in {"l1", "elasticnet"}:
        penalty = "l1" if method == "l1" else "elasticnet"
        l1_ratio = None if method == "l1" else 0.5
        clf = LogisticRegression(
            penalty=penalty,
            solver="saga",
            l1_ratio=l1_ratio,
            max_iter=1500,
            random_state=random_state,
            n_jobs=1,
        )
        Xs = StandardScaler().fit_transform(X)
        clf.fit(Xs, y)
        return clean_scores(np.abs(clf.coef_).ravel())
    if method == "extratrees":
        clf = ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=1, class_weight="balanced")
        clf.fit(X, y)
        return clean_scores(clf.feature_importances_)
    if method == "mrmr":
        try:
            from mrmr import mrmr_classif
        except ImportError as exc:
            raise ImportError("mrmr-selection is required for mRMR scoring.") from exc
        k_rank = min(200, X.shape[1])
        names = [f"f{i}" for i in range(X.shape[1])]
        try:
            selected = mrmr_classif(X=pd.DataFrame(X, columns=names), y=pd.Series(y), K=k_rank, show_progress=False)
        except TypeError:
            selected = mrmr_classif(X=pd.DataFrame(X, columns=names), y=pd.Series(y), K=k_rank)
        scores = np.zeros(X.shape[1])
        for rank, name in enumerate(selected):
            scores[int(name[1:])] = k_rank - rank
        return scores
    if method == "relieff":
        try:
            from skrebate import ReliefF
        except ImportError as exc:
            raise ImportError("skrebate is required for ReliefF scoring.") from exc
        n_neighbors = max(1, min(10, len(y) - 1))
        rf = ReliefF(n_features_to_select=min(200, X.shape[1]), n_neighbors=n_neighbors, n_jobs=1)
        rf.fit(X, y)
        scores = np.zeros(X.shape[1])
        scores[:] = rf.feature_importances_
        return clean_scores(scores)
    raise ValueError(f"Unknown scoring method: {method}")


def top_indices_from_scores(scores, k):
    scores = clean_scores(scores)
    return np.argsort(-scores)[: min(k, len(scores))]


def rra_aggregate(score_map, k):
    if len(score_map) < 2:
        raise ValueError("RRA requires at least two ranking inputs.")
    names = list(score_map.keys())
    lengths = {len(score_map[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError("All score arrays must have the same length.")
    n_features = lengths.pop()
    ranks = np.zeros((len(names), n_features))
    for i, name in enumerate(names):
        scores = clean_scores(score_map[name])
        order = np.argsort(-scores)
        rank_pos = np.empty_like(order)
        rank_pos[order] = np.arange(n_features)
        ranks[i] = (rank_pos + 1.0) / n_features
    sorted_r = np.sort(ranks, axis=0)
    rho = np.ones(n_features)
    n_methods = len(names)
    for ord_k in range(n_methods):
        p_k = beta_dist.cdf(sorted_r[ord_k], ord_k + 1, n_methods - ord_k)
        rho = np.minimum(rho, p_k)
    rho = np.minimum(rho * n_features, 1.0)
    selected = np.argsort(rho)[: min(k, n_features)]
    return selected, rho, names


def looks_like_gene_symbols(feature_names):
    matched = [g for g in feature_names if str(g).upper() in set(DEMO_BACKGROUND_GENES)]
    return len(matched) >= 3


def parse_gmt(data_bytes):
    text = data_bytes.decode("utf-8-sig")
    gene_sets = {}
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            term = parts[0].strip()
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            if term and genes:
                gene_sets[term] = genes
    if not gene_sets:
        raise ValueError("No valid GMT gene sets were found.")
    return gene_sets


def enrichment_table_from_gene_sets(selected_genes, gene_sets, background_genes=None, source="built-in"):
    selected = {str(g).upper() for g in selected_genes}
    if background_genes is None:
        background = {str(g).upper() for genes in gene_sets.values() for g in genes}
        background.update(selected)
    else:
        background = {str(g).upper() for g in background_genes}
    M = len(background)
    N = len(selected & background)
    rows = []
    for term, genes in gene_sets.items():
        term_genes = {g.upper() for g in genes} & background
        overlap = selected & term_genes
        if not term_genes or N == 0:
            pval = 1.0
        else:
            pval = hypergeom.sf(len(overlap) - 1, M, len(term_genes), N)
        rows.append(
            {
                "term": term,
                "source": source,
                "overlap": len(overlap),
                "set_size": len(term_genes),
                "selected_size": N,
                "p_value": pval,
                "neg_log10_p": -np.log10(max(pval, 1e-300)),
                "genes": ", ".join(sorted(overlap)) if overlap else "",
            }
        )
    out = pd.DataFrame(rows).sort_values(["p_value", "overlap"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["fdr_bh"] = np.minimum.accumulate((out["p_value"] * len(out) / out["rank"]).iloc[::-1]).iloc[::-1].clip(upper=1).values
    return out


def enrichment_table(selected_genes, background_genes=None):
    return enrichment_table_from_gene_sets(selected_genes, GENE_SETS, background_genes=background_genes, source="built-in")


def gprofiler_enrichment(selected_genes, organism="hsapiens", timeout=12):
    genes = [str(g).upper() for g in selected_genes if str(g).strip()]
    if not genes:
        return pd.DataFrame()
    payload = {
        "organism": organism,
        "query": genes,
        "sources": ["GO:BP", "REAC", "WP"],
        "user_threshold": 1.0,
        "no_evidences": False,
    }
    req = urllib.request.Request(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "clinifs-platform"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    query_genes = genes
    rows = []
    for item in result.get("result", []):
        intersections = item.get("intersections") or item.get("intersection") or []
        if isinstance(intersections, str):
            genes_text = intersections
        elif intersections and all(isinstance(x, list) for x in intersections):
            genes_text = ", ".join(gene for gene, evidence in zip(query_genes, intersections) if evidence)
        else:
            genes_text = ", ".join(str(g).upper() for g in intersections)
        pval = float(item.get("p_value", 1.0))
        rows.append(
            {
                "term": item.get("name") or item.get("native") or "g:Profiler term",
                "source": item.get("source", "g:Profiler"),
                "overlap": int(item.get("intersection_size", 0) or 0),
                "set_size": int(item.get("term_size", 0) or 0),
                "selected_size": int(item.get("query_size", len(genes)) or len(genes)),
                "p_value": pval,
                "neg_log10_p": -np.log10(max(pval, 1e-300)),
                "genes": genes_text,
                "fdr_bh": pval,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["term", "source", "overlap", "set_size", "selected_size", "p_value", "neg_log10_p", "genes", "rank", "fdr_bh"])
    out = pd.DataFrame(rows).sort_values(["p_value", "overlap"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def expression_heatmap_frame(df, selected_genes, label_col="label", max_genes=30, ordering="label_panel_mean"):
    genes = [g for g in selected_genes if g in df.columns][:max_genes]
    if not genes:
        return pd.DataFrame()
    X = df[genes].astype(float)
    z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    z[label_col] = df[label_col].values
    z["_panel_mean"] = z[genes].mean(axis=1)
    z["_original_order"] = np.arange(len(z))
    if ordering == "label":
        z = z.sort_values([label_col, "_original_order"], ascending=[True, True]).reset_index(drop=True)
    elif ordering == "original":
        z = z.sort_values(["_original_order"]).reset_index(drop=True)
    elif ordering == "cluster":
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import pdist
            if len(z) > 2:
                order = leaves_list(linkage(pdist(z[genes].values, metric="euclidean"), method="average"))
                z = z.iloc[order].reset_index(drop=True)
            else:
                z = z.sort_values(["_original_order"]).reset_index(drop=True)
        except Exception:
            z = z.sort_values([label_col, "_panel_mean", "_original_order"], ascending=[True, False, True]).reset_index(drop=True)
    else:
        z = z.sort_values([label_col, "_panel_mean", "_original_order"], ascending=[True, False, True]).reset_index(drop=True)
    z["sample"] = [f"S{i+1:02d}" for i in range(len(z))]
    z = z.drop(columns=["_panel_mean", "_original_order"])
    long = z.melt(id_vars=["sample", label_col], var_name="gene", value_name="z_score")
    return long


def network_edges(enrich_df, selected_genes, top_terms=5):
    selected = {str(g).upper() for g in selected_genes}
    terms = enrich_df.head(top_terms)
    edges = []
    nodes = []
    for _, row in terms.iterrows():
        term = row["term"]
        nodes.append({"id": term, "type": "term", "score": row["neg_log10_p"]})
        genes = [g for g in row["genes"].split(", ") if g]
        for gene in genes:
            if gene in selected:
                nodes.append({"id": gene, "type": "gene", "score": 1.0})
                edges.append((term, gene))
    node_df = pd.DataFrame(nodes).drop_duplicates("id") if nodes else pd.DataFrame(columns=["id", "type", "score"])
    edge_df = pd.DataFrame(edges, columns=["source", "target"]) if edges else pd.DataFrame(columns=["source", "target"])
    return node_df, edge_df
