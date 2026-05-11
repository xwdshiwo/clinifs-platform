import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _analysis import enrichment_table, enrichment_table_from_gene_sets, expression_heatmap_frame, gprofiler_enrichment, network_edges, parse_gmt
from _i18n import tr

PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "clinifs_figure",
        "height": 900,
        "width": 1400,
        "scale": 2,
    },
}


def figure_download(fig, filename, label=None):
    st.download_button(
        label or tr("⬇ Download interactive figure HTML", "⬇ 下载交互式图表 HTML"),
        data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
        file_name=filename,
        mime="text/html",
        key=f"download_{filename}",
        on_click="ignore",
    )


def render_plotly(fig, filename):
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    st.caption(tr("Use the camera icon in the chart toolbar to download a PNG snapshot.", "可使用图表工具栏中的相机按钮下载 PNG 图片。"))
    figure_download(fig, filename)


def make_rho_profile_figure(rho_scores, selected_count):
    rho_plot = pd.DataFrame({"rank": np.arange(1, len(rho_scores) + 1), "rho_score": np.sort(rho_scores)})
    fig = px.line(
        rho_plot.head(min(200, len(rho_plot))),
        x="rank",
        y="rho_score",
        title=tr("RRA ρ-score profile", "RRA ρ 分数分布"),
        labels={"rank": tr("Consensus rank", "共识排序"), "rho_score": "ρ-score"},
    )
    fig.add_vline(x=selected_count, line_dash="dash", line_color="#E86254", annotation_text=f"k={selected_count}")
    fig.update_layout(height=420)
    return fig


def make_network_figure(node_df, edge_df):
    term_ids = node_df[node_df["type"] == "term"]["id"].tolist()
    gene_ids = node_df[node_df["type"] == "gene"]["id"].tolist()
    positions = {}
    for i, term in enumerate(term_ids):
        angle = 2 * np.pi * i / max(1, len(term_ids))
        positions[term] = (0.35 * np.cos(angle), 0.35 * np.sin(angle))
    for i, gene in enumerate(gene_ids):
        angle = 2 * np.pi * i / max(1, len(gene_ids))
        positions[gene] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))
    edge_x, edge_y = [], []
    for _, row in edge_df.iterrows():
        x0, y0 = positions[row["source"]]
        x1, y1 = positions[row["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#B0B7C3"), hoverinfo="skip"))
    for node_type, color, size in [("term", "#E86254", 22), ("gene", "#4C809E", 14)]:
        sub = node_df[node_df["type"] == node_type]
        fig.add_trace(
            go.Scatter(
                x=[positions[i][0] for i in sub["id"]],
                y=[positions[i][1] for i in sub["id"]],
                text=sub["id"],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=size, color=color, line=dict(width=1, color="white")),
                name=tr("Gene sets" if node_type == "term" else "Genes", "基因集" if node_type == "term" else "基因"),
            )
        )
    fig.update_layout(
        title=tr("Gene-set membership network", "基因集成员网络"),
        height=560,
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def render_gene_panel_summary(
    raw_df,
    selected_genes,
    feature_names,
    label_col="label",
    prefix="online",
    enrichment_mode="built_in",
    custom_gmt_bytes=None,
    gprofiler_confirmed=False,
    heatmap_ordering="label_panel_mean",
):
    st.subheader(tr("Exploratory gene-set summary", "探索性基因集概览"))
    if enrichment_mode == "gprofiler":
        st.caption(
            tr(
                "g:Profiler mode sends only the selected gene symbols to the external g:Profiler service. The full expression matrix is not sent. If the API is unavailable, use the built-in or custom GMT mode.",
                "g:Profiler 模式只会把入选基因名发送到外部 g:Profiler 服务，不发送完整表达矩阵。若 API 不可用，请使用内置或自定义 GMT 模式。",
            )
        )
        if not gprofiler_confirmed:
            st.warning(tr("Confirm external API use to run g:Profiler enrichment.", "请确认允许使用外部 API 后再运行 g:Profiler 富集。"))
            enrich_df = enrichment_table(selected_genes, background_genes=feature_names)
        else:
            try:
                enrich_df = gprofiler_enrichment(selected_genes)
            except Exception as exc:
                st.warning(tr(f"g:Profiler request failed, falling back to built-in enrichment: {exc}", f"g:Profiler 请求失败，已回退到内置富集：{exc}"))
                enrich_df = enrichment_table(selected_genes, background_genes=feature_names)
    elif enrichment_mode == "custom_gmt":
        st.caption(
            tr(
                "Custom GMT mode uses your uploaded gene-set file with the same hypergeometric test and BH-FDR correction. Please ensure you have the right to use the uploaded gene sets.",
                "自定义 GMT 模式使用你上传的基因集文件，并沿用超几何检验和 BH-FDR 校正。请确保你有权使用上传的基因集。",
            )
        )
        if custom_gmt_bytes is None:
            st.warning(tr("Upload a GMT file or switch to the built-in enrichment mode.", "请上传 GMT 文件，或切换到内置富集模式。"))
            enrich_df = enrichment_table(selected_genes, background_genes=feature_names)
        else:
            try:
                gene_sets = parse_gmt(custom_gmt_bytes)
                enrich_df = enrichment_table_from_gene_sets(selected_genes, gene_sets, background_genes=feature_names, source="custom GMT")
            except Exception as exc:
                st.warning(tr(f"Could not parse GMT, falling back to built-in enrichment: {exc}", f"GMT 解析失败，已回退到内置富集：{exc}"))
                enrich_df = enrichment_table(selected_genes, background_genes=feature_names)
    else:
        st.caption(
            tr(
                "The default online enrichment is computed in the current session from a small built-in cancer-related gene-set dictionary. No external enrichment package or offline database bundle is downloaded.",
                "默认在线富集是在当前会话中基于小型内置癌症相关基因集词典计算；没有下载外部富集离线包或数据库包。",
            )
        )
        enrich_df = enrichment_table(selected_genes, background_genes=feature_names)
    st.caption(
        tr(
            "The heatmap z-scores each selected gene. The current sample ordering can be changed in the analysis settings.",
            "热图会对每个入选基因做 z-score；当前样本排序方式可在分析设置中调整。",
        )
    )
    show_enrich = enrich_df[enrich_df["overlap"] > 0].head(10)
    st.download_button(
        tr("⬇ Download enrichment table CSV", "⬇ 下载富集结果 CSV"),
        data=enrich_df.to_csv(index=False).encode(),
        file_name=f"{prefix}_enrichment_table.csv",
        mime="text/csv",
        on_click="ignore",
    )
    if show_enrich.empty:
        st.info(tr("No selected genes matched the current gene-set source.", "选中基因未匹配到当前基因集来源。"))
    else:
        e1, e2 = st.columns([1.0, 1.0])
        with e1:
            fig_dot = px.scatter(
                show_enrich.sort_values("neg_log10_p"),
                x="neg_log10_p",
                y="term",
                size="overlap",
                color="fdr_bh",
                color_continuous_scale="RdBu_r",
                title=tr("Gene-set enrichment overview", "基因集富集概览"),
                labels={"neg_log10_p": "-log10(p)", "term": tr("Gene set", "基因集"), "overlap": tr("Overlap", "重叠数"), "fdr_bh": "FDR"},
            )
            fig_dot.update_layout(height=480, yaxis_title=None)
            render_plotly(fig_dot, f"{prefix}_enrichment_dotplot.html")
        with e2:
            visible_cols = [c for c in ["term", "source", "overlap", "set_size", "p_value", "fdr_bh", "genes"] if c in show_enrich.columns]
            st.dataframe(show_enrich[visible_cols].round(4), width="stretch", height=480)
    heat_df = expression_heatmap_frame(raw_df, selected_genes, label_col=label_col, max_genes=25, ordering=heatmap_ordering)
    if not heat_df.empty:
        st.download_button(
            tr("⬇ Download heatmap matrix CSV", "⬇ 下载热图矩阵 CSV"),
            data=heat_df.to_csv(index=False).encode(),
            file_name=f"{prefix}_heatmap_long_table.csv",
            mime="text/csv",
            on_click="ignore",
        )
        fig_heat = px.imshow(
            heat_df.pivot(index="gene", columns="sample", values="z_score"),
            color_continuous_scale="RdBu_r",
            zmin=-2.5,
            zmax=2.5,
            title=tr("Selected panel expression heatmap", "选中面板表达热图"),
            labels={"x": tr("Samples", "样本"), "y": tr("Genes", "基因"), "color": "z-score"},
        )
        fig_heat.update_layout(height=520)
        render_plotly(fig_heat, f"{prefix}_expression_heatmap.html")
    node_df, edge_df = network_edges(enrich_df, selected_genes, top_terms=5)
    if not show_enrich.empty and not node_df.empty and not edge_df.empty:
        fig_net = make_network_figure(node_df, edge_df)
        render_plotly(fig_net, f"{prefix}_geneset_network.html")
        st.download_button(
            tr("⬇ Download network edges CSV", "⬇ 下载网络边表 CSV"),
            data=edge_df.to_csv(index=False).encode(),
            file_name=f"{prefix}_network_edges.csv",
            mime="text/csv",
            on_click="ignore",
        )
    return enrich_df
