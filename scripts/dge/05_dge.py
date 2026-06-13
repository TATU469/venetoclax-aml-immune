"""
Pseudobulk DGE — GSE306339, post vs pre venetoclax+AZA.

Runs PyDESeq2 per cell type where ≥3 matched patient pairs available.
Focuses on malignant myeloid progenitors (GMP, Progenitor_LSC) and
cytotoxic lymphocytes (NK_cell, CD8_T_cell).

Outputs:
  results/tables/dge/dge_{cell_type}_post_vs_pre.tsv
  results/tables/dge/dge_summary.tsv
  results/figures/dge/volcano_{cell_type}.png
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "."
PROC_DIR = os.path.join(PROJECT, "data/processed")
DGE_DIR  = os.path.join(PROJECT, "results/tables/dge")
FIG_DIR  = os.path.join(PROJECT, "results/figures/dge")
for d in [DGE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

adata = sc.read_h5ad(os.path.join(PROC_DIR, "gse306339_annotated.h5ad"))

# Restore raw counts for DGE
if adata.raw is not None:
    counts_adata = adata.raw.to_adata()
    counts_adata.obs = adata.obs.copy()
else:
    counts_adata = adata

MIN_CELLS   = 10
MIN_PAIRS   = 3
MIN_COUNTS  = 10

PRIORITY_TYPES = ["NK_cell", "CD8_T_cell", "GMP", "Progenitor_LSC", "Monocyte"]

summary_records = []

def run_dge(cell_type, adata_ct):
    log.info("--- DGE: %s ---", cell_type)

    # Pseudobulk: aggregate counts per patient × timepoint
    pb_records = []
    for (pid, tp), grp in adata_ct.obs.groupby(["patient_id", "timepoint"]):
        if len(grp) < MIN_CELLS:
            continue
        idx = grp.index
        mat = counts_adata[idx].X
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        pb_records.append({
            "patient_id": pid, "timepoint": tp,
            "n_cells": len(grp),
            "counts": mat.sum(axis=0).astype(int),
        })

    if len(pb_records) < MIN_PAIRS * 2:
        log.warning("  Skipping %s — only %d pseudobulk samples", cell_type, len(pb_records))
        return None

    pb_df = pd.DataFrame({"patient_id": [r["patient_id"] for r in pb_records],
                           "timepoint":  [r["timepoint"]  for r in pb_records],
                           "n_cells":    [r["n_cells"]    for r in pb_records]})
    count_mat = pd.DataFrame(
        np.stack([r["counts"] for r in pb_records]),
        columns=counts_adata.var_names,
        index=[f"{r['patient_id']}_{r['timepoint']}" for r in pb_records],
    )

    # Keep only patients with both pre and post
    has_both = pb_df.groupby("patient_id")["timepoint"].apply(
        lambda x: set(x) >= {"pre", "post"}
    )
    valid_patients = has_both[has_both].index.tolist()
    if len(valid_patients) < MIN_PAIRS:
        log.warning("  Skipping %s — only %d paired patients", cell_type, len(valid_patients))
        return None

    keep_idx = pb_df[pb_df["patient_id"].isin(valid_patients)].index
    count_mat = count_mat.iloc[keep_idx]
    pb_df     = pb_df.iloc[keep_idx].reset_index(drop=True)
    count_mat.index = pb_df.index

    # Filter lowly expressed genes
    gene_keep = (count_mat >= MIN_COUNTS).any(axis=0)
    count_mat = count_mat.loc[:, gene_keep]

    # PyDESeq2
    try:
        dds = DeseqDataSet(
            counts=count_mat,
            metadata=pb_df[["timepoint", "patient_id"]],
            design_factors=["patient_id", "timepoint"],
            refit_cooks=True,
        )
        dds.deseq2()
        stat = DeseqStats(dds, contrast=["timepoint", "post", "pre"])
        stat.summary()
        res = stat.results_df.reset_index().rename(columns={"index": "gene_symbol"})
        res["cell_type"] = cell_type
        res = res[res["padj"].notna()].copy()
        res = res.sort_values("padj")

        n_sig = (res["padj"] < 0.05).sum()
        n_up  = ((res["padj"] < 0.05) & (res["log2FoldChange"] > 0)).sum()
        n_dn  = ((res["padj"] < 0.05) & (res["log2FoldChange"] < 0)).sum()
        log.info("  %s: %d sig (↑%d ↓%d), n_pairs=%d",
                 cell_type, n_sig, n_up, n_dn, len(valid_patients))

        out_path = os.path.join(DGE_DIR, f"dge_{cell_type}_post_vs_pre.tsv")
        res.to_csv(out_path, sep="\t", index=False)

        summary_records.append({
            "cell_type": cell_type, "n_pseudobulk": len(count_mat),
            "n_paired_patients": len(valid_patients),
            "n_genes_tested": len(res), "n_sig_fdr05": n_sig,
            "n_up": n_up, "n_down": n_dn,
        })

        # Volcano
        plot_volcano(res, cell_type)
        return res

    except Exception as e:
        log.error("  PyDESeq2 failed for %s: %s", cell_type, e)
        return None


def plot_volcano(res, cell_type):
    res = res.copy()
    res["-log10_padj"] = -np.log10(res["padj"].clip(lower=1e-10))
    sig = res["padj"] < 0.05
    top_genes = res[sig].nsmallest(10, "padj")["gene_symbol"].tolist()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(res.loc[~sig, "log2FoldChange"], res.loc[~sig, "-log10_padj"],
               c="#bdc3c7", s=8, alpha=0.25, linewidths=0)
    ax.scatter(res.loc[sig & (res["log2FoldChange"] > 0), "log2FoldChange"],
               res.loc[sig & (res["log2FoldChange"] > 0), "-log10_padj"],
               c="#2980b9", s=25, alpha=0.8, linewidths=0, label="Up (post)")
    ax.scatter(res.loc[sig & (res["log2FoldChange"] < 0), "log2FoldChange"],
               res.loc[sig & (res["log2FoldChange"] < 0), "-log10_padj"],
               c="#e74c3c", s=25, alpha=0.8, linewidths=0, label="Down (post)")

    from matplotlib.patheffects import withStroke
    stroke = [withStroke(linewidth=2, foreground="white")]
    for _, row in res[res["gene_symbol"].isin(top_genes)].iterrows():
        ax.annotate(row["gene_symbol"],
                    xy=(row["log2FoldChange"], row["-log10_padj"]),
                    xytext=(row["log2FoldChange"] + 0.1, row["-log10_padj"] + 0.1),
                    fontsize=7, fontweight="bold",
                    path_effects=stroke, zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.5))

    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linestyle="-", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("log₂ fold change (post / pre)", fontsize=10)
    ax.set_ylabel("−log₁₀(adjusted p)", fontsize=10)
    ax.set_title(f"{cell_type} — venetoclax+AZA response\n(post vs pre)", fontsize=10)
    ax.legend(fontsize=8)
    n_sig = sig.sum()
    ax.text(0.98, 0.98, f"n = {n_sig} sig. (FDR<0.05)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"volcano_{cell_type}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# Run for priority cell types first, then others
all_types = PRIORITY_TYPES + [ct for ct in adata.obs["cell_type"].unique()
                               if ct not in PRIORITY_TYPES]

for ct in all_types:
    sub = adata[adata.obs["cell_type"] == ct]
    if sub.n_obs < MIN_CELLS * MIN_PAIRS:
        log.info("Skipping %s — too few cells (%d)", ct, sub.n_obs)
        continue
    run_dge(ct, sub)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(os.path.join(PROJECT, "results/tables"), "dge_summary.tsv"),
                  sep="\t", index=False)
log.info("\n=== DGE SUMMARY ===\n%s", summary_df.to_string(index=False))
log.info("=== DGE complete ===")
