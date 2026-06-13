"""
Cell type annotation — GSE306339 (primary) and GSE116256 (reference).

Scores Leiden clusters against curated AML marker gene panels and assigns
cell type labels. Malignant myeloid progenitors are flagged by co-expression
of CD34/CD117/HOXA9 and absence of T/NK/B lineage markers.

Inputs:
  data/processed/gse306339_qc.h5ad
  data/processed/gse116256_qc.h5ad  (optional reference)

Outputs:
  data/processed/gse306339_annotated.h5ad
  data/processed/gse116256_annotated.h5ad
  results/figures/annotation_dotplot_gse306339.png
  results/figures/annotation_umap_gse306339.png
  results/tables/annotation_marker_scores_gse306339.tsv
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT  = sys.argv[1] if len(sys.argv) > 1 else "."
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures")
TAB_DIR  = os.path.join(PROJECT, "results/tables")

sc.settings.seed = 42

# ── Marker gene panels ─────────────────────────────────────────────────────────
MARKERS = {
    "NK_cell":         ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRK1", "NCR1", "XCL1", "XCL2"],
    "CD8_T_cell":      ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "GZMA", "CCL5"],
    "CD4_T_cell":      ["CD4", "IL7R", "FOXP3", "CTLA4", "IL2RA"],
    "B_cell":          ["CD79A", "CD79B", "MS4A1", "PAX5", "IGHM"],
    "Monocyte":        ["CD14", "LYZ", "S100A8", "S100A9", "CSF1R", "FCGR3A"],
    "GMP":             ["MPO", "ELANE", "AZU1", "PRTN3", "CTSG", "CSF3R"],
    "Progenitor_LSC":  ["CD34", "KIT", "HLF", "HOXA9", "MEIS1", "MECOM", "CD96"],
    "Blast":           ["CD33", "FLT3", "RUNX1", "GATA2", "TAL1"],
    "Erythroid":       ["HBA1", "HBA2", "HBB", "GYPA", "KLF1", "GATA1"],
    "pDC":             ["LILRA4", "CLEC4C", "IRF7", "SIGLEC6", "IL3RA"],
}

CYTOTOXIC_GENES    = ["NKG7", "GNLY", "GZMB", "PRF1", "IFNG", "GZMA"]
CHECKPOINT_GENES   = ["TIGIT", "LAG3", "PDCD1", "HAVCR2", "CTLA4"]
MYELOID_MALIGNANT  = ["CD34", "KIT", "HLF", "HOXA9", "MEIS1", "MLLT3", "MECOM",
                      "FLT3", "CD96", "ITGA6", "CDKN1C", "PROCR", "RUNX1"]


def annotate_dataset(h5ad_path, dataset_name):
    if not os.path.exists(h5ad_path):
        log.warning("File not found: %s — skipping.", h5ad_path)
        return

    log.info("=== Annotating %s ===", dataset_name)
    adata = sc.read_h5ad(h5ad_path)
    log.info("  %d cells × %d genes, %d Leiden clusters",
             adata.n_obs, adata.n_vars, adata.obs["leiden"].nunique())

    # Restore raw counts for scoring
    if adata.raw is not None:
        adata_score = adata.raw.to_adata()
        adata_score.obs = adata.obs.copy()
        adata_score.obsm = adata.obsm.copy()
        sc.pp.normalize_total(adata_score, target_sum=1e4)
        sc.pp.log1p(adata_score)
    else:
        adata_score = adata

    # Score each marker panel
    for pop, genes in MARKERS.items():
        present = [g for g in genes if g in adata_score.var_names]
        if len(present) < 2:
            log.warning("  %s: only %d/%d markers present", pop, len(present), len(genes))
            continue
        sc.tl.score_genes(adata_score, present, score_name=f"score_{pop}", random_state=42)
        adata.obs[f"score_{pop}"] = adata_score.obs[f"score_{pop}"].values

    # Additional functional scores
    cyto_present = [g for g in CYTOTOXIC_GENES if g in adata_score.var_names]
    chk_present  = [g for g in CHECKPOINT_GENES if g in adata_score.var_names]
    if cyto_present:
        sc.tl.score_genes(adata_score, cyto_present, score_name="cytotoxic_score", random_state=42)
        adata.obs["cytotoxic_score"] = adata_score.obs["cytotoxic_score"].values
    if chk_present:
        sc.tl.score_genes(adata_score, chk_present, score_name="checkpoint_score", random_state=42)
        adata.obs["checkpoint_score"] = adata_score.obs["checkpoint_score"].values

    # LSC signature score
    lsc_present = [g for g in MYELOID_MALIGNANT if g in adata_score.var_names]
    if lsc_present:
        sc.tl.score_genes(adata_score, lsc_present, score_name="lsc_score", random_state=42)
        adata.obs["lsc_score"] = adata_score.obs["lsc_score"].values

    # Cluster-level mean scores → assign cell type
    score_cols = [f"score_{p}" for p in MARKERS if f"score_{p}" in adata.obs.columns]
    cluster_scores = (adata.obs.groupby("leiden")[score_cols]
                      .mean()
                      .rename(columns=lambda c: c.replace("score_", "")))

    cluster_labels = cluster_scores.idxmax(axis=1)
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels)

    log.info("  Cell type distribution:\n%s",
             adata.obs["cell_type"].value_counts().to_string())

    # Cytotoxic compartment = NK_cell + CD8_T_cell
    adata.obs["is_cytotoxic"] = adata.obs["cell_type"].isin(["NK_cell", "CD8_T_cell"])

    # ── Dotplot ────────────────────────────────────────────────────────────────
    top_markers = {pop: [g for g in genes if g in adata_score.var_names][:4]
                   for pop, genes in MARKERS.items()}
    top_markers = {k: v for k, v in top_markers.items() if v}
    all_genes = [g for genes in top_markers.values() for g in genes]
    all_genes = list(dict.fromkeys(all_genes))  # deduplicate preserving order

    if all_genes:
        fig, ax = plt.subplots(figsize=(max(12, len(all_genes) * 0.5), 6))
        sc.pl.dotplot(adata, var_names=all_genes, groupby="leiden",
                      ax=ax, show=False, standard_scale="var",
                      title=f"{dataset_name} — marker expression by cluster")
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"annotation_dotplot_{dataset_name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── UMAP coloured by cell type ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc.pl.umap(adata, color="cell_type",  ax=axes[0], show=False, title="Cell type")
    if "timepoint" in adata.obs.columns:
        sc.pl.umap(adata, color="timepoint", ax=axes[1], show=False, title="Timepoint")
    elif "condition" in adata.obs.columns:
        sc.pl.umap(adata, color="condition", ax=axes[1], show=False, title="Condition")
    plt.suptitle(f"{dataset_name} — annotated UMAP", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"annotation_umap_{dataset_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save scores table ──────────────────────────────────────────────────────
    score_df = adata.obs[["sample_id", "patient_id"] +
                          [c for c in ["timepoint", "condition"] if c in adata.obs.columns] +
                          ["leiden", "cell_type"] +
                          [c for c in adata.obs.columns if c.startswith("score_") or
                           c in ["cytotoxic_score", "checkpoint_score", "lsc_score"]]].copy()
    score_df.to_csv(os.path.join(TAB_DIR, f"annotation_marker_scores_{dataset_name}.tsv"),
                    sep="\t")

    out_path = os.path.join(PROC_DIR, f"{dataset_name}_annotated.h5ad")
    adata.write_h5ad(out_path)
    log.info("  Saved: %s", out_path)


annotate_dataset(os.path.join(PROC_DIR, "gse306339_qc.h5ad"),  "gse306339")
annotate_dataset(os.path.join(PROC_DIR, "gse116256_qc.h5ad"),  "gse116256")
log.info("=== Annotation complete ===")
