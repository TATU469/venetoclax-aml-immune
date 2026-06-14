"""
Pseudobulk DGE — NK and CD8 T cells, Responders vs Non-responders, pre-treatment.

Two complementary approaches to avoid inflated cell-level p-values:
  1. pydeseq2 on aggregated pseudobulk counts (sum per patient)
  2. Patient-level Wilcoxon on mean expression per patient (n=8R vs n=3NR)

Input: gse269669_annotated.h5ad (.raw = log-normalised all genes)

Outputs:
  results/tables/validation/pseudobulk_NK_pydeseq2.tsv
  results/tables/validation/pseudobulk_CD8_pydeseq2.tsv
  results/tables/validation/pseudobulk_NK_wilcoxon_patient.tsv
  results/tables/validation/pseudobulk_CD8_wilcoxon_patient.tsv
  results/figures/validation/pseudobulk_NK_volcano.png
  results/figures/validation/pseudobulk_CD8_volcano.png
  results/figures/validation/pseudobulk_exhaustion_dotplot.png
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats, sparse
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "."
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures/validation")
TAB_DIR  = os.path.join(PROJECT, "results/tables/validation")
for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

RESPONSE_MAP = {
    "PT33": "Responder", "PT34": "Non-responder", "PT35": "Responder",
    "PT36": "Responder", "PT37": "Responder",     "PT38": "Responder",
    "PT39": "Responder", "PT40": "Non-responder", "PT41": "Responder",
    "PT42": "Responder", "PT43": "Non-responder",
}

AZIMUTH_MAP = {
    "NK": "NK_cell", "NK CD56+": "NK_cell", "ILC": "NK_cell",
    "CD4 Memory": "CD4_T_cell", "CD4 Naive": "CD4_T_cell",
    "CD4 Effector": "CD4_T_cell", "Treg": "CD4_T_cell",
    "CD8 Effector_1": "CD8_T_cell", "CD8 Effector_2": "CD8_T_cell",
    "CD8 Memory": "CD8_T_cell", "CD8 Naive": "CD8_T_cell", "MAIT": "CD8_T_cell",
    "Memory B": "B_cell", "Naive B": "B_cell", "Plasma": "B_cell",
    "CD14 Mono": "Monocyte", "CD16 Mono": "Monocyte", "Macrophage": "Monocyte",
    "Late Eryth": "Erythroid", "Early Eryth": "Erythroid",
    "GMP": "Progenitor", "HSC": "Progenitor",
}

EXHAUSTION_GENES   = ["TIGIT", "LAG3", "PDCD1", "HAVCR2", "CTLA4",
                       "TOX", "TOX2", "ENTPD1", "BATF", "IKZF2"]
CYTOTOXICITY_GENES = ["GZMB", "GZMA", "GZMH", "GZMK", "PRF1",
                       "NKG7", "GNLY", "IFNG", "XCL1", "XCL2"]
HIGHLIGHT_GENES    = set(EXHAUSTION_GENES + CYTOTOXICITY_GENES +
                         ["KLRK1", "KLRD1", "NCR1", "FCGR3A", "CD8A", "CD8B",
                          "CD3D", "CD3G", "NCAM1", "KIR2DL1", "KIR3DL1"])

# ── Load ──────────────────────────────────────────────────────────────────────
log.info("Loading GSE269669...")
adata = sc.read_h5ad(os.path.join(PROC_DIR, "gse269669_annotated.h5ad"))

if "seurat_celltype" in adata.obs.columns:
    adata.obs["cell_type_broad"] = (
        adata.obs["seurat_celltype"].astype(str).map(AZIMUTH_MAP).fillna("Other")
    )
else:
    adata.obs["cell_type_broad"] = adata.obs["cell_type"]

adata.obs["response"] = adata.obs["patient_id"].astype(str).map(RESPONSE_MAP)

# Restore log-normalised full matrix from .raw
log.info("Restoring log-normalised matrix from .raw...")
adata_lognorm = adata.raw.to_adata()
adata_lognorm.obs = adata.obs.copy()
log.info("Raw matrix: %d cells × %d genes", adata_lognorm.n_obs, adata_lognorm.n_vars)

# ── Pseudobulk helpers ────────────────────────────────────────────────────────
def build_pseudobulk(adata_ln, cell_type, timepoint="pre"):
    """Aggregate log-normalised counts per patient → pseudobulk count matrix."""
    sub = adata_ln[
        (adata_ln.obs["cell_type_broad"] == cell_type) &
        (adata_ln.obs["timepoint"] == timepoint) &
        (adata_ln.obs["response"].isin(["Responder", "Non-responder"]))
    ].copy()

    patients  = sorted(sub.obs["patient_id"].unique())
    responses = [RESPONSE_MAP[p] for p in patients]
    n_cells   = {p: (sub.obs["patient_id"] == p).sum() for p in patients}
    log.info("  %s: %d patients, cell counts: %s", cell_type, len(patients),
             {p: n_cells[p] for p in patients})

    # Sum expm1(log-norm) per patient → approximate pseudobulk counts, round to int
    X = sub.X if not sparse.issparse(sub.X) else sub.X.toarray()
    pb_rows = []
    for p in patients:
        mask = sub.obs["patient_id"].values == p
        # expm1 recovers normalized counts; sum across cells for pseudobulk
        pb_rows.append(np.expm1(X[mask]).sum(axis=0))

    pb = pd.DataFrame(
        np.array(pb_rows),
        index=patients,
        columns=sub.var_names,
    )
    # Remove genes with zero counts across all patients
    pb = pb.loc[:, pb.sum(axis=0) > 0]
    # Round to integer for pydeseq2
    pb_int = pb.round().astype(int)

    meta = pd.DataFrame({"response": responses}, index=patients)
    return pb_int, meta, n_cells

def run_pydeseq2(pb_counts, meta, label, min_cells=50):
    """Run pydeseq2 on pseudobulk count matrix."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        log.error("pydeseq2 not installed")
        return None

    log.info("Running pydeseq2 for %s (n=%d samples)...", label, len(pb_counts))
    dds = DeseqDataSet(
        counts=pb_counts,
        metadata=meta,
        design_factors="response",
        ref_level=["response", "Non-responder"],
        refit_cooks=True,
        quiet=True,
    )
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["response", "Responder", "Non-responder"], quiet=True)
    ds.summary()
    res = ds.results_df.copy()
    res = res.reset_index().rename(columns={"index": "gene"})
    res = res.sort_values("padj")
    log.info("pydeseq2 %s: %d significant (FDR<0.05)", label,
             (res["padj"] < 0.05).sum())
    return res

def run_patient_wilcoxon(pb_mean, meta, label):
    """Patient-level Wilcoxon: mean expression per patient, 8R vs 3NR."""
    R_idx  = meta[meta["response"] == "Responder"].index
    NR_idx = meta[meta["response"] == "Non-responder"].index
    genes = pb_mean.columns
    rows = []
    for gene in genes:
        r_vals  = pb_mean.loc[R_idx,  gene].values
        nr_vals = pb_mean.loc[NR_idx, gene].values
        if r_vals.std() == 0 and nr_vals.std() == 0:
            continue
        _, p = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        lfc = np.log2((r_vals.mean() + 1e-6) / (nr_vals.mean() + 1e-6))
        rows.append({"gene": gene, "log2FC_R_vs_NR": lfc,
                     "mean_R": r_vals.mean(), "mean_NR": nr_vals.mean(), "pval": p})
    res = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(res["pval"], method="fdr_bh")
    res["padj"] = padj
    res = res.sort_values("padj")
    log.info("Patient Wilcoxon %s: %d FDR<0.05, %d p<0.05",
             label, (res["padj"] < 0.05).sum(), (res["pval"] < 0.05).sum())

    # Key genes
    key = res[res["gene"].isin(HIGHLIGHT_GENES)].copy()
    log.info("Key genes in %s:\n%s",
             label, key[["gene", "log2FC_R_vs_NR", "pval", "padj"]].to_string(index=False))
    return res

# ── Run for NK and CD8 ────────────────────────────────────────────────────────
results_pydeseq = {}
results_wilcox  = {}

for ct in ["NK_cell", "CD8_T_cell"]:
    log.info("\n=== %s pseudobulk DGE ===", ct)
    pb_int, meta, _ = build_pseudobulk(adata_lognorm, ct)

    # Mean expression per patient (for Wilcoxon)
    pb_mean = pb_int.copy().astype(float)

    # pydeseq2
    label = ct.replace("_cell", "").replace("_T", "T")
    res_dds = run_pydeseq2(pb_int, meta, label)
    if res_dds is not None:
        results_pydeseq[ct] = res_dds
        fname = f"pseudobulk_{ct.replace('_cell','').replace('_T','T')}_pydeseq2.tsv"
        res_dds.to_csv(os.path.join(TAB_DIR, fname), sep="\t", index=False)

    # Patient Wilcoxon
    res_wlx = run_patient_wilcoxon(pb_mean, meta, label)
    results_wilcox[ct] = res_wlx
    fname = f"pseudobulk_{ct.replace('_cell','').replace('_T','T')}_wilcoxon_patient.tsv"
    res_wlx.to_csv(os.path.join(TAB_DIR, fname), sep="\t", index=False)

# ── Volcano plots (pydeseq2) ──────────────────────────────────────────────────
for ct, res in results_pydeseq.items():
    if res is None or res.empty:
        continue
    fig, ax = plt.subplots(figsize=(9, 7))
    lfc = res["log2FoldChange"].fillna(0).values
    pv  = np.clip(-np.log10(res["padj"].fillna(1).replace(0, 1e-10)), 0, 15)
    sig = (res["padj"].fillna(1) < 0.05) & (np.abs(lfc) > 0.5)

    ax.scatter(lfc[~sig], pv[~sig], c="#bdc3c7", s=8, alpha=0.4, linewidths=0)
    ax.scatter(lfc[sig & (lfc > 0)], pv[sig & (lfc > 0)],
               c="#2980b9", s=18, alpha=0.8, linewidths=0, label="Up in Responder")
    ax.scatter(lfc[sig & (lfc < 0)], pv[sig & (lfc < 0)],
               c="#e74c3c", s=18, alpha=0.8, linewidths=0, label="Up in Non-responder")

    for _, row in res[res["gene"].isin(HIGHLIGHT_GENES)].iterrows():
        y = -np.log10(max(row["padj"], 1e-10)) if not pd.isna(row["padj"]) else 0
        lf = row["log2FoldChange"] if not pd.isna(row["log2FoldChange"]) else 0
        color = ("#2980b9" if lf > 0 else "#e74c3c") \
                if (not pd.isna(row["padj"]) and row["padj"] < 0.05) else "#7f8c8d"
        ax.annotate(row["gene"], (lf, y), fontsize=8, color=color,
                    xytext=(3, 0), textcoords="offset points")
        ax.scatter([lf], [y], c=color, s=35, zorder=5, edgecolors="white", linewidths=0.5)

    ax.axhline(-np.log10(0.05), color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(0.5, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(-0.5, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Log2 fold change (Responder / Non-responder)", fontsize=10)
    ax.set_ylabel("-log10(FDR-adjusted p-value)", fontsize=10)
    ax.set_title(f"Pseudobulk DESeq2 — {ct.replace('_', ' ')}\n"
                 f"Pre-treatment, GSE269669 (n=8 R / n=3 NR)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"pseudobulk_{ct.replace('_cell','').replace('_T','T')}_volcano.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved volcano: %s", fname)

# ── Dot plot: exhaustion + cytotoxicity genes across patients ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
gene_panel = [g for g in EXHAUSTION_GENES + CYTOTOXICITY_GENES
              if g in adata_lognorm.var_names]

for ax, ct in zip(axes, ["NK_cell", "CD8_T_cell"]):
    sub = adata_lognorm[
        (adata_lognorm.obs["cell_type_broad"] == ct) &
        (adata_lognorm.obs["timepoint"] == "pre")
    ]
    # Mean per patient
    rows = []
    for pid in sorted(sub.obs["patient_id"].unique()):
        mask = sub.obs["patient_id"] == pid
        expr = pd.Series(
            np.asarray(sub[mask][:, gene_panel].X.mean(axis=0)).flatten(),
            index=gene_panel
        )
        expr["patient_id"] = pid
        expr["response"] = RESPONSE_MAP[pid]
        rows.append(expr)
    pt_df = pd.DataFrame(rows).set_index("patient_id")
    gene_vals = pt_df[gene_panel].astype(float)

    # Sort patients: Responders first
    order = sorted(pt_df.index, key=lambda p: (pt_df.loc[p, "response"] == "Responder", p),
                   reverse=True)
    gene_vals = gene_vals.loc[order]

    im = ax.imshow(gene_vals.T.values, aspect="auto", cmap="RdBu_r",
                   vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{p}\n({'R' if RESPONSE_MAP[p]=='Responder' else 'NR'})" for p in order],
        fontsize=7, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(gene_panel)))
    ax.set_yticklabels(gene_panel, fontsize=8)
    # Divider between exhaustion and cytotoxicity genes
    n_exh = sum(1 for g in gene_panel if g in EXHAUSTION_GENES)
    ax.axhline(n_exh - 0.5, color="white", lw=2)
    ax.set_title(f"{ct.replace('_', ' ')} pre-treatment\n(R=Responder, NR=Non-responder)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean log-norm expression", shrink=0.7)

plt.suptitle("Pre-treatment exhaustion & cytotoxicity gene expression per patient\n"
             "GSE269669 — ven+AZA+magrolimab", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG_DIR, "pseudobulk_exhaustion_dotplot.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("Saved exhaustion dotplot")

log.info("=== Pseudobulk DGE complete ===")
