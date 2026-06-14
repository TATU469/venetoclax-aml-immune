"""
DGE within NK and CD8 T cells — Responders vs Non-responders, pre-treatment.

GSE269669: 11 TP53-mut AML, ven+AZA+magrolimab.
Hypothesis: NK/CD8 cells in non-responders are functionally exhausted despite
high numbers; responders have more cytotoxically active NK/CD8 cells.

Uses log-normalised counts from adata.raw (all genes, pre-scaling).
Cell-level Wilcoxon rank-sum test (scanpy rank_genes_groups).

Outputs:
  results/tables/validation/dge_NK_R_vs_NR.tsv
  results/tables/validation/dge_CD8_R_vs_NR.tsv
  results/figures/validation/dge_NK_volcano.png
  results/figures/validation/dge_CD8_volcano.png
  results/figures/validation/dge_exhaustion_scores.png
  results/figures/validation/dge_heatmap_NK.png
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats
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
NK_RECEPTOR_GENES  = ["KLRK1", "KLRD1", "NCR1", "FCGR3A", "NCAM1",
                       "KIR2DL1", "KIR2DL2", "KIR3DL1", "LILRB1"]

# ── Load ──────────────────────────────────────────────────────────────────────
log.info("Loading GSE269669...")
adata = sc.read_h5ad(os.path.join(PROC_DIR, "gse269669_annotated.h5ad"))
log.info("Loaded: %d cells, raw available: %s", adata.n_obs, adata.raw is not None)

if "seurat_celltype" in adata.obs.columns:
    adata.obs["cell_type_broad"] = (
        adata.obs["seurat_celltype"].astype(str).map(AZIMUTH_MAP).fillna("Other")
    )
else:
    adata.obs["cell_type_broad"] = adata.obs["cell_type"]

adata.obs["response"] = adata.obs["patient_id"].astype(str).map(RESPONSE_MAP)

# ── Restore log-normalised matrix from .raw for DGE ──────────────────────────
if adata.raw is not None:
    adata_raw = adata.raw.to_adata()
    adata_raw.obs = adata.obs.copy()
    log.info("Using .raw: %d cells × %d genes", adata_raw.n_obs, adata_raw.n_vars)
else:
    adata_raw = adata
    log.warning("No .raw found — using scaled HVG matrix (less ideal for DGE)")

def run_dge(adata_full, cell_type, label):
    """DGE: Responder vs Non-responder within cell_type, pre-treatment only."""
    sub = adata_full[
        (adata_full.obs["cell_type_broad"] == cell_type) &
        (adata_full.obs["timepoint"] == "pre") &
        (adata_full.obs["response"].isin(["Responder", "Non-responder"]))
    ].copy()

    n_R  = (sub.obs["response"] == "Responder").sum()
    n_NR = (sub.obs["response"] == "Non-responder").sum()
    log.info("%s pre-treatment: %d Responder cells, %d Non-responder cells", cell_type, n_R, n_NR)

    if n_R < 10 or n_NR < 10:
        log.warning("Too few cells for DGE in %s", cell_type)
        return None

    sc.tl.rank_genes_groups(sub, groupby="response", groups=["Responder"],
                             reference="Non-responder", method="wilcoxon",
                             n_genes=sub.n_vars, key_added="dge")

    res = sc.get.rank_genes_groups_df(sub, group="Responder", key="dge")
    res.columns = ["gene", "scores", "logfoldchanges", "pvals", "pvals_adj"]
    res = res.sort_values("pvals_adj")
    res["cell_type"] = cell_type
    log.info("Top upregulated in Responders:\n%s",
             res[res["logfoldchanges"] > 0].head(15)[
                 ["gene", "logfoldchanges", "pvals_adj"]].to_string(index=False))
    log.info("Top downregulated in Responders (=upregulated in NR):\n%s",
             res[res["logfoldchanges"] < 0].head(15)[
                 ["gene", "logfoldchanges", "pvals_adj"]].to_string(index=False))

    # Check key gene sets
    for gset_name, gset in [("Exhaustion", EXHAUSTION_GENES),
                              ("Cytotoxicity", CYTOTOXICITY_GENES),
                              ("NK_receptors", NK_RECEPTOR_GENES)]:
        present = res[res["gene"].isin(gset)][["gene", "logfoldchanges", "pvals_adj"]]
        log.info("%s genes in %s:\n%s", gset_name, cell_type, present.to_string(index=False))

    return res, sub

# ── Run DGE for NK and CD8 ────────────────────────────────────────────────────
dge_results = {}
adata_subsets = {}

for ct in ["NK_cell", "CD8_T_cell"]:
    result = run_dge(adata_raw, ct, ct)
    if result is not None:
        dge_results[ct], adata_subsets[ct] = result
        out_name = "dge_NK_R_vs_NR.tsv" if ct == "NK_cell" else "dge_CD8_R_vs_NR.tsv"
        dge_results[ct].to_csv(os.path.join(TAB_DIR, out_name), sep="\t", index=False)

# ── Volcano plots ─────────────────────────────────────────────────────────────
HIGHLIGHT = set(EXHAUSTION_GENES + CYTOTOXICITY_GENES + NK_RECEPTOR_GENES)

for ct, res in dge_results.items():
    fig, ax = plt.subplots(figsize=(9, 7))
    lfc = res["logfoldchanges"].values
    pv  = np.clip(-np.log10(res["pvals_adj"].replace(0, 1e-300)), 0, 50)

    # Background points
    sig = (res["pvals_adj"] < 0.05) & (np.abs(lfc) > 0.25)
    ax.scatter(lfc[~sig], pv[~sig], c="#bdc3c7", s=6, alpha=0.4, linewidths=0)
    ax.scatter(lfc[sig & (lfc > 0)], pv[sig & (lfc > 0)],
               c="#2980b9", s=12, alpha=0.7, linewidths=0, label="Up in Responder")
    ax.scatter(lfc[sig & (lfc < 0)], pv[sig & (lfc < 0)],
               c="#e74c3c", s=12, alpha=0.7, linewidths=0, label="Up in Non-responder")

    # Label highlight genes
    for _, row in res[res["gene"].isin(HIGHLIGHT)].iterrows():
        y = -np.log10(max(row["pvals_adj"], 1e-300))
        color = ("#2980b9" if row["logfoldchanges"] > 0 else "#e74c3c") \
                if row["pvals_adj"] < 0.05 else "#7f8c8d"
        ax.annotate(row["gene"], (row["logfoldchanges"], y),
                    fontsize=7, color=color,
                    xytext=(3, 0), textcoords="offset points")
        ax.scatter([row["logfoldchanges"]], [y], c=color, s=25, zorder=5)

    ax.axhline(-np.log10(0.05), color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(0.25, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(-0.25, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Log2 fold change (Responder / Non-responder)", fontsize=10)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=10)
    ax.set_title(f"{ct.replace('_', ' ')} — Responder vs Non-responder\n"
                 f"Pre-treatment, GSE269669 (n=8R / n=3NR)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = "dge_NK_volcano.png" if ct == "NK_cell" else "dge_CD8_volcano.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved volcano: %s", fname)

# ── Exhaustion vs cytotoxicity score comparison ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ct in zip(axes, ["NK_cell", "CD8_T_cell"]):
    if ct not in adata_subsets:
        continue
    sub = adata_subsets[ct]

    exh_genes = [g for g in EXHAUSTION_GENES if g in sub.var_names]
    cyt_genes  = [g for g in CYTOTOXICITY_GENES if g in sub.var_names]

    if exh_genes:
        sc.tl.score_genes(sub, exh_genes, score_name="exhaustion_score", random_state=42)
    if cyt_genes:
        sc.tl.score_genes(sub, cyt_genes, score_name="cytotoxicity_score", random_state=42)

    for score, color_R, color_NR, label in [
        ("exhaustion_score",  "#e74c3c", "#2980b9", "Exhaustion"),
        ("cytotoxicity_score","#2980b9", "#e74c3c", "Cytotoxicity"),
    ]:
        if score not in sub.obs.columns:
            continue
        R_scores  = sub.obs[sub.obs["response"] == "Responder"][score]
        NR_scores = sub.obs[sub.obs["response"] == "Non-responder"][score]
        _, p = stats.mannwhitneyu(R_scores, NR_scores, alternative="two-sided")
        log.info("%s %s: R median=%.3f NR median=%.3f p=%.4f",
                 ct, score, R_scores.median(), NR_scores.median(), p)

    if "exhaustion_score" in sub.obs.columns and "cytotoxicity_score" in sub.obs.columns:
        for resp, marker, color in [("Responder", "o", "#2980b9"),
                                     ("Non-responder", "s", "#e74c3c")]:
            grp = sub.obs[sub.obs["response"] == resp]
            ax.scatter(grp["cytotoxicity_score"], grp["exhaustion_score"],
                       c=color, marker=marker, s=4, alpha=0.3, linewidths=0)
        # Patient-level medians
        for pid, grp in sub.obs.groupby("patient_id"):
            resp = RESPONSE_MAP.get(pid, "unknown")
            color = "#2980b9" if resp == "Responder" else "#e74c3c"
            ax.scatter(grp["cytotoxicity_score"].median(),
                       grp["exhaustion_score"].median(),
                       c=color, s=80, zorder=10,
                       edgecolors="white", linewidths=1)
            ax.annotate(pid, (grp["cytotoxicity_score"].median(),
                              grp["exhaustion_score"].median()),
                        fontsize=6, xytext=(3, 2), textcoords="offset points")

    ax.set_xlabel("Cytotoxicity score", fontsize=10)
    ax.set_ylabel("Exhaustion score", fontsize=10)
    ax.set_title(f"{ct.replace('_', ' ')} pre-treatment\nCytotoxicity vs Exhaustion",
                 fontsize=10)

legend_els = [mpatches.Patch(color="#2980b9", label="Responder (n=8)"),
              mpatches.Patch(color="#e74c3c", label="Non-responder (n=3)")]
fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=9, frameon=False)
plt.suptitle("Pre-treatment functional state: Responders vs Non-responders\n"
             "GSE269669 — ven+AZA+magrolimab", fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(FIG_DIR, "dge_exhaustion_scores.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Heatmap: top DGE genes + key markers for NK ───────────────────────────────
if "NK_cell" in dge_results and "NK_cell" in adata_subsets:
    sub = adata_subsets["NK_cell"]
    res = dge_results["NK_cell"]

    top_up   = res[res["logfoldchanges"] > 0].head(15)["gene"].tolist()
    top_down = res[res["logfoldchanges"] < 0].head(15)["gene"].tolist()
    key_genes = list(dict.fromkeys(
        [g for g in EXHAUSTION_GENES + CYTOTOXICITY_GENES + NK_RECEPTOR_GENES
         if g in sub.var_names] + top_up + top_down
    ))[:40]

    if key_genes:
        sc.pl.heatmap(sub, key_genes, groupby="response",
                      standard_scale="var", show=False,
                      figsize=(14, max(4, len(key_genes) * 0.28)))
        plt.suptitle("NK cell gene expression — Responder vs Non-responder (pre-treatment)",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "dge_heatmap_NK.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved NK heatmap")

log.info("=== DGE response analysis complete ===")
