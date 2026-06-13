"""
QC and preprocessing — GSE311458 (expansion cohort)

4 paired AML patients, pre/post venetoclax+AZA+chidamide.
Clinical outcomes: NR, PR, CR#1, CR#2 (one patient per group).
Files: MTX/TSV per sample.

Outputs:
  data/processed/gse311458_qc.h5ad
  data/processed/gse311458_annotated.h5ad
  results/tables/gse311458_qc_stats.tsv
"""

import sys, os, logging, warnings, glob, re
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT  = sys.argv[1] if len(sys.argv) > 1 else "."
RAW_DIR  = os.path.join(PROJECT, "data/raw/GSE311458")
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures")
TAB_DIR  = os.path.join(PROJECT, "results/tables")
for d in [PROC_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

sc.settings.seed = 42
MAD_THRESH = 3
MT_CAP = 0.20

# Outcome labels from the GEO submission
OUTCOME_MAP = {"NR": "Non-responder", "PR": "Partial_response",
               "CR1": "Complete_response_1", "CR2": "Complete_response_2"}

def detect_samples(raw_dir):
    mtx_files = sorted(glob.glob(os.path.join(raw_dir, "*matrix.mtx.gz")))
    if not mtx_files:
        mtx_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*matrix.mtx.gz"),
                                      recursive=True))
    samples = {}
    for mf in mtx_files:
        fname  = os.path.basename(mf)
        prefix = fname.replace("_matrix.mtx.gz", "")
        parts  = prefix.split("_")
        # Expected: GSMxxxxxx_PatientX_{pre|post}_...
        patient   = next((p for p in parts if re.match(r"[Pp]atient\d+|AML|P\d+", p)),
                         parts[1] if len(parts) > 1 else "unknown")
        timepoint = "post" if any("post" in p.lower() for p in parts) else "pre"
        # Outcome from GSM metadata — map NR/PR/CR
        outcome = next((OUTCOME_MAP[p] for p in parts if p in OUTCOME_MAP), "unknown")
        patient = re.sub(r"-(pre|post)$", "", patient, flags=re.IGNORECASE)
        sample_id = f"{patient}_{timepoint}"
        samples[sample_id] = {
            "prefix": prefix, "dir": raw_dir, "matrix": mf,
            "patient": patient, "timepoint": timepoint, "outcome": outcome,
        }
    return samples

samples = detect_samples(RAW_DIR)
if not samples:
    log.error("No MTX files found in %s", RAW_DIR)
    sys.exit(1)
log.info("Detected %d samples: %s", len(samples), list(samples.keys()))

adatas, qc_records = [], []

for sid, info in samples.items():
    log.info("--- Loading %s ---", sid)
    try:
        adata = sc.read_10x_mtx(info["dir"], var_names="gene_symbols",
                                  make_unique=True, prefix=info["prefix"] + "_")
    except Exception as e:
        log.warning("  Failed: %s", e)
        continue

    adata.obs["sample_id"]  = sid
    adata.obs["patient_id"] = info["patient"]
    adata.obs["timepoint"]  = info["timepoint"]
    adata.obs["outcome"]    = info["outcome"]
    adata.obs["dataset"]    = "GSE311458"
    adata.obs_names = [f"{sid}_{bc}" for bc in adata.obs_names]

    n_raw = adata.n_obs
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)

    def mad_filter(s):
        med = s.median(); mad = (s - med).abs().median()
        return (s >= med - MAD_THRESH * mad) & (s <= med + MAD_THRESH * mad)

    keep = (mad_filter(adata.obs["total_counts"]) &
            mad_filter(adata.obs["n_genes_by_counts"]) &
            (adata.obs["pct_counts_mt"] < MT_CAP * 100))
    adata = adata[keep].copy()

    if adata.n_obs >= 50:
        scrub = scr.Scrublet(adata.X, random_state=42)
        try:
            scores, doublets = scrub.scrub_doublets(
                min_counts=2, min_cells=3,
                n_prin_comps=min(20, adata.n_obs // 3), verbose=False)
            adata.obs["doublet_score"] = scores
            adata.obs["is_doublet"]    = doublets
            adata = adata[~adata.obs["is_doublet"]].copy()
        except Exception as e:
            log.warning("  Scrublet failed: %s", e)

    log.info("  %s: %d → %d cells (%.1f%%)",
             sid, n_raw, adata.n_obs, 100 * adata.n_obs / max(n_raw, 1))
    qc_records.append({
        "sample_id": sid, "patient": info["patient"],
        "timepoint": info["timepoint"], "outcome": info["outcome"],
        "n_raw": n_raw, "n_final": adata.n_obs,
        "median_genes": int(adata.obs["n_genes_by_counts"].median()),
        "median_umi":   int(adata.obs["total_counts"].median()),
    })
    adatas.append(adata)

if not adatas:
    log.error("No samples loaded.")
    sys.exit(1)

adata_all = adatas[0].concatenate(adatas[1:], join="outer", fill_value=0) \
    if len(adatas) > 1 else adatas[0]
log.info("Concatenated: %d cells × %d genes", adata_all.n_obs, adata_all.n_vars)

pd.DataFrame(qc_records).to_csv(
    os.path.join(TAB_DIR, "gse311458_qc_stats.tsv"), sep="\t", index=False)

# Normalise → HVG → PCA → Harmony → UMAP
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)
adata_all.raw = adata_all.copy()

sc.pp.highly_variable_genes(adata_all, n_top_genes=3000, flavor="seurat_v3",
                              batch_key="sample_id")
adata_all = adata_all[:, adata_all.var["highly_variable"]].copy()
sc.pp.scale(adata_all, max_value=10)
sc.tl.pca(adata_all, n_comps=50, svd_solver="arpack", random_state=42)

import harmonypy as hm
ho = hm.run_harmony(adata_all.obsm["X_pca"], adata_all.obs, ["patient_id"],
                     random_state=42)
z = ho.Z_corr
if z.shape[0] != adata_all.n_obs:
    z = z.T
adata_all.obsm["X_pca_harmony"] = z

sc.pp.neighbors(adata_all, use_rep="X_pca_harmony", n_neighbors=15,
                n_pcs=30, random_state=42)
sc.tl.umap(adata_all, random_state=42)
sc.tl.leiden(adata_all, resolution=0.5, random_state=42)

# Annotate with same marker panels as GSE306339
MARKERS = {
    "NK_cell":        ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRK1", "NCR1"],
    "CD8_T_cell":     ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "GZMA"],
    "CD4_T_cell":     ["CD4", "IL7R", "FOXP3", "CTLA4"],
    "B_cell":         ["CD79A", "CD79B", "MS4A1", "PAX5"],
    "Monocyte":       ["CD14", "LYZ", "S100A8", "S100A9", "CSF1R"],
    "GMP":            ["MPO", "ELANE", "AZU1", "PRTN3", "CSF3R"],
    "Progenitor_LSC": ["CD34", "KIT", "HLF", "HOXA9", "MEIS1", "MECOM"],
    "Blast":          ["CD33", "FLT3", "RUNX1", "GATA2"],
    "Erythroid":      ["HBA1", "HBA2", "HBB", "GYPA", "KLF1"],
    "pDC":            ["LILRA4", "CLEC4C", "IRF7"],
}

for pop, genes in MARKERS.items():
    present = [g for g in genes if g in adata_all.var_names]
    if len(present) >= 2:
        sc.tl.score_genes(adata_all, present, score_name=f"score_{pop}",
                          random_state=42)

score_cols = [f"score_{p}" for p in MARKERS if f"score_{p}" in adata_all.obs.columns]
cluster_scores = adata_all.obs.groupby("leiden")[score_cols].mean() \
    .rename(columns=lambda c: c.replace("score_", ""))
adata_all.obs["cell_type"] = adata_all.obs["leiden"].map(
    cluster_scores.idxmax(axis=1))

log.info("Cell types:\n%s", adata_all.obs["cell_type"].value_counts().to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sc.pl.umap(adata_all, color="cell_type",  ax=axes[0], show=False, title="Cell type")
sc.pl.umap(adata_all, color="timepoint",  ax=axes[1], show=False, title="Timepoint")
sc.pl.umap(adata_all, color="outcome",    ax=axes[2], show=False, title="Outcome")
plt.suptitle("GSE311458 — ven+AZA+chidamide (4 patients)", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gse311458_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

adata_all.write_h5ad(os.path.join(PROC_DIR, "gse311458_annotated.h5ad"))
log.info("=== GSE311458 QC + annotation complete ===")
