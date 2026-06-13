"""
QC and preprocessing — GSE306339 (TP53-mutated AML, venetoclax + azacitidine)

3 patients (AML-A15, AML-A16, AML-A27), pre/post treatment pairs.
Files: MTX + barcodes + features per sample.

Outputs:
  data/processed/gse306339_raw.h5ad   — post-QC, pre-normalisation
  data/processed/gse306339_qc.h5ad    — normalised, HVG, PCA, Harmony
  results/tables/gse306339_qc_stats.tsv
  results/figures/gse306339_qc_metrics.png
"""

import sys, os, logging, warnings, glob
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
RAW_DIR  = os.path.join(PROJECT, "data/raw/GSE306339")
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures")
TAB_DIR  = os.path.join(PROJECT, "results/tables")
for d in [PROC_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

sc.settings.seed = 42
np.random.seed(42)

# ── Sample manifest ────────────────────────────────────────────────────────────
# GSE306339: 6 GSM entries, naming pattern GSM_*_AML-A15_pre, etc.
# Files unpacked from RAW.tar: {GSM}_{patient}_{timepoint}_barcodes.tsv.gz etc.
# We auto-detect sample folders from extracted files.

def detect_samples(raw_dir):
    """Find all MTX files in raw_dir (flat layout: GSMxxx_AML-Axx-pre_matrix.mtx.gz)."""
    mtx_files = glob.glob(os.path.join(raw_dir, "*_matrix.mtx.gz"))
    samples = {}
    for mf in sorted(mtx_files):
        fname = os.path.basename(mf)
        # Pattern: GSMxxxxxx_AML-Axx-{pre|post}_matrix.mtx.gz
        prefix = fname.replace("_matrix.mtx.gz", "")  # e.g. GSM9197630_AML-A15-pre
        parts  = prefix.split("_")
        # patient = part containing AML-Axx, timepoint = pre/post
        patient   = next((p for p in parts if p.startswith("AML-A")), "unknown")
        timepoint = "post" if any("post" in p.lower() for p in parts) else "pre"
        sample_id = f"{patient}_{timepoint}"
        barcodes = os.path.join(raw_dir, f"{prefix}_barcodes.tsv.gz")
        features = os.path.join(raw_dir, f"{prefix}_features.tsv.gz")
        samples[sample_id] = {
            "prefix": prefix, "dir": raw_dir,
            "matrix": mf, "barcodes": barcodes, "features": features,
            "patient": patient, "timepoint": timepoint,
        }
    return samples

samples = detect_samples(RAW_DIR)
if not samples:
    log.error("No MTX files found in %s — check extraction.", RAW_DIR)
    sys.exit(1)
log.info("Detected %d samples: %s", len(samples), list(samples.keys()))

# ── Per-sample QC ─────────────────────────────────────────────────────────────
MAD_THRESH = 3
MT_CAP     = 0.20
DOUBLET_THRESHOLD = 0.25

adatas, qc_records = [], []

for sid, info in samples.items():
    log.info("--- Loading %s ---", sid)
    adata = sc.read_10x_mtx(
        info["dir"],
        var_names="gene_symbols",
        make_unique=True,
        prefix=info["prefix"] + "_",
    )
    adata.obs["sample_id"]  = sid
    adata.obs["patient_id"] = info["patient"]
    adata.obs["timepoint"]  = info["timepoint"]
    adata.obs_names = [f"{sid}_{bc}" for bc in adata.obs_names]

    n_raw = adata.n_obs
    log.info("  %s: %d cells, %d genes (raw)", sid, adata.n_obs, adata.n_vars)

    # Mito fraction
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)

    # MAD-based filtering
    def mad_filter(series, n_mads=MAD_THRESH):
        med = series.median()
        mad = (series - med).abs().median()
        return (series >= med - n_mads * mad) & (series <= med + n_mads * mad)

    keep = (
        mad_filter(adata.obs["total_counts"]) &
        mad_filter(adata.obs["n_genes_by_counts"]) &
        (adata.obs["pct_counts_mt"] < MT_CAP * 100)
    )
    adata = adata[keep].copy()
    log.info("  %s: %d cells retained after QC (%.1f%%)",
             sid, adata.n_obs, 100 * adata.n_obs / n_raw)

    # Doublet removal
    scrub = scr.Scrublet(adata.X, random_state=42)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2, min_cells=3, n_prin_comps=20, verbose=False
    )
    adata.obs["doublet_score"]  = doublet_scores
    adata.obs["is_doublet"]     = predicted_doublets
    n_before = adata.n_obs
    adata = adata[~adata.obs["is_doublet"]].copy()
    log.info("  %s: %d → %d after doublet removal", sid, n_before, adata.n_obs)

    qc_records.append({
        "sample_id": sid, "patient": info["patient"], "timepoint": info["timepoint"],
        "n_raw": n_raw, "n_after_qc": n_before, "n_final": adata.n_obs,
        "pct_retained": round(100 * adata.n_obs / n_raw, 1),
        "median_genes": int(adata.obs["n_genes_by_counts"].median()),
        "median_umi":   int(adata.obs["total_counts"].median()),
        "median_mt_pct": round(adata.obs["pct_counts_mt"].median(), 2),
    })
    adatas.append(adata)

# ── Concatenate ───────────────────────────────────────────────────────────────
adata_all = adatas[0].concatenate(adatas[1:], join="outer", fill_value=0)
log.info("Concatenated: %d cells × %d genes", adata_all.n_obs, adata_all.n_vars)

# Save raw h5ad
raw_path = os.path.join(PROC_DIR, "gse306339_raw.h5ad")
adata_all.write_h5ad(raw_path)
log.info("Saved raw h5ad: %s", raw_path)

# ── QC stats table ────────────────────────────────────────────────────────────
qc_df = pd.DataFrame(qc_records)
qc_df.to_csv(os.path.join(TAB_DIR, "gse306339_qc_stats.tsv"), sep="\t", index=False)
log.info("\n%s", qc_df.to_string(index=False))

# ── QC violin plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label in zip(axes,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    ["Total UMIs", "Genes detected", "% mitochondrial"]):
    for sid in adata_all.obs["sample_id"].unique():
        vals = adata_all.obs.loc[adata_all.obs["sample_id"] == sid, col]
        ax.violinplot(vals, positions=[list(adata_all.obs["sample_id"].unique()).index(sid)],
                      showmedians=True, widths=0.7)
    ax.set_xticks(range(len(adata_all.obs["sample_id"].unique())))
    ax.set_xticklabels(adata_all.obs["sample_id"].unique(), rotation=45, ha="right", fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel(label, fontsize=9)
plt.suptitle("GSE306339 — per-sample QC metrics (post-filter)", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gse306339_qc_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Normalise → HVG → PCA → Harmony ──────────────────────────────────────────
log.info("Normalising...")
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)
adata_all.raw = adata_all.copy()

sc.pp.highly_variable_genes(adata_all, n_top_genes=3000, flavor="seurat_v3",
                             batch_key="sample_id")
adata_all = adata_all[:, adata_all.var["highly_variable"]].copy()
log.info("HVGs: %d", adata_all.n_vars)

sc.pp.scale(adata_all, max_value=10)
sc.tl.pca(adata_all, n_comps=50, svd_solver="arpack", random_state=42)

log.info("Running Harmony...")
import harmonypy as hm
ho = hm.run_harmony(adata_all.obsm["X_pca"], adata_all.obs, ["patient_id"], random_state=42)
z = ho.Z_corr
if z.shape[0] != adata_all.n_obs:
    z = z.T
adata_all.obsm["X_pca_harmony"] = z

sc.pp.neighbors(adata_all, use_rep="X_pca_harmony", n_neighbors=15, n_pcs=30, random_state=42)
sc.tl.umap(adata_all, random_state=42)
sc.tl.leiden(adata_all, resolution=0.5, random_state=42)

# ── UMAP coloured by timepoint + patient ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_all, color="timepoint",  ax=axes[0], show=False, title="Timepoint")
sc.pl.umap(adata_all, color="patient_id", ax=axes[1], show=False, title="Patient")
sc.pl.umap(adata_all, color="leiden",     ax=axes[2], show=False, title="Leiden clusters")
plt.suptitle("GSE306339 — UMAP (Harmony-corrected)", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gse306339_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

proc_path = os.path.join(PROC_DIR, "gse306339_qc.h5ad")
adata_all.write_h5ad(proc_path)
log.info("Saved processed h5ad: %s", proc_path)
log.info("=== GSE306339 QC complete ===")
