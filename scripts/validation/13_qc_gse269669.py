"""
QC and preprocessing — GSE269669 (primary expansion cohort)

11 TP53-mutated AML patients, pre/post venetoclax+AZA+magrolimab.
Paired design — highest-priority dataset for statistical power.
Loads from converted MTX directories produced by 12_convert_gse269669.R.

Outputs:
  data/processed/gse269669_annotated.h5ad
  results/tables/gse269669_qc_stats.tsv
  results/figures/gse269669_umap.png
"""

import sys, os, logging, warnings, glob, re, gzip
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
RAW_DIR  = os.path.join(PROJECT, "data/raw/GSE269669/converted")
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures")
TAB_DIR  = os.path.join(PROJECT, "results/tables")
for d in [PROC_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

sc.settings.seed = 42
MAD_THRESH = 3
MT_CAP     = 0.20

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

TIMEPOINT_SUFFIX = {"a": "pre", "b": "post", "c": "post_c2"}

def parse_patient(sample_dir):
    """Extract patient ID — strips trailing letter suffix: PT33A → PT33."""
    name = os.path.basename(sample_dir.rstrip("/"))
    m = re.match(r"^(PT\d+)[A-Ca-c]$", name)
    return m.group(1) if m else name

def parse_timepoint(sample_dir):
    """Map letter suffix to timepoint: A=pre, B=post, C=post_c2."""
    name = os.path.basename(sample_dir.rstrip("/"))
    m = re.match(r"^PT\d+([A-Ca-c])$", name)
    return TIMEPOINT_SUFFIX.get(m.group(1).lower(), "pre") if m else "pre"

def is_tme_sample(sample_dir):
    """Keep only PTxxA/B/C directories (TME immune cells from avm_tme.rds).

    PTxx_Pre/PTxx_Post directories come from avm_aml.rds (blasts) and are excluded.
    """
    name = os.path.basename(sample_dir.rstrip("/"))
    return bool(re.match(r"^PT\d+[A-Ca-c]$", name))

def load_metadata_from_tsv(sample_dir):
    """Load per-cell metadata saved by R script, if available."""
    meta_path = os.path.join(sample_dir, "metadata.tsv.gz")
    if os.path.exists(meta_path):
        try:
            return pd.read_csv(meta_path, sep="\t", index_col="barcode")
        except Exception:
            pass
    return None

def mad_filter(s):
    med = s.median()
    mad = (s - med).abs().median()
    return (s >= med - MAD_THRESH * mad) & (s <= med + MAD_THRESH * mad)

# ── Detect converted sample directories ──────────────────────────────────────
sample_dirs = sorted([
    d for d in glob.glob(os.path.join(RAW_DIR, "*/"))
    if os.path.isfile(os.path.join(d, "matrix.mtx.gz"))
])

if not sample_dirs:
    log.error("No converted MTX directories found in %s", RAW_DIR)
    log.error("Run 12_convert_gse269669.R first.")
    sys.exit(1)

# Keep only TME-file samples (PTxxA/B/C); drop AML-file duplicates (PTxx_Pre/Post)
sample_dirs = [d for d in sample_dirs if is_tme_sample(d)]
log.info("After filtering to TME samples: %d directories", len(sample_dirs))
for d in sample_dirs:
    log.info("  %s → patient=%s timepoint=%s",
             os.path.basename(d), parse_patient(d), parse_timepoint(d))

adatas, qc_records = [], []

for sample_dir in sample_dirs:
    sid = os.path.basename(sample_dir.rstrip("/"))
    patient_id = parse_patient(sample_dir)
    timepoint  = parse_timepoint(sample_dir)
    log.info("--- Loading %s (patient=%s, timepoint=%s) ---", sid, patient_id, timepoint)

    try:
        adata = sc.read_10x_mtx(sample_dir, var_names="gene_symbols",
                                  make_unique=True)
    except Exception as e:
        log.warning("  Failed to load MTX: %s", e)
        continue

    # Load per-cell metadata from R export
    meta = load_metadata_from_tsv(sample_dir)

    # Prefer Azimuth L2 annotation, fall back to marker scoring
    celltype_col = None
    if meta is not None:
        for col in ["predicted.celltype.l2", "predicted.celltype.l1",
                    "cell_type", "celltype", "annotation"]:
            if col in meta.columns:
                celltype_col = col
                log.info("  Using cell type annotation: %s", col)
                break

    adata.obs["sample_id"]  = sid
    adata.obs["patient_id"] = patient_id
    adata.obs["timepoint"]  = timepoint
    adata.obs["dataset"]    = "GSE269669"
    adata.obs["outcome"]    = "TP53_venetoclax_magrolimab"

    # Transfer any existing cell type annotation
    if celltype_col and meta is not None:
        common = adata.obs_names.intersection(meta.index)
        if len(common) > 0:
            adata.obs["seurat_celltype"] = meta.loc[
                meta.index.isin(common), celltype_col].reindex(adata.obs_names)
            log.info("  Transferred %s annotation from Seurat", celltype_col)

    adata.obs_names = [f"{sid}_{bc}" for bc in adata.obs_names]

    n_raw = adata.n_obs
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)

    keep = (mad_filter(adata.obs["total_counts"]) &
            mad_filter(adata.obs["n_genes_by_counts"]) &
            (adata.obs["pct_counts_mt"] < MT_CAP * 100))
    adata = adata[keep].copy()

    if adata.n_obs >= 50:
        try:
            scrub = scr.Scrublet(adata.X, random_state=42)
            scores, doublets = scrub.scrub_doublets(
                min_counts=2, min_cells=3,
                n_prin_comps=min(20, adata.n_obs // 3), verbose=False)
            adata.obs["doublet_score"] = scores
            adata.obs["is_doublet"]    = doublets
            adata = adata[~adata.obs["is_doublet"]].copy()
        except Exception as e:
            log.warning("  Scrublet failed: %s", e)

    log.info("  %s: %d → %d cells (patient=%s, timepoint=%s)",
             sid, n_raw, adata.n_obs, patient_id, timepoint)
    qc_records.append({
        "sample_id": sid, "patient_id": patient_id, "timepoint": timepoint,
        "n_raw": n_raw, "n_final": adata.n_obs,
        "median_genes": int(adata.obs["n_genes_by_counts"].median()),
        "median_umi":   int(adata.obs["total_counts"].median()),
    })
    adatas.append(adata)

if not adatas:
    log.error("No samples loaded.")
    sys.exit(1)

if len(adatas) > 1:
    try:
        import anndata as ad
        adata_all = ad.concat(adatas, join="outer", fill_value=0)
    except Exception:
        adata_all = adatas[0].concatenate(adatas[1:], join="outer", fill_value=0)
else:
    adata_all = adatas[0]
log.info("Concatenated: %d cells × %d genes", adata_all.n_obs, adata_all.n_vars)

pd.DataFrame(qc_records).to_csv(
    os.path.join(TAB_DIR, "gse269669_qc_stats.tsv"), sep="\t", index=False)

# ── Normalise → HVG → PCA → Harmony → UMAP ───────────────────────────────────
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

# ── Cell type annotation ───────────────────────────────────────────────────────
for pop, genes in MARKERS.items():
    present = [g for g in genes if g in adata_all.var_names]
    if len(present) >= 2:
        sc.tl.score_genes(adata_all, present, score_name=f"score_{pop}",
                          random_state=42)

score_cols = [f"score_{p}" for p in MARKERS if f"score_{p}" in adata_all.obs.columns]
if score_cols:
    cluster_scores = adata_all.obs.groupby("leiden")[score_cols].mean() \
        .rename(columns=lambda c: c.replace("score_", ""))
    adata_all.obs["cell_type"] = adata_all.obs["leiden"].map(
        cluster_scores.idxmax(axis=1))

log.info("Cell types:\n%s", adata_all.obs["cell_type"].value_counts().to_string())

# ── UMAP figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sc.pl.umap(adata_all, color="cell_type", ax=axes[0], show=False, title="Cell type")
sc.pl.umap(adata_all, color="timepoint", ax=axes[1], show=False, title="Timepoint")
sc.pl.umap(adata_all, color="patient_id", ax=axes[2], show=False, title="Patient")
plt.suptitle("GSE269669 — ven+AZA+magrolimab, 11 TP53-mut patients", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gse269669_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

adata_all.write_h5ad(os.path.join(PROC_DIR, "gse269669_annotated.h5ad"))
log.info("=== GSE269669 QC + annotation complete ===")
