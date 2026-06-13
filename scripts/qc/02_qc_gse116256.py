"""
QC and preprocessing — GSE116256 (van Galen 2019, AML hierarchy reference)

16 AML patients at diagnosis + 5 healthy bone marrow controls.
Files: per-sample TXT matrices (genes × cells) in GSE116256_RAW.tar.

Outputs:
  data/processed/gse116256_raw.h5ad
  data/processed/gse116256_qc.h5ad
  results/tables/gse116256_qc_stats.tsv
  results/figures/gse116256_qc_metrics.png
"""

import sys, os, logging, warnings, glob, gzip, io, re
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT  = sys.argv[1] if len(sys.argv) > 1 else "."
RAW_DIR  = os.path.join(PROJECT, "data/raw/GSE116256")
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures")
TAB_DIR  = os.path.join(PROJECT, "results/tables")
for d in [PROC_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

sc.settings.seed = 42
np.random.seed(42)

MAD_THRESH = 3
MT_CAP     = 0.20

# ── Detect sample files ────────────────────────────────────────────────────────
# van Galen files: GSMxxxxxx_AMLxxx-D0.dem.txt.gz  (count matrices)
#                  GSMxxxxxx_AMLxxx-D0.anno.txt.gz (cell type annotations)
# Healthy BM:      GSMxxxxxx_BM1.dem.txt.gz
# Exclude cell lines (MUTZ3, OCI-AML3) and treatment timepoints (D14, D29, etc.)
# Keep: *-D0.dem.txt.gz (AML diagnosis) + BM*.dem.txt.gz (healthy BM)
dem_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.dem.txt.gz")))
aml_d0 = [f for f in dem_files if "-D0.dem.txt.gz" in os.path.basename(f)]
hbm    = [f for f in dem_files if re.search(r"_BM\d", os.path.basename(f))]
use_files = aml_d0 + hbm
log.info("AML D0: %d files | Healthy BM: %d files | Total: %d",
         len(aml_d0), len(hbm), len(use_files))

def load_txt_gz(path):
    """Load genes×cells TXT matrix from van Galen format."""
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split("\t")
        rows, gene_names = [], []
        for line in f:
            parts = line.strip().split("\t")
            gene_names.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    # First element of header is the gene-column label (empty or "Gene") — skip it
    cell_ids = header[1:] if header[0] in ("", "Gene", "gene", "GENE") else header
    mat = sp.csr_matrix(np.array(rows, dtype=np.float32).T)
    adata = sc.AnnData(X=mat,
                       obs=pd.DataFrame(index=cell_ids),
                       var=pd.DataFrame(index=gene_names))
    return adata

adatas, qc_records = [], []

for fp in use_files:
    fname = os.path.basename(fp)
    # Pattern: GSMxxxxxx_AMLxxx-D0.dem.txt.gz  or  GSMxxxxxx_BM1.dem.txt.gz
    sample_id = fname.split("_", 1)[1].replace(".dem.txt.gz", "")  # e.g. AML1012-D0
    condition = "HBM" if sample_id.startswith("BM") else "AML_Dx"
    anno_path = fp.replace(".dem.txt.gz", ".anno.txt.gz")

    log.info("--- Loading %s (%s) ---", sample_id, condition)
    try:
        adata = load_txt_gz(fp)
    except Exception as e:
        log.warning("  Failed to load %s: %s", fname, e)
        continue

    adata.obs["sample_id"]   = sample_id
    adata.obs["patient_id"]  = sample_id
    adata.obs["condition"]   = condition
    adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]

    # Load van Galen cell type annotations if available
    if os.path.exists(anno_path):
        try:
            anno = pd.read_csv(anno_path, sep="\t", index_col=0, compression="gzip")
            # anno index = cell barcodes, columns include CellType
            cell_type_col = next((c for c in anno.columns if "cell" in c.lower()
                                   or "type" in c.lower() or "anno" in c.lower()), anno.columns[0])
            anno_map = anno[cell_type_col].to_dict()
            adata.obs["vangalen_celltype"] = [
                anno_map.get(bc.split("_")[-1], anno_map.get(bc, "Unknown"))
                for bc in adata.obs_names
            ]
            log.info("  %s: van Galen annotations loaded (%d unique types)",
                     sample_id, adata.obs["vangalen_celltype"].nunique())
        except Exception as e:
            log.warning("  Could not load annotations for %s: %s", sample_id, e)
            adata.obs["vangalen_celltype"] = "Unknown"
    else:
        adata.obs["vangalen_celltype"] = "Unknown"

    n_raw = adata.n_obs
    log.info("  %s: %d cells, %d genes (raw)", sample_id, adata.n_obs, adata.n_vars)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)

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
    log.info("  %s: %d cells retained (%.1f%%)",
             sample_id, adata.n_obs, 100 * adata.n_obs / max(n_raw, 1))

    if adata.n_obs >= 50:
        scrub = scr.Scrublet(adata.X, random_state=42)
        try:
            doublet_scores, predicted_doublets = scrub.scrub_doublets(
                min_counts=2, min_cells=3, n_prin_comps=min(20, adata.n_obs // 3),
                verbose=False
            )
            adata.obs["doublet_score"]  = doublet_scores
            adata.obs["is_doublet"]     = predicted_doublets
            n_before = adata.n_obs
            adata = adata[~adata.obs["is_doublet"]].copy()
            log.info("  %s: %d → %d after doublet removal", sample_id, n_before, adata.n_obs)
        except Exception as e:
            log.warning("  Scrublet failed for %s: %s", sample_id, e)
            adata.obs["doublet_score"] = 0.0
            adata.obs["is_doublet"]    = False

    qc_records.append({
        "sample_id": sample_id, "condition": condition,
        "n_raw": n_raw, "n_final": adata.n_obs,
        "pct_retained": round(100 * adata.n_obs / max(n_raw, 1), 1),
        "median_genes": int(adata.obs["n_genes_by_counts"].median()),
        "median_umi":   int(adata.obs["total_counts"].median()),
    })
    adatas.append(adata)

if not adatas:
    log.error("No samples loaded. Exiting.")
    sys.exit(1)

adata_all = adatas[0].concatenate(adatas[1:], join="outer", fill_value=0) if len(adatas) > 1 else adatas[0]
log.info("Concatenated: %d cells × %d genes", adata_all.n_obs, adata_all.n_vars)

raw_path = os.path.join(PROC_DIR, "gse116256_raw.h5ad")
adata_all.write_h5ad(raw_path)
log.info("Saved raw h5ad: %s", raw_path)

qc_df = pd.DataFrame(qc_records)
qc_df.to_csv(os.path.join(TAB_DIR, "gse116256_qc_stats.tsv"), sep="\t", index=False)
log.info("\n%s", qc_df.to_string(index=False))

# ── Normalise → HVG → PCA → Harmony ──────────────────────────────────────────
log.info("Normalising...")
sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)
adata_all.raw = adata_all.copy()

sc.pp.highly_variable_genes(adata_all, n_top_genes=3000, flavor="seurat_v3",
                             batch_key="sample_id")
adata_all = adata_all[:, adata_all.var["highly_variable"]].copy()

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

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc.pl.umap(adata_all, color="condition",  ax=axes[0], show=False, title="Condition")
sc.pl.umap(adata_all, color="leiden",     ax=axes[1], show=False, title="Leiden clusters")
plt.suptitle("GSE116256 — UMAP (Harmony-corrected, AML Dx + HBM)", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gse116256_umap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

proc_path = os.path.join(PROC_DIR, "gse116256_qc.h5ad")
adata_all.write_h5ad(proc_path)
log.info("Saved processed h5ad: %s", proc_path)
log.info("=== GSE116256 QC complete ===")
