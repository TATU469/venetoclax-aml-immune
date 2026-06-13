"""
07_functional_analysis.py — Four analytical extensions for the venetoclax paper

1. Cytotoxic functional scoring (NKG7/GNLY/GZMB/PRF1) pre vs post in NK + CD8
2. Checkpoint/exhaustion scoring (TIGIT/LAG3/PD-1/HAVCR2) pre vs post
3. NKG2D ligand expression on blasts pre vs post (MICA/MICB/ULBPs)
4. Van Galen integration — project pre/post cells onto AML hierarchy

Inputs:
  data/processed/gse306339_annotated.h5ad
  data/processed/gse116256_annotated.h5ad

Outputs:
  results/figures/functional/  (cytotoxic, exhaustion, nkg2d, integration plots)
  results/tables/functional/   (paired stats for each analysis)
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT  = sys.argv[1] if len(sys.argv) > 1 else "."
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures/functional")
TAB_DIR  = os.path.join(PROJECT, "results/tables/functional")
for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

sc.settings.seed = 42
np.random.seed(42)

# ── Gene panels ───────────────────────────────────────────────────────────────
CYTOTOXIC_GENES  = ["NKG7", "GNLY", "GZMB", "PRF1", "IFNG", "GZMA", "GZMH"]
CHECKPOINT_GENES = ["TIGIT", "LAG3", "PDCD1", "HAVCR2", "CTLA4", "VSIR"]
NKG2D_LIGANDS    = ["MICA", "MICB", "ULBP1", "ULBP2", "ULBP3", "ULBP4",
                    "RAET1E", "RAET1G", "RAET1L"]

COLORS = {"pre": "#e74c3c", "post": "#2980b9"}
N_BOOT = 1000


def paired_stats(df, score_col, group_col="timepoint",
                 group_a="pre", group_b="post", patient_col="patient_id"):
    """Patient-level mean scores → paired Wilcoxon + bootstrap median diff."""
    means = df.groupby([patient_col, group_col])[score_col].mean().reset_index()
    pivot = means.pivot(index=patient_col, columns=group_col, values=score_col).dropna()
    if len(pivot) < 2:
        return None
    a, b = pivot[group_a].values, pivot[group_b].values
    _, p  = stats.wilcoxon(a, b) if len(pivot) >= 3 else (None, 1.0)
    diffs = b - a
    boot  = [np.median(np.random.choice(diffs, len(diffs), replace=True))
             for _ in range(N_BOOT)]
    return {
        "n_patients": len(pivot),
        "mean_pre": a.mean(), "mean_post": b.mean(),
        "median_diff": np.median(diffs),
        "ci_lo": np.percentile(boot, 2.5),
        "ci_hi": np.percentile(boot, 97.5),
        "wilcoxon_p": p,
        "pivot": pivot,
    }


def spaghetti_ax(ax, pivot, title, ylabel):
    for pid in pivot.index:
        ax.plot([0, 1], [pivot.loc[pid, "pre"], pivot.loc[pid, "post"]],
                "o-", color="grey", alpha=0.6, lw=1.5, ms=5)
    ax.scatter([0]*len(pivot), pivot["pre"],  c=COLORS["pre"],  zorder=5, s=50)
    ax.scatter([1]*len(pivot), pivot["post"], c=COLORS["post"], zorder=5, s=50)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.4, 1.4)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Loading GSE306339 annotated...")
adata = sc.read_h5ad(os.path.join(PROC_DIR, "gse306339_annotated.h5ad"))

# Restore normalised expression for scoring
if adata.raw is not None:
    adata_norm = adata.raw.to_adata()
    adata_norm.obs  = adata.obs.copy()
    adata_norm.obsm = adata.obsm.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
else:
    adata_norm = adata

log.info("  %d cells, patients: %s",
         adata.n_obs, sorted(adata.obs["patient_id"].unique()))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CYTOTOXIC FUNCTIONAL SCORING
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== 1. Cytotoxic functional scoring ===")

cyto_present = [g for g in CYTOTOXIC_GENES if g in adata_norm.var_names]
log.info("  Cytotoxic genes present: %d/%d: %s",
         len(cyto_present), len(CYTOTOXIC_GENES), cyto_present)
sc.tl.score_genes(adata_norm, cyto_present, score_name="cytotoxic_score", random_state=42)
adata.obs["cytotoxic_score"] = adata_norm.obs["cytotoxic_score"].values

cyto_results = []
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for i, ct in enumerate(["NK_cell", "CD8_T_cell", "CD4_T_cell"]):
    sub = adata.obs[adata.obs["cell_type"] == ct].copy()
    r = paired_stats(sub, "cytotoxic_score")
    if r:
        p_str = f"p={r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] < 1 else "n.s."
        title = (f"{ct}\nΔ={r['median_diff']:+.3f} "
                 f"({r['ci_lo']:+.3f}–{r['ci_hi']:+.3f})\n{p_str}")
        spaghetti_ax(axes[i], r["pivot"], title, "Cytotoxic score")
        cyto_results.append({"cell_type": ct, "score": "cytotoxic", **{k: v
                              for k, v in r.items() if k != "pivot"}})
        log.info("  %s: pre=%.3f post=%.3f Δ=%.3f (%+.3f–%+.3f) p=%.3f",
                 ct, r["mean_pre"], r["mean_post"],
                 r["median_diff"], r["ci_lo"], r["ci_hi"], r["wilcoxon_p"])

plt.suptitle("Cytotoxic gene expression score — pre vs post venetoclax+AZA",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cytotoxic_score_pre_post.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECKPOINT / EXHAUSTION SCORING
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== 2. Checkpoint/exhaustion scoring ===")

chk_present = [g for g in CHECKPOINT_GENES if g in adata_norm.var_names]
log.info("  Checkpoint genes present: %d/%d: %s",
         len(chk_present), len(CHECKPOINT_GENES), chk_present)
sc.tl.score_genes(adata_norm, chk_present, score_name="checkpoint_score", random_state=42)
adata.obs["checkpoint_score"] = adata_norm.obs["checkpoint_score"].values

# Also score individual checkpoint genes for granularity
for g in chk_present:
    adata.obs[f"expr_{g}"] = adata_norm[:, g].X.toarray().flatten() \
        if hasattr(adata_norm[:, g].X, "toarray") \
        else np.array(adata_norm[:, g].X).flatten()

chk_results = []
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
panel = 0

for ct in ["NK_cell", "CD8_T_cell", "CD4_T_cell"]:
    sub = adata.obs[adata.obs["cell_type"] == ct].copy()

    # Composite checkpoint score
    r = paired_stats(sub, "checkpoint_score")
    if r:
        p_str = f"p={r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] < 1 else "n.s."
        spaghetti_ax(axes[panel],
                     r["pivot"],
                     f"{ct}\nCheckpoint score\n{p_str}",
                     "Checkpoint score")
        chk_results.append({"cell_type": ct, "score": "checkpoint", **{k: v
                             for k, v in r.items() if k != "pivot"}})
        log.info("  %s checkpoint: pre=%.3f post=%.3f Δ=%.3f p=%.3f",
                 ct, r["mean_pre"], r["mean_post"],
                 r["median_diff"], r["wilcoxon_p"])
    panel += 1

    # Cytotoxic vs checkpoint scatter per timepoint
    if panel < len(axes):
        ax = axes[panel]
        for tp, color in COLORS.items():
            sub_tp = sub[sub["timepoint"] == tp]
            ax.scatter(sub_tp["cytotoxic_score"], sub_tp["checkpoint_score"],
                       c=color, s=3, alpha=0.3, label=tp, linewidths=0)
        ax.set_xlabel("Cytotoxic score", fontsize=8)
        ax.set_ylabel("Checkpoint score", fontsize=8)
        ax.set_title(f"{ct} — cytotoxic vs checkpoint", fontsize=9)
        ax.legend(fontsize=7, markerscale=3)
        panel += 1

for ax in axes[panel:]:
    ax.set_visible(False)

plt.suptitle("Checkpoint/exhaustion — pre vs post venetoclax+AZA", fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "checkpoint_score_pre_post.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NKG2D LIGAND EXPRESSION ON BLASTS
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== 3. NKG2D ligand expression on blasts ===")

nkg2d_present = [g for g in NKG2D_LIGANDS if g in adata_norm.var_names]
log.info("  NKG2D ligands present: %d/%d: %s",
         len(nkg2d_present), len(NKG2D_LIGANDS), nkg2d_present)

nkg2d_results = []

if nkg2d_present:
    sc.tl.score_genes(adata_norm, nkg2d_present, score_name="nkg2d_ligand_score",
                      random_state=42)
    adata.obs["nkg2d_ligand_score"] = adata_norm.obs["nkg2d_ligand_score"].values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, ct in enumerate(["Blast", "GMP", "Monocyte"]):
        sub = adata.obs[adata.obs["cell_type"] == ct].copy()
        if len(sub) < 10:
            axes[i].set_visible(False)
            continue
        r = paired_stats(sub, "nkg2d_ligand_score")
        if r:
            p_str = f"p={r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] < 1 else "n.s."
            spaghetti_ax(axes[i], r["pivot"],
                         f"{ct}\nNKG2D ligand score\n{p_str}",
                         "NKG2D ligand score")
            nkg2d_results.append({"cell_type": ct, "score": "nkg2d_ligand", **{k: v
                                   for k, v in r.items() if k != "pivot"}})
            log.info("  %s NKG2D ligands: pre=%.3f post=%.3f Δ=%.3f p=%.3f",
                     ct, r["mean_pre"], r["mean_post"],
                     r["median_diff"], r["wilcoxon_p"])

    plt.suptitle("NKG2D ligand expression — pre vs post venetoclax+AZA", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "nkg2d_ligand_pre_post.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-gene heatmap on blasts
    blast_cells = adata.obs["cell_type"] == "Blast"
    if blast_cells.sum() > 0:
        gene_means = []
        for tp in ["pre", "post"]:
            mask = blast_cells & (adata.obs["timepoint"] == tp)
            if mask.sum() == 0:
                continue
            vals = adata_norm[mask, nkg2d_present].X
            if hasattr(vals, "toarray"):
                vals = vals.toarray()
            gene_means.append(pd.Series(vals.mean(axis=0),
                                         index=nkg2d_present, name=tp))
        if len(gene_means) == 2:
            gm_df = pd.DataFrame(gene_means)
            lfc = np.log2((gm_df.loc["post"] + 0.01) / (gm_df.loc["pre"] + 0.01))
            lfc = lfc.sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(7, 4))
            colors_bar = ["#2980b9" if v > 0 else "#e74c3c" for v in lfc]
            ax.barh(range(len(lfc)), lfc, color=colors_bar, alpha=0.85)
            ax.set_yticks(range(len(lfc)))
            ax.set_yticklabels(lfc.index, fontsize=9)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("log₂FC (post / pre)", fontsize=10)
            ax.set_title("NKG2D ligand expression in Blasts\n(post vs pre venetoclax+AZA)",
                         fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "nkg2d_ligand_blast_lfc.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
else:
    log.warning("No NKG2D ligands found in variable genes — may have been filtered at HVG step.")
    # Fallback: check raw
    if adata.raw is not None:
        nkg2d_raw = [g for g in NKG2D_LIGANDS if g in adata.raw.var_names]
        log.info("  NKG2D ligands in raw: %s", nkg2d_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VAN GALEN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=== 4. Van Galen integration ===")

log.info("  Loading GSE116256 annotated...")
ref = sc.read_h5ad(os.path.join(PROC_DIR, "gse116256_annotated.h5ad"))
log.info("  Reference: %d cells, cell types: %s",
         ref.n_obs, sorted(ref.obs["cell_type"].unique()))

# Tag datasets
adata.obs["dataset"] = "GSE306339_venetoclax"
ref.obs["dataset"]   = "GSE116256_vanGalen"

# Use van Galen annotations where available
if "vangalen_celltype" in ref.obs.columns:
    ref.obs["label"] = (ref.obs["vangalen_celltype"].astype(str)
                        .replace("nan", np.nan)
                        .fillna(ref.obs["cell_type"].astype(str)))
else:
    ref.obs["label"] = ref.obs["cell_type"].astype(str)

# Restore raw counts for both
def get_raw_adata(ad):
    if ad.raw is not None:
        out = ad.raw.to_adata()
        out.obs = ad.obs.copy()
        return out
    return ad.copy()

adata_raw = get_raw_adata(adata)
ref_raw   = get_raw_adata(ref)

# Common genes
common_genes = adata_raw.var_names.intersection(ref_raw.var_names)
log.info("  Common genes: %d", len(common_genes))

adata_sub = adata_raw[:, common_genes].copy()
ref_sub   = ref_raw[:, common_genes].copy()

# Concatenate
combined = adata_sub.concatenate(ref_sub, join="inner",
                                  batch_key="batch",
                                  batch_categories=["venetoclax", "vangalen"])
log.info("  Combined: %d cells × %d genes", combined.n_obs, combined.n_vars)

# Normalise
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)

# HVG on combined
sc.pp.highly_variable_genes(combined, n_top_genes=3000, flavor="seurat_v3",
                             batch_key="batch")
combined_hvg = combined[:, combined.var["highly_variable"]].copy()

sc.pp.scale(combined_hvg, max_value=10)
sc.tl.pca(combined_hvg, n_comps=50, svd_solver="arpack", random_state=42)

# Harmony by dataset
import harmonypy as hm
log.info("  Running Harmony on combined dataset...")
ho = hm.run_harmony(combined_hvg.obsm["X_pca"], combined_hvg.obs,
                    ["batch"], random_state=42)
z = ho.Z_corr
if z.shape[0] != combined_hvg.n_obs:
    z = z.T
combined_hvg.obsm["X_pca_harmony"] = z

sc.pp.neighbors(combined_hvg, use_rep="X_pca_harmony", n_neighbors=15,
                n_pcs=30, random_state=42)
sc.tl.umap(combined_hvg, random_state=42)

# Transfer UMAP coords back to combined
combined.obsm["X_umap"] = combined_hvg.obsm["X_umap"]
combined.obs["dataset"]    = combined_hvg.obs["dataset"].astype(str)
combined.obs["cell_type"]  = combined.obs["cell_type"].astype(str)
combined.obs["timepoint"]  = combined.obs["timepoint"].astype(str).replace("nan", "reference")

# ── Integration UMAPs ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: by dataset, by cell type, by timepoint
sc.pl.umap(combined, color="dataset",    ax=axes[0, 0], show=False,
           title="Dataset", palette={"GSE306339_venetoclax": "#e74c3c",
                                     "GSE116256_vanGalen": "#2980b9"})
sc.pl.umap(combined, color="cell_type",  ax=axes[0, 1], show=False, title="Cell type")

# Timepoint — only venetoclax cells are coloured
timepoint_col = combined.obs["timepoint"].astype(str).replace("nan", "reference")
combined.obs["timepoint_display"] = timepoint_col
sc.pl.umap(combined, color="timepoint_display", ax=axes[0, 2], show=False,
           title="Timepoint (venetoclax only)")

# Row 2: residual blast analysis
# Highlight blasts from venetoclax dataset coloured by timepoint
blast_mask = combined.obs["cell_type"] == "Blast"
venet_mask  = combined.obs["dataset"] == "GSE306339_venetoclax"

blast_coords = combined[blast_mask & venet_mask].obsm["X_umap"]
pre_blast  = combined[blast_mask & venet_mask & (combined.obs["timepoint"] == "pre")]
post_blast = combined[blast_mask & venet_mask & (combined.obs["timepoint"] == "post")]

# Plot all cells as background
axes[1, 0].scatter(combined.obsm["X_umap"][:, 0], combined.obsm["X_umap"][:, 1],
                   c="#e0e0e0", s=2, alpha=0.1, linewidths=0)
axes[1, 0].scatter(pre_blast.obsm["X_umap"][:, 0],  pre_blast.obsm["X_umap"][:, 1],
                   c="#e74c3c", s=10, alpha=0.7, label=f"Blast pre (n={len(pre_blast)})")
axes[1, 0].scatter(post_blast.obsm["X_umap"][:, 0], post_blast.obsm["X_umap"][:, 1],
                   c="#2980b9", s=10, alpha=0.7, label=f"Blast post (n={len(post_blast)})")
axes[1, 0].set_title("Residual blasts on AML hierarchy", fontsize=9)
axes[1, 0].legend(fontsize=7)
axes[1, 0].set_xlabel("UMAP1", fontsize=8); axes[1, 0].set_ylabel("UMAP2", fontsize=8)

# Van Galen cell type labels on the reference
if "vangalen_celltype" in combined.obs.columns:
    sc.pl.umap(combined[combined.obs["dataset"] == "GSE116256_vanGalen"],
               color="vangalen_celltype", ax=axes[1, 1], show=False,
               title="Van Galen annotations (reference)")
else:
    sc.pl.umap(combined[combined.obs["dataset"] == "GSE116256_vanGalen"],
               color="cell_type", ax=axes[1, 1], show=False,
               title="Van Galen cell types (reference)")

# NK expansion: highlight NK cells by timepoint
nk_venet = combined[venet_mask & (combined.obs["cell_type"] == "NK_cell")]
pre_nk   = nk_venet[nk_venet.obs["timepoint"] == "pre"]
post_nk  = nk_venet[nk_venet.obs["timepoint"] == "post"]
axes[1, 2].scatter(combined.obsm["X_umap"][:, 0], combined.obsm["X_umap"][:, 1],
                   c="#e0e0e0", s=2, alpha=0.1, linewidths=0)
axes[1, 2].scatter(pre_nk.obsm["X_umap"][:, 0],  pre_nk.obsm["X_umap"][:, 1],
                   c="#e74c3c", s=8, alpha=0.7, label=f"NK pre (n={len(pre_nk)})")
axes[1, 2].scatter(post_nk.obsm["X_umap"][:, 0], post_nk.obsm["X_umap"][:, 1],
                   c="#2980b9", s=8, alpha=0.7, label=f"NK post (n={len(post_nk)})")
axes[1, 2].set_title("NK cell expansion on AML hierarchy", fontsize=9)
axes[1, 2].legend(fontsize=7)
axes[1, 2].set_xlabel("UMAP1", fontsize=8); axes[1, 2].set_ylabel("UMAP2", fontsize=8)

plt.suptitle("Van Galen AML hierarchy integration — venetoclax pre/post projected",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "vangalen_integration_umap.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# Save integrated h5ad
combined.write_h5ad(os.path.join(PROC_DIR, "integrated_combined.h5ad"))
log.info("  Saved integrated h5ad.")

# ── Blast neighbourhood analysis ──────────────────────────────────────────────
# For pre vs post blasts: what van Galen cell types are their nearest neighbours?
log.info("  Analysing blast neighbourhood in reference space...")
blast_venet_idx = np.where(blast_mask & venet_mask)[0]
if len(blast_venet_idx) > 0:
    from sklearn.neighbors import NearestNeighbors
    ref_idx    = np.where(combined.obs["dataset"] == "GSE116256_vanGalen")[0]
    ref_coords = combined_hvg.obsm["X_pca_harmony"][ref_idx]
    blast_coords_harmony = combined_hvg.obsm["X_pca_harmony"][blast_venet_idx]

    nbrs = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(ref_coords)
    _, indices = nbrs.kneighbors(blast_coords_harmony)

    ref_labels = combined.obs.iloc[ref_idx]["cell_type"].values
    blast_obs   = combined.obs.iloc[blast_venet_idx].copy()
    blast_obs["nn_celltypes"] = [
        pd.Series(ref_labels[idx]).value_counts().idxmax()
        for idx in indices
    ]

    nn_summary = blast_obs.groupby(["timepoint", "nn_celltypes"]).size().reset_index(name="n")
    nn_summary["pct"] = nn_summary.groupby("timepoint")["n"].transform(lambda x: 100 * x / x.sum())
    log.info("  Blast nearest-neighbour cell types:\n%s", nn_summary.to_string(index=False))
    nn_summary.to_csv(os.path.join(TAB_DIR, "blast_nn_hierarchy.tsv"), sep="\t", index=False)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, tp in enumerate(["pre", "post"]):
        sub = nn_summary[nn_summary["timepoint"] == tp].sort_values("pct", ascending=True)
        axes[i].barh(sub["nn_celltypes"], sub["pct"], color=COLORS[tp], alpha=0.85)
        axes[i].set_xlabel("% nearest neighbours", fontsize=9)
        axes[i].set_title(f"Blast NN hierarchy — {tp}\n(n={int(sub['n'].sum())} blasts)", fontsize=9)
        axes[i].set_xlim(0, 100)
    plt.suptitle("Residual blast position in AML hierarchy\n(nearest-neighbour mapping to van Galen reference)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "blast_hierarchy_nn.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════
all_results = cyto_results + chk_results + nkg2d_results
if all_results:
    summary = pd.DataFrame(all_results)[
        ["cell_type", "score", "n_patients", "mean_pre", "mean_post",
         "median_diff", "ci_lo", "ci_hi", "wilcoxon_p"]
    ]
    summary.to_csv(os.path.join(TAB_DIR, "functional_analysis_summary.tsv"),
                   sep="\t", index=False)
    log.info("\n=== FUNCTIONAL ANALYSIS SUMMARY ===\n%s", summary.to_string(index=False))

log.info("=== All four analyses complete ===")
