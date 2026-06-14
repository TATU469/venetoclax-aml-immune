"""
GSE269669-only composition analysis — 11 TP53-mut patients, ven+AZA+magrolimab.

Uses Azimuth L2 cell type labels (seurat_celltype) for accurate NK vs CD8-T distinction.
Runs paired Wilcoxon pre vs post (n=11) and pre vs post_c2 (n= patients with C sample).

Outputs:
  results/tables/validation/gse269669_composition_results.tsv
  results/figures/validation/gse269669_spaghetti.png
  results/figures/validation/gse269669_forest.png
  results/figures/validation/gse269669_longitudinal.png
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats
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

N_BOOT   = 1000
BH_ALPHA = 0.05

# Azimuth L2 → broad category mapping
AZIMUTH_MAP = {
    "NK":              "NK_cell",
    "NK CD56+":        "NK_cell",
    "ILC":             "NK_cell",
    "CD4 Memory":      "CD4_T_cell",
    "CD4 Naive":       "CD4_T_cell",
    "CD4 Effector":    "CD4_T_cell",
    "Treg":            "CD4_T_cell",
    "CD8 Effector_1":  "CD8_T_cell",
    "CD8 Effector_2":  "CD8_T_cell",
    "CD8 Memory":      "CD8_T_cell",
    "CD8 Naive":       "CD8_T_cell",
    "MAIT":            "CD8_T_cell",
    "Memory B":        "B_cell",
    "Naive B":         "B_cell",
    "Transitional B":  "B_cell",
    "Plasma":          "B_cell",
    "CD14 Mono":       "Monocyte",
    "CD16 Mono":       "Monocyte",
    "Macrophage":      "Monocyte",
    "cDC1":            "DC",
    "cDC2":            "DC",
    "pDC":             "DC",
    "Late Eryth":      "Erythroid",
    "Early Eryth":     "Erythroid",
    "GMP":             "Progenitor",
    "HSC":             "Progenitor",
    "LMPP":            "Progenitor",
    "Prog Mk":         "Progenitor",
    "Platelet":        "Progenitor",
}

# ── Load GSE269669 ────────────────────────────────────────────────────────────
log.info("Loading GSE269669...")
h5_path = os.path.join(PROC_DIR, "gse269669_annotated.h5ad")
if not os.path.exists(h5_path):
    log.error("gse269669_annotated.h5ad not found — run 13_qc_gse269669.py first")
    sys.exit(1)
adata = sc.read_h5ad(h5_path)
log.info("Loaded: %d cells × %d genes, %d patients",
         adata.n_obs, adata.n_vars, adata.obs["patient_id"].nunique())
log.info("Timepoints: %s", dict(adata.obs["timepoint"].value_counts()))

# Use Azimuth L2 annotation, fall back to marker-based
if "seurat_celltype" in adata.obs.columns:
    adata.obs["cell_type_broad"] = (
        adata.obs["seurat_celltype"].astype(str).map(AZIMUTH_MAP).fillna("Other")
    )
    log.info("Using Azimuth L2 annotation (seurat_celltype)")
    log.info("Broad categories:\n%s",
             adata.obs["cell_type_broad"].value_counts().to_string())
else:
    adata.obs["cell_type_broad"] = adata.obs["cell_type"]
    log.info("Azimuth annotation not found — using marker-based cell_type")

obs = adata.obs[["patient_id", "timepoint", "cell_type_broad",
                  "seurat_celltype" if "seurat_celltype" in adata.obs.columns
                  else "cell_type"]].copy()
for col in obs.columns:
    obs[col] = obs[col].astype(str)

# ── Helper: paired Wilcoxon composition test ─────────────────────────────────
def run_composition(obs_df, tp1="pre", tp2="post", label=""):
    sub = obs_df[obs_df["timepoint"].isin([tp1, tp2])].copy()
    counts = (sub.groupby(["patient_id", "timepoint", "cell_type_broad"])
              .size().reset_index(name="n"))
    totals = counts.groupby(["patient_id", "timepoint"])["n"].sum().reset_index(name="total")
    counts = counts.merge(totals, on=["patient_id", "timepoint"])
    counts["fraction"] = counts["n"] / counts["total"]

    cell_types = sorted(counts["cell_type_broad"].unique())
    results = []
    for ct in cell_types:
        if ct == "Other":
            continue
        ct_data = counts[counts["cell_type_broad"] == ct]
        pivot = ct_data.pivot_table(index="patient_id", columns="timepoint",
                                     values="fraction").reset_index()
        pivot = pivot.dropna(subset=[tp1, tp2])
        n = len(pivot)
        if n < 3:
            log.warning("Skipping %s (%s vs %s) — only %d paired patients", ct, tp1, tp2, n)
            continue
        pre_vals  = pivot[tp1].values
        post_vals = pivot[tp2].values
        fc = (post_vals + 1e-6) / (pre_vals + 1e-6)
        _, p = stats.wilcoxon(pre_vals, post_vals)
        rng = np.random.RandomState(42)
        boot = [np.median(fc[rng.choice(n, n, replace=True)]) for _ in range(N_BOOT)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        results.append({
            "cell_type": ct, "comparison": f"{tp1}_vs_{tp2}", "n_patients": n,
            "mean_pre": pre_vals.mean(), "mean_post": post_vals.mean(),
            "median_fc": np.median(fc), "fc_ci_lo": ci_lo, "fc_ci_hi": ci_hi,
            "wilcoxon_p": p,
        })
        log.info("%s %s (n=%d): %s=%.3f %s=%.3f FC=%.2f (%.2f–%.2f) p=%.4f",
                 ct, label, n, tp1, pre_vals.mean(), tp2, post_vals.mean(),
                 np.median(fc), ci_lo, ci_hi, p)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    _, padj, _, _ = multipletests(df["wilcoxon_p"], method="fdr_bh")
    df["padj"] = padj
    df["significant"] = df["padj"] < BH_ALPHA
    return df

# ── Primary: pre vs post (n=11) ───────────────────────────────────────────────
log.info("\n=== Pre vs Post (n=11) ===")
res_pre_post = run_composition(obs, "pre", "post", "[pre→post]")

log.info("\nSignificant (FDR<0.05) pre vs post:\n%s",
         res_pre_post[res_pre_post["significant"]][
             ["cell_type", "n_patients", "median_fc", "padj"]].to_string(index=False)
         if not res_pre_post.empty and res_pre_post["significant"].any() else "  None")
log.info("\nAll pre vs post results:\n%s",
         res_pre_post[["cell_type", "n_patients", "median_fc",
                        "wilcoxon_p", "padj"]].to_string(index=False)
         if not res_pre_post.empty else "  No results")

# ── Secondary: pre vs post_c2 ─────────────────────────────────────────────────
n_c2 = obs[obs["timepoint"] == "post_c2"]["patient_id"].nunique()
log.info("\n=== Pre vs Post_C2 (n=%d) ===", n_c2)
res_pre_c2 = run_composition(obs, "pre", "post_c2", "[pre→post_c2]")

# ── Save results ──────────────────────────────────────────────────────────────
all_res = pd.concat([res_pre_post, res_pre_c2], ignore_index=True)
all_res.to_csv(os.path.join(TAB_DIR, "gse269669_composition_results.tsv"),
               sep="\t", index=False)

# ── Spaghetti plot — pre vs post ──────────────────────────────────────────────
if not res_pre_post.empty:
    counts_pp = (obs[obs["timepoint"].isin(["pre", "post"])]
                 .groupby(["patient_id", "timepoint", "cell_type_broad"])
                 .size().reset_index(name="n"))
    totals_pp = counts_pp.groupby(["patient_id", "timepoint"])["n"].sum().reset_index(name="total")
    counts_pp = counts_pp.merge(totals_pp, on=["patient_id", "timepoint"])
    counts_pp["fraction"] = counts_pp["n"] / counts_pp["total"]

    key_types = res_pre_post.nsmallest(6, "wilcoxon_p")["cell_type"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, ct in enumerate(key_types):
        ax = axes[i]
        pivot = (counts_pp[counts_pp["cell_type_broad"] == ct]
                 .pivot_table(index="patient_id", columns="timepoint", values="fraction")
                 .dropna(subset=["pre", "post"]))
        for pid in pivot.index:
            ax.plot([0, 1], [pivot.loc[pid, "pre"], pivot.loc[pid, "post"]],
                    "o-", color="#7f8c8d", alpha=0.6, lw=1.5, ms=5)
        # Median line
        ax.plot([0, 1], [pivot["pre"].median(), pivot["post"].median()],
                "o-", color="#2c3e50", lw=3, ms=8, zorder=10, label="Median")
        row = res_pre_post[res_pre_post["cell_type"] == ct].iloc[0]
        sig = " *" if row["padj"] < BH_ALPHA else (" †" if row["wilcoxon_p"] < 0.05 else "")
        p_str = f"p={row['wilcoxon_p']:.3f}" if row["wilcoxon_p"] >= 0.001 else "p<0.001"
        ax.set_title(f"{ct}\nFC={row['median_fc']:.2f} ({row['fc_ci_lo']:.2f}–{row['fc_ci_hi']:.2f})\n{p_str}{sig}",
                     fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=9)
        ax.set_ylabel("Fraction", fontsize=8)
        ax.set_xlim(-0.35, 1.35)

    plt.suptitle("GSE269669: Cell composition pre vs post ven+AZA+magrolimab\n"
                 f"(n=11 TP53-mut AML patients; † p<0.05, * FDR<0.05)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "gse269669_spaghetti.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Forest plot ───────────────────────────────────────────────────────────────
if not res_pre_post.empty:
    forest = res_pre_post.sort_values("median_fc")
    fig, ax = plt.subplots(figsize=(8, max(4, len(forest) * 0.5)))
    colors = ["#e74c3c" if s else ("#e67e22" if p < 0.05 else "#7f8c8d")
              for s, p in zip(forest["significant"], forest["wilcoxon_p"])]
    ax.scatter(forest["median_fc"], range(len(forest)), c=colors, zorder=5, s=60)
    ax.errorbar(forest["median_fc"], range(len(forest)),
                xerr=[forest["median_fc"] - forest["fc_ci_lo"],
                      forest["fc_ci_hi"] - forest["median_fc"]],
                fmt="none", color="grey", lw=1.2, capsize=3)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(forest)))
    ax.set_yticklabels(forest["cell_type"], fontsize=9)
    ax.set_xlabel("Median fold change (post / pre)", fontsize=10)
    ax.set_title("GSE269669 — composition change after ven+AZA+magrolimab\n"
                 "(n=11 TP53-mut; red=FDR<0.05, orange=p<0.05)", fontsize=10)
    legend_els = [mpatches.Patch(color="#e74c3c", label="FDR<0.05"),
                  mpatches.Patch(color="#e67e22", label="p<0.05"),
                  mpatches.Patch(color="#7f8c8d", label="n.s.")]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gse269669_forest.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Longitudinal plot: pre → post → post_c2 ──────────────────────────────────
key_immune = ["NK_cell", "CD8_T_cell", "CD4_T_cell", "B_cell", "Monocyte"]
key_immune = [c for c in key_immune if c in obs["cell_type_broad"].unique()]

counts_all = (obs.groupby(["patient_id", "timepoint", "cell_type_broad"])
              .size().reset_index(name="n"))
totals_all = counts_all.groupby(["patient_id", "timepoint"])["n"].sum().reset_index(name="total")
counts_all = counts_all.merge(totals_all, on=["patient_id", "timepoint"])
counts_all["fraction"] = counts_all["n"] / counts_all["total"]

tp_order = {"pre": 0, "post": 1, "post_c2": 2}
fig, axes = plt.subplots(1, len(key_immune), figsize=(4 * len(key_immune), 5))
if len(key_immune) == 1:
    axes = [axes]

for ax, ct in zip(axes, key_immune):
    ct_data = counts_all[counts_all["cell_type_broad"] == ct].copy()
    ct_data["tp_num"] = ct_data["timepoint"].map(tp_order)
    for pid in ct_data["patient_id"].unique():
        p_data = ct_data[ct_data["patient_id"] == pid].sort_values("tp_num")
        ax.plot(p_data["tp_num"], p_data["fraction"],
                "o-", color="#7f8c8d", alpha=0.5, lw=1.5, ms=4)
    # Median per timepoint
    med = ct_data.groupby("tp_num")["fraction"].median()
    ax.plot(med.index, med.values, "o-", color="#2c3e50", lw=3, ms=8, zorder=10)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Pre", "Post\n(C1)", "Post\n(C2)"], fontsize=9)
    ax.set_title(ct.replace("_", " "), fontsize=10)
    ax.set_ylabel("Fraction", fontsize=8)

plt.suptitle("GSE269669: Longitudinal immune dynamics — ven+AZA+magrolimab\n"
             f"(n=11 TP53-mut AML; thick line = median)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG_DIR, "gse269669_longitudinal.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

log.info("=== GSE269669 composition analysis complete ===")
