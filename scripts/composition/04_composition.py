"""
Immune composition analysis — venetoclax pre vs post (GSE306339).

Paired Wilcoxon signed-rank tests with bootstrap CIs on patient-level
cell-type fractions. Primary comparison: pre vs post venetoclax+AZA.

Outputs:
  results/tables/composition_venetoclax_results.tsv
  results/figures/composition_spaghetti.png
  results/figures/composition_forest.png
"""

import sys, os, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "."
PROC_DIR = os.path.join(PROJECT, "data/processed")
FIG_DIR  = os.path.join(PROJECT, "results/figures")
TAB_DIR  = os.path.join(PROJECT, "results/tables")
import scanpy as sc

adata = sc.read_h5ad(os.path.join(PROC_DIR, "gse306339_annotated.h5ad"))
log.info("Loaded: %d cells, %d patients", adata.n_obs, adata.obs["patient_id"].nunique())

# ── Cell-type fractions per patient per timepoint ─────────────────────────────
counts = (adata.obs.groupby(["patient_id", "timepoint", "cell_type"])
          .size().reset_index(name="n"))
totals = counts.groupby(["patient_id", "timepoint"])["n"].sum().reset_index(name="total")
counts = counts.merge(totals, on=["patient_id", "timepoint"])
counts["fraction"] = counts["n"] / counts["total"]

cell_types = sorted(counts["cell_type"].unique())
patients   = sorted(counts["patient_id"].unique())

N_BOOTSTRAP = 1000
BH_ALPHA    = 0.05

results = []

for ct in cell_types:
    ct_data = counts[counts["cell_type"] == ct]
    pivot   = ct_data.pivot_table(index="patient_id", columns="timepoint",
                                   values="fraction").dropna(subset=["pre", "post"])
    n_pairs = len(pivot)
    if n_pairs < 2:
        log.warning("Skipping %s — only %d paired patients", ct, n_pairs)
        continue

    pre_vals  = pivot["pre"].values
    post_vals = pivot["post"].values
    diffs     = post_vals - pre_vals
    fold_changes = (post_vals + 1e-6) / (pre_vals + 1e-6)

    stat, p = stats.wilcoxon(pre_vals, post_vals)

    # Bootstrap CI on median fold change
    boot_fcs = []
    rng = np.random.RandomState(42)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_pairs, n_pairs, replace=True)
        boot_fcs.append(np.median(fold_changes[idx]))
    ci_lo, ci_hi = np.percentile(boot_fcs, [2.5, 97.5])

    results.append({
        "cell_type": ct, "n_patients": n_pairs,
        "mean_pre": pre_vals.mean(), "mean_post": post_vals.mean(),
        "median_fold_change": np.median(fold_changes),
        "fc_ci_lo": ci_lo, "fc_ci_hi": ci_hi,
        "wilcoxon_p": p,
    })
    log.info("%s (n=%d): pre=%.3f post=%.3f FC=%.2f (%.2f–%.2f) p=%.4f",
             ct, n_pairs, pre_vals.mean(), post_vals.mean(),
             np.median(fold_changes), ci_lo, ci_hi, p)

results_df = pd.DataFrame(results)
# BH correction
from statsmodels.stats.multitest import multipletests
_, padj, _, _ = multipletests(results_df["wilcoxon_p"], method="fdr_bh")
results_df["padj"] = padj
results_df["significant"] = results_df["padj"] < BH_ALPHA

results_df.to_csv(os.path.join(TAB_DIR, "composition_venetoclax_results.tsv"),
                  sep="\t", index=False)
log.info("\nSignificant changes:\n%s",
         results_df[results_df["significant"]][
             ["cell_type", "n_patients", "median_fold_change", "fc_ci_lo", "fc_ci_hi", "padj"]
         ].to_string(index=False))

# ── Spaghetti plots for key cell types ────────────────────────────────────────
key_types = results_df.sort_values("wilcoxon_p").head(6)["cell_type"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

COLORS = {"pre": "#e74c3c", "post": "#2980b9"}

for i, ct in enumerate(key_types):
    ax = axes[i]
    ct_data = counts[counts["cell_type"] == ct]
    pivot = ct_data.pivot_table(index="patient_id", columns="timepoint",
                                 values="fraction").dropna(subset=["pre", "post"])
    for pid in pivot.index:
        ax.plot([0, 1], [pivot.loc[pid, "pre"], pivot.loc[pid, "post"]],
                "o-", color="grey", alpha=0.6, linewidth=1.5, markersize=5)
    ax.scatter([0] * len(pivot), pivot["pre"],  color=COLORS["pre"],  zorder=5, s=40)
    ax.scatter([1] * len(pivot), pivot["post"], color=COLORS["post"], zorder=5, s=40)

    row = results_df[results_df["cell_type"] == ct].iloc[0]
    p_str = f"p={row['wilcoxon_p']:.3f}" if row["wilcoxon_p"] >= 0.001 else "p<0.001"
    sig_str = " *" if row["significant"] else ""
    ax.set_title(f"{ct}\nFC={row['median_fold_change']:.2f} {p_str}{sig_str}", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"], fontsize=9)
    ax.set_ylabel("Fraction", fontsize=8)
    ax.set_xlim(-0.3, 1.3)

for ax in axes[len(key_types):]:
    ax.set_visible(False)

plt.suptitle("Cell-type fractions — pre vs post venetoclax+AZA (GSE306339)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "composition_spaghetti.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Forest plot ───────────────────────────────────────────────────────────────
forest_df = results_df.sort_values("median_fold_change")
fig, ax = plt.subplots(figsize=(8, max(4, len(forest_df) * 0.45)))
y_pos = range(len(forest_df))
colors = ["#e74c3c" if sig else "#7f8c8d" for sig in forest_df["significant"]]

ax.scatter(forest_df["median_fold_change"], y_pos, c=colors, zorder=5, s=50)
ax.errorbar(forest_df["median_fold_change"], y_pos,
            xerr=[forest_df["median_fold_change"] - forest_df["fc_ci_lo"],
                  forest_df["fc_ci_hi"] - forest_df["median_fold_change"]],
            fmt="none", color="grey", linewidth=1.2, capsize=3)
ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(forest_df["cell_type"], fontsize=9)
ax.set_xlabel("Median fold change (post / pre)", fontsize=10)
ax.set_title("Cell composition changes after venetoclax+AZA\n(red = FDR < 0.05)", fontsize=10)

sig_patch = mpatches.Patch(color="#e74c3c", label="FDR < 0.05")
ns_patch  = mpatches.Patch(color="#7f8c8d", label="Not significant")
ax.legend(handles=[sig_patch, ns_patch], fontsize=8, loc="lower right")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "composition_forest.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("=== Composition analysis complete ===")
