"""
Pooled composition analysis — GSE306339 (n=3) + GSE311458 (n=4) = n=7 patients.

Paired Wilcoxon pre/post venetoclax across both cohorts combined.
Also tests response stratification (responders vs non-responders) using
GSE311458 outcome labels.

Outputs:
  results/tables/validation/pooled_composition_results.tsv
  results/figures/validation/pooled_composition_spaghetti.png
  results/figures/validation/pooled_composition_forest.png
  results/figures/validation/response_stratified_nk.png
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
COLORS   = {"pre": "#e74c3c", "post": "#2980b9"}

# ── Load both datasets ────────────────────────────────────────────────────────
log.info("Loading GSE306339...")
a1 = sc.read_h5ad(os.path.join(PROC_DIR, "gse306339_annotated.h5ad"))
a1.obs["cohort"] = "GSE306339"
if "outcome" not in a1.obs.columns:
    a1.obs["outcome"] = "TP53_venetoclax"

log.info("Loading GSE311458...")
h5_path = os.path.join(PROC_DIR, "gse311458_annotated.h5ad")
if not os.path.exists(h5_path):
    log.error("GSE311458 h5ad not found — run 09_qc_gse311458.py first")
    sys.exit(1)
a2 = sc.read_h5ad(h5_path)
a2.obs["cohort"] = "GSE311458"

# Harmonise obs columns
for col in ["sample_id", "patient_id", "timepoint", "cell_type", "cohort", "outcome"]:
    for ad in [a1, a2]:
        if col not in ad.obs.columns:
            ad.obs[col] = "unknown"

# Concatenate obs only (no need to merge expression for composition)
obs_all = pd.concat([a1.obs[["sample_id", "patient_id", "timepoint",
                               "cell_type", "cohort", "outcome"]],
                      a2.obs[["sample_id", "patient_id", "timepoint",
                               "cell_type", "cohort", "outcome"]]])

# Make patient IDs unique across cohorts
obs_all["patient_uid"] = obs_all["cohort"] + "_" + obs_all["patient_id"].astype(str)

log.info("Pooled: %d cells, %d patients (%d GSE306339 + %d GSE311458)",
         len(obs_all),
         obs_all["patient_uid"].nunique(),
         a1.obs["patient_id"].nunique(),
         a2.obs["patient_id"].nunique())

# ── Cell-type fractions per patient per timepoint ─────────────────────────────
counts = (obs_all.groupby(["patient_uid", "timepoint", "cell_type"])
          .size().reset_index(name="n"))
totals = counts.groupby(["patient_uid", "timepoint"])["n"].sum().reset_index(name="total")
counts = counts.merge(totals, on=["patient_uid", "timepoint"])
counts["fraction"] = counts["n"] / counts["total"]

cell_types = sorted(counts["cell_type"].unique())
results = []

for ct in cell_types:
    ct_data = counts[counts["cell_type"] == ct]
    pivot   = ct_data.pivot_table(index="patient_uid", columns="timepoint",
                                   values="fraction").dropna(subset=["pre", "post"])
    n = len(pivot)
    if n < 3:
        log.warning("Skipping %s — only %d paired patients", ct, n)
        continue

    pre, post = pivot["pre"].values, pivot["post"].values
    fc = (post + 1e-6) / (pre + 1e-6)
    _, p = stats.wilcoxon(pre, post)

    rng = np.random.RandomState(42)
    boot = [np.median(fc[rng.choice(n, n, replace=True)]) for _ in range(N_BOOT)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    results.append({
        "cell_type": ct, "n_patients": n,
        "mean_pre": pre.mean(), "mean_post": post.mean(),
        "median_fc": np.median(fc), "fc_ci_lo": ci_lo, "fc_ci_hi": ci_hi,
        "wilcoxon_p": p,
    })
    log.info("%s (n=%d): pre=%.3f post=%.3f FC=%.2f (%.2f–%.2f) p=%.4f",
             ct, n, pre.mean(), post.mean(), np.median(fc), ci_lo, ci_hi, p)

results_df = pd.DataFrame(results)
_, padj, _, _ = multipletests(results_df["wilcoxon_p"], method="fdr_bh")
results_df["padj"] = padj
results_df["significant"] = results_df["padj"] < BH_ALPHA
results_df.to_csv(os.path.join(TAB_DIR, "pooled_composition_results.tsv"),
                  sep="\t", index=False)

log.info("\nSignificant (FDR<0.05):\n%s",
         results_df[results_df["significant"]][
             ["cell_type", "n_patients", "median_fc", "fc_ci_lo", "fc_ci_hi", "padj"]
         ].to_string(index=False) or "  None")
log.info("\nAll results:\n%s",
         results_df[["cell_type", "n_patients", "median_fc", "wilcoxon_p", "padj"]
                    ].to_string(index=False))

# ── Spaghetti plots ───────────────────────────────────────────────────────────
key_types = results_df.nsmallest(6, "wilcoxon_p")["cell_type"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
cohort_markers = {"GSE306339": "o", "GSE311458": "s"}

for i, ct in enumerate(key_types):
    ax = axes[i]
    ct_data = counts[counts["cell_type"] == ct]
    pivot   = ct_data.pivot_table(index="patient_uid", columns="timepoint",
                                   values="fraction").dropna(subset=["pre", "post"])
    for pid in pivot.index:
        cohort = "GSE306339" if pid.startswith("GSE306339") else "GSE311458"
        ax.plot([0, 1], [pivot.loc[pid, "pre"], pivot.loc[pid, "post"]],
                "-", color="grey", alpha=0.5, lw=1.5)
        ax.scatter([0], [pivot.loc[pid, "pre"]],
                   c=COLORS["pre"],  marker=cohort_markers[cohort], s=45, zorder=5)
        ax.scatter([1], [pivot.loc[pid, "post"]],
                   c=COLORS["post"], marker=cohort_markers[cohort], s=45, zorder=5)

    row = results_df[results_df["cell_type"] == ct].iloc[0]
    sig = " *" if row["significant"] else ""
    p_str = f"p={row['wilcoxon_p']:.3f}" if row["wilcoxon_p"] >= 0.001 else "p<0.001"
    ax.set_title(f"{ct}\nFC={row['median_fc']:.2f} ({row['fc_ci_lo']:.2f}–{row['fc_ci_hi']:.2f})\n{p_str}{sig}",
                 fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"], fontsize=9)
    ax.set_ylabel("Fraction", fontsize=8)
    ax.set_xlim(-0.35, 1.35)

legend_els = [mpatches.Patch(color="white", label="●  GSE306339 (TP53-mut)"),
              mpatches.Patch(color="white", label="■  GSE311458 (mixed)")]
fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=8, frameon=False)
plt.suptitle(f"Pooled composition: pre vs post venetoclax (n=7 patients)", fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(FIG_DIR, "pooled_composition_spaghetti.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Forest plot ───────────────────────────────────────────────────────────────
forest = results_df.sort_values("median_fc")
fig, ax = plt.subplots(figsize=(8, max(4, len(forest) * 0.45)))
colors = ["#e74c3c" if s else "#7f8c8d" for s in forest["significant"]]
ax.scatter(forest["median_fc"], range(len(forest)), c=colors, zorder=5, s=55)
ax.errorbar(forest["median_fc"], range(len(forest)),
            xerr=[forest["median_fc"] - forest["fc_ci_lo"],
                  forest["fc_ci_hi"] - forest["median_fc"]],
            fmt="none", color="grey", lw=1.2, capsize=3)
ax.axvline(1.0, color="black", lw=0.8, ls="--")
ax.set_yticks(range(len(forest)))
ax.set_yticklabels(forest["cell_type"], fontsize=9)
ax.set_xlabel("Median fold change (post / pre)", fontsize=10)
ax.set_title(f"Pooled composition changes — venetoclax+AZA\n"
             f"(n=7 patients; red = FDR<0.05)", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "pooled_composition_forest.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Response stratification (GSE311458 has outcome labels) ────────────────────
obs_311 = obs_all[obs_all["cohort"] == "GSE311458"].copy()
if obs_311["outcome"].nunique() > 1:
    log.info("Response stratification in GSE311458...")
    counts_311 = (obs_311.groupby(["patient_uid", "timepoint", "cell_type", "outcome"])
                  .size().reset_index(name="n"))
    totals_311  = counts_311.groupby(["patient_uid", "timepoint"])["n"].sum() \
                             .reset_index(name="total")
    counts_311  = counts_311.merge(totals_311, on=["patient_uid", "timepoint"])
    counts_311["fraction"] = counts_311["n"] / counts_311["total"]

    # NK cell fraction change by response
    nk_311 = counts_311[counts_311["cell_type"] == "NK_cell"].copy()
    nk_pivot = nk_311.pivot_table(index=["patient_uid", "outcome"],
                                    columns="timepoint", values="fraction").reset_index()
    nk_pivot = nk_pivot.dropna(subset=["pre", "post"])
    nk_pivot["delta"] = nk_pivot["post"] - nk_pivot["pre"]
    nk_pivot["fc"]    = (nk_pivot["post"] + 1e-6) / (nk_pivot["pre"] + 1e-6)
    log.info("NK changes by outcome:\n%s",
             nk_pivot[["outcome", "pre", "post", "delta", "fc"]].to_string(index=False))

    if len(nk_pivot) >= 3:
        fig, ax = plt.subplots(figsize=(7, 5))
        outcomes = nk_pivot["outcome"].unique()
        palette  = plt.cm.Set1(np.linspace(0, 0.8, len(outcomes)))
        for j, (_, row) in enumerate(nk_pivot.iterrows()):
            color = palette[list(outcomes).index(row["outcome"])]
            ax.plot([0, 1], [row["pre"], row["post"]], "o-", color=color,
                    lw=2, ms=7, label=row["outcome"])
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, title="Outcome")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=10)
        ax.set_ylabel("NK cell fraction", fontsize=10)
        ax.set_title("NK cell expansion by treatment response\n(GSE311458, n=4 patients)",
                     fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "response_stratified_nk.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

log.info("=== Pooled composition complete ===")
