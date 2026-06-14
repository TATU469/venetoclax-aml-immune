"""
Response-stratified immune composition — GSE269669 (n=11 TP53-mut, ven+AZA+magrolimab).

Compares NK/CD8 T cell dynamics between responders (CR/MLFS/CRi, n=8)
and non-responders (NR, n=3).

Clinical labels from BestResponse column in Seurat AML metadata:
  CR/CRi/MLFS → Responder (n=8): PT33, PT35, PT36, PT37, PT38, PT39, PT41, PT42
  NR          → Non-responder (n=3): PT34, PT40, PT43

Outputs:
  results/tables/validation/gse269669_response_stratification.tsv
  results/figures/validation/gse269669_response_nk_cd8.png
  results/figures/validation/gse269669_response_fc.png
  results/figures/validation/gse269669_baseline_by_response.png
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

# Clinical response labels from BestResponse column in Seurat AML metadata
RESPONSE_MAP = {
    "PT33": "Responder",   # CR
    "PT34": "Non-responder",  # NR
    "PT35": "Responder",   # CR
    "PT36": "Responder",   # CR
    "PT37": "Responder",   # MLFS
    "PT38": "Responder",   # CR
    "PT39": "Responder",   # CR
    "PT40": "Non-responder",  # NR
    "PT41": "Responder",   # CRi
    "PT42": "Responder",   # CR
    "PT43": "Non-responder",  # NR
}

BEST_RESPONSE = {
    "PT33": "CR",  "PT34": "NR",   "PT35": "CR",  "PT36": "CR",
    "PT37": "MLFS","PT38": "CR",   "PT39": "CR",  "PT40": "NR",
    "PT41": "CRi", "PT42": "CR",   "PT43": "NR",
}

AZIMUTH_MAP = {
    "NK": "NK_cell", "NK CD56+": "NK_cell", "ILC": "NK_cell",
    "CD4 Memory": "CD4_T_cell", "CD4 Naive": "CD4_T_cell",
    "CD4 Effector": "CD4_T_cell", "Treg": "CD4_T_cell",
    "CD8 Effector_1": "CD8_T_cell", "CD8 Effector_2": "CD8_T_cell",
    "CD8 Memory": "CD8_T_cell", "CD8 Naive": "CD8_T_cell", "MAIT": "CD8_T_cell",
    "Memory B": "B_cell", "Naive B": "B_cell", "Transitional B": "B_cell", "Plasma": "B_cell",
    "CD14 Mono": "Monocyte", "CD16 Mono": "Monocyte", "Macrophage": "Monocyte",
    "cDC1": "DC", "cDC2": "DC", "pDC": "DC",
    "Late Eryth": "Erythroid", "Early Eryth": "Erythroid",
    "GMP": "Progenitor", "HSC": "Progenitor", "LMPP": "Progenitor",
    "Prog Mk": "Progenitor", "Platelet": "Progenitor",
}

COLORS = {"Responder": "#2980b9", "Non-responder": "#e74c3c"}

# ── Load ──────────────────────────────────────────────────────────────────────
log.info("Loading GSE269669...")
adata = sc.read_h5ad(os.path.join(PROC_DIR, "gse269669_annotated.h5ad"))

if "seurat_celltype" in adata.obs.columns:
    adata.obs["cell_type_broad"] = (
        adata.obs["seurat_celltype"].astype(str).map(AZIMUTH_MAP).fillna("Other")
    )
else:
    adata.obs["cell_type_broad"] = adata.obs["cell_type"]

adata.obs["response"]      = adata.obs["patient_id"].astype(str).map(RESPONSE_MAP)
adata.obs["best_response"] = adata.obs["patient_id"].astype(str).map(BEST_RESPONSE)

log.info("Response groups:\n%s",
         adata.obs.drop_duplicates("patient_id")[["patient_id", "best_response", "response"]]
         .sort_values("patient_id").to_string(index=False))

# Filter to pre/post only
obs = adata.obs[adata.obs["timepoint"].isin(["pre", "post"])].copy()
for col in ["patient_id", "timepoint", "cell_type_broad", "response", "best_response"]:
    obs[col] = obs[col].astype(str)

# ── Cell type fractions ───────────────────────────────────────────────────────
counts = (obs.groupby(["patient_id", "timepoint", "cell_type_broad", "response", "best_response"])
          .size().reset_index(name="n"))
totals = counts.groupby(["patient_id", "timepoint"])["n"].sum().reset_index(name="total")
counts = counts.merge(totals, on=["patient_id", "timepoint"])
counts["fraction"] = counts["n"] / counts["total"]

cell_types = [c for c in sorted(counts["cell_type_broad"].unique()) if c != "Other"]
results = []

for ct in cell_types:
    ct_data = counts[counts["cell_type_broad"] == ct]
    pivot = ct_data.pivot_table(
        index=["patient_id", "response", "best_response"],
        columns="timepoint", values="fraction"
    ).reset_index().dropna(subset=["pre", "post"])

    if len(pivot) < 3:
        continue

    pivot["fc"]    = (pivot["post"] + 1e-6) / (pivot["pre"] + 1e-6)
    pivot["delta"] = pivot["post"] - pivot["pre"]

    R  = pivot[pivot["response"] == "Responder"]
    NR = pivot[pivot["response"] == "Non-responder"]

    log.info("\n%s — Responders (n=%d) vs Non-responders (n=%d):", ct, len(R), len(NR))
    log.info("  R:  pre=%.3f post=%.3f FC=%.2f",
             R["pre"].mean(), R["post"].mean(), R["fc"].median())
    log.info("  NR: pre=%.3f post=%.3f FC=%.2f",
             NR["pre"].mean(), NR["post"].mean(), NR["fc"].median())

    p_fc = np.nan
    if len(R) >= 2 and len(NR) >= 2:
        _, p_fc = stats.mannwhitneyu(R["fc"], NR["fc"], alternative="two-sided")
        log.info("  FC Mann-Whitney p=%.4f", p_fc)

    _, p_pre = stats.mannwhitneyu(R["pre"], NR["pre"], alternative="two-sided") \
        if len(R) >= 2 and len(NR) >= 2 else (None, np.nan)
    log.info("  Baseline (pre) Mann-Whitney p=%.4f", p_pre)

    results.append({
        "cell_type": ct,
        "n_responders": len(R), "n_non_responders": len(NR),
        "R_median_fc": R["fc"].median(), "NR_median_fc": NR["fc"].median(),
        "R_mean_pre": R["pre"].mean(), "NR_mean_pre": NR["pre"].mean(),
        "R_mean_post": R["post"].mean(), "NR_mean_post": NR["post"].mean(),
        "p_fc_mannwhitney": p_fc,
        "p_pre_mannwhitney": p_pre,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(TAB_DIR, "gse269669_response_stratification.tsv"),
                  sep="\t", index=False)

log.info("\n=== Response stratification summary ===")
log.info("\n%s", results_df[["cell_type", "R_median_fc", "NR_median_fc",
                              "p_fc_mannwhitney", "p_pre_mannwhitney"]].to_string(index=False))

# ── Figure 1: NK + CD8 spaghetti by response ─────────────────────────────────
key_types = ["NK_cell", "CD8_T_cell", "CD4_T_cell", "B_cell"]
key_types = [c for c in key_types if c in counts["cell_type_broad"].unique()]

fig, axes = plt.subplots(1, len(key_types), figsize=(4.5 * len(key_types), 5))
if len(key_types) == 1:
    axes = [axes]

for ax, ct in zip(axes, key_types):
    ct_data = counts[counts["cell_type_broad"] == ct]
    pivot = ct_data.pivot_table(
        index=["patient_id", "response", "best_response"],
        columns="timepoint", values="fraction"
    ).reset_index().dropna(subset=["pre", "post"])

    for _, row in pivot.iterrows():
        color = COLORS.get(row["response"], "grey")
        ax.plot([0, 1], [row["pre"], row["post"]], "o-",
                color=color, alpha=0.7, lw=2, ms=6,
                label=row["response"])
        ax.text(1.05, row["post"], row["best_response"], fontsize=6,
                color=color, va="center")

    # Group medians
    for resp, color in COLORS.items():
        grp = pivot[pivot["response"] == resp]
        if len(grp) > 0:
            ax.plot([0, 1], [grp["pre"].median(), grp["post"].median()],
                    "o-", color=color, lw=3.5, ms=10, zorder=10,
                    markeredgecolor="white", markeredgewidth=1.5)

    row_r = results_df[results_df["cell_type"] == ct]
    p_str = ""
    if not row_r.empty:
        p = row_r["p_fc_mannwhitney"].values[0]
        p_str = f"\np(FC)={p:.3f}" if not np.isnan(p) else ""

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"], fontsize=10)
    ax.set_title(f"{ct.replace('_', ' ')}{p_str}", fontsize=10)
    ax.set_ylabel("Cell fraction", fontsize=9)
    ax.set_xlim(-0.3, 1.6)

legend_els = [mpatches.Patch(color=COLORS["Responder"], label="Responder (CR/MLFS/CRi, n=8)"),
              mpatches.Patch(color=COLORS["Non-responder"], label="Non-responder (NR, n=3)")]
fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=9, frameon=False)
plt.suptitle("GSE269669: Immune dynamics by treatment response\n"
             "ven+AZA+magrolimab, 11 TP53-mut AML patients", fontsize=11)
plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(os.path.join(FIG_DIR, "gse269669_response_nk_cd8.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Fold change comparison ─────────────────────────────────────────
if not results_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    w = 0.35
    ax.bar(x - w/2, results_df["R_median_fc"],  w, color=COLORS["Responder"],
           alpha=0.8, label="Responder (n=8)")
    ax.bar(x + w/2, results_df["NR_median_fc"], w, color=COLORS["Non-responder"],
           alpha=0.8, label="Non-responder (n=3)")
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in results_df["cell_type"]], fontsize=9)
    ax.set_ylabel("Median fold change (post/pre)", fontsize=10)
    ax.set_title("Immune composition fold change by response\n"
                 "GSE269669 — ven+AZA+magrolimab", fontsize=10)
    # Add p-values
    for i, row in results_df.iterrows():
        p = row["p_fc_mannwhitney"]
        if not np.isnan(p):
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else (f"p={p:.2f}" if p < 0.2 else ""))
            if sig:
                y_max = max(row["R_median_fc"], row["NR_median_fc"]) + 0.1
                ax.text(i, y_max, sig, ha="center", fontsize=10, color="black")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gse269669_response_fc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Figure 3: Baseline (pre) fractions by response ────────────────────────────
if not results_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, results_df["R_mean_pre"],  w, color=COLORS["Responder"],
           alpha=0.8, label="Responder")
    ax.bar(x + w/2, results_df["NR_mean_pre"], w, color=COLORS["Non-responder"],
           alpha=0.8, label="Non-responder")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in results_df["cell_type"]], fontsize=9)
    ax.set_ylabel("Mean pre-treatment fraction", fontsize=10)
    ax.set_title("Pre-treatment immune composition by response\n"
                 "GSE269669 — ven+AZA+magrolimab", fontsize=10)
    for i, row in results_df.iterrows():
        p = row["p_pre_mannwhitney"]
        if not np.isnan(p):
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else (f"p={p:.2f}" if p < 0.2 else ""))
            if sig:
                y_max = max(row["R_mean_pre"], row["NR_mean_pre"]) + 0.02
                ax.text(i, y_max, sig, ha="center", fontsize=10, color="black")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gse269669_baseline_by_response.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

log.info("=== Response stratification complete ===")
