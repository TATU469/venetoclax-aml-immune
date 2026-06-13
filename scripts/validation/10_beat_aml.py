"""
BEAT AML bulk RNA-seq validation — NK/cytotoxic signature vs venetoclax sensitivity

Tyner et al. 2018, Nature (~430 AML patients, ex vivo drug AUC + bulk RNA-seq)

Tests whether NK/cytotoxic gene expression predicts ex vivo venetoclax sensitivity
(area under the dose-response curve, lower AUC = more sensitive).

Outputs:
  results/figures/validation/beat_aml_nk_venetoclax.png
  results/figures/validation/beat_aml_km_nk_os.png
  results/tables/validation/beat_aml_correlation_results.tsv
"""

import sys, os, logging, warnings, gzip
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

PROJECT  = sys.argv[1] if len(sys.argv) > 1 else "."
RAW_DIR  = os.path.join(PROJECT, "data/raw/BEAT_AML")
FIG_DIR  = os.path.join(PROJECT, "results/figures/validation")
TAB_DIR  = os.path.join(PROJECT, "results/tables/validation")
for d in [FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

# ── NK / cytotoxic signature genes (from our scRNA-seq analysis) ──────────────
NK_CYTOTOXIC_GENES = ["NKG7", "GNLY", "GZMB", "PRF1", "GZMA", "GZMH",
                       "IFNG", "NCR1", "KLRD1", "KLRK1", "XCL1", "XCL2"]
NK_INHIBITORY_GENES = ["TIGIT", "LAG3", "PDCD1", "HAVCR2", "CTLA4"]

# ── Load BEAT AML expression matrix ──────────────────────────────────────────
def load_beat_expr(raw_dir):
    """Try multiple known formats of BEAT AML expression data."""
    # Option 1: processed matrix text file
    for fname in ["GSE114922_TPM_table.txt.gz", "GSE114922_CPM_table.txt.gz",
                  "GSE114922_Count_table.txt.gz", "GSE114922_matrix.txt.gz",
                  "beat_aml_expr.txt.gz", "beat_aml_expr.csv"]:
        fp = os.path.join(raw_dir, fname)
        if os.path.exists(fp):
            log.info("Loading expression from %s", fname)
            sep = "\t" if fname.endswith(".txt") or fname.endswith(".txt.gz") else ","
            return pd.read_csv(fp, sep=sep, index_col=0, compression="gzip"
                                if fname.endswith(".gz") else None)
    # Option 2: RAW tar extracted files
    import glob as gb
    txt_files = gb.glob(os.path.join(raw_dir, "*.txt.gz"))
    if txt_files:
        log.info("Loading from extracted txt.gz: %s", os.path.basename(txt_files[0]))
        return pd.read_csv(txt_files[0], sep="\t", index_col=0, compression="gzip")
    return None

# ── Load drug sensitivity ──────────────────────────────────────────────────────
def load_drug_sensitivity(raw_dir):
    """Load BEAT AML drug sensitivity (venetoclax AUC)."""
    for fname in ["beataml_drug_sensitivity.txt", "beataml_probit_curve_fits_v4_dbgap.txt",
                  "drug_sensitivity.tsv", "drug_sensitivity.txt"]:
        fp = os.path.join(raw_dir, fname)
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            log.info("Loading drug sensitivity from %s", fname)
            return pd.read_csv(fp, sep="\t")
    # Try GEO series matrix for clinical/drug info
    return None

expr = load_beat_expr(RAW_DIR)
drug = load_drug_sensitivity(RAW_DIR)

if expr is None:
    log.error("No BEAT AML expression matrix found in %s", RAW_DIR)
    log.error("Please download manually:")
    log.error("  wget -O %s/GSE114922_matrix.txt.gz "
              "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114922/suppl/"
              "GSE114922_matrix.txt.gz", RAW_DIR)
    sys.exit(1)

log.info("Expression matrix: %d genes × %d samples", *expr.shape)

# ── Compute NK signature score ────────────────────────────────────────────────
def signature_score(expr_df, genes):
    """Mean log2(TPM+1) across available genes; expr in log2 or TPM."""
    present = [g for g in genes if g in expr_df.index]
    log.info("  Signature genes found: %d/%d: %s", len(present), len(genes), present)
    if len(present) < 2:
        return None
    # Normalise: assume expr values are TPM or FPKM — log2 transform
    sub = expr_df.loc[present]
    if sub.max().max() > 50:  # likely not already log-transformed
        sub = np.log2(sub + 1)
    return sub.mean(axis=0)

log.info("Computing NK cytotoxic score...")
nk_score = signature_score(expr, NK_CYTOTOXIC_GENES)
if nk_score is None:
    log.error("Insufficient NK genes in expression matrix.")
    sys.exit(1)

nk_score = nk_score.rename("NK_cytotoxic_score")
score_df = nk_score.reset_index()
score_df.columns = ["sample_id", "NK_cytotoxic_score"]

inhibitory_score = signature_score(expr, NK_INHIBITORY_GENES)
if inhibitory_score is not None:
    score_df["NK_inhibitory_score"] = inhibitory_score.values

# ── Merge with drug sensitivity ────────────────────────────────────────────────
results = []

if drug is not None:
    log.info("Drug sensitivity columns: %s", list(drug.columns[:10]))

    # Find venetoclax column
    ven_col = next((c for c in drug.columns if "venetoclax" in c.lower()
                    or "abt-199" in c.lower() or "abt199" in c.lower()), None)
    sample_col = next((c for c in drug.columns if "sample" in c.lower()
                       or "lab_id" in c.lower() or "id" in c.lower()), drug.columns[0])

    if ven_col:
        log.info("Venetoclax column: %s", ven_col)
        ven_df = drug[[sample_col, ven_col]].dropna().copy()
        ven_df.columns = ["sample_id", "venetoclax_auc"]

        merged = score_df.merge(ven_df, on="sample_id", how="inner")
        log.info("Merged: %d samples with both NK score and venetoclax AUC", len(merged))

        if len(merged) >= 10:
            r, p = stats.spearmanr(merged["NK_cytotoxic_score"],
                                    merged["venetoclax_auc"])
            log.info("NK score vs venetoclax AUC: r=%.3f p=%.4f (n=%d)", r, p, len(merged))
            results.append({"analysis": "NK_vs_venetoclax_AUC",
                             "n": len(merged), "spearman_r": r, "p": p})

            # Scatter plot
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(merged["NK_cytotoxic_score"], merged["venetoclax_auc"],
                       c="#2980b9", s=20, alpha=0.6, linewidths=0)
            # Add regression line
            m, b = np.polyfit(merged["NK_cytotoxic_score"], merged["venetoclax_auc"], 1)
            x_line = np.linspace(merged["NK_cytotoxic_score"].min(),
                                  merged["NK_cytotoxic_score"].max(), 100)
            ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5, alpha=0.8)
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.set_title(f"NK cytotoxic score vs venetoclax AUC\n"
                         f"BEAT AML (n={len(merged)}) — Spearman r={r:.3f}, {p_str}",
                         fontsize=10)
            ax.set_xlabel("NK cytotoxic score (bulk RNA-seq)", fontsize=10)
            ax.set_ylabel("Venetoclax ex vivo AUC\n(lower = more sensitive)", fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "beat_aml_nk_venetoclax.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # High vs low NK — venetoclax AUC comparison
            median_nk = merged["NK_cytotoxic_score"].median()
            merged["NK_group"] = np.where(merged["NK_cytotoxic_score"] >= median_nk,
                                           "NK_high", "NK_low")
            hi = merged[merged["NK_group"] == "NK_high"]["venetoclax_auc"]
            lo = merged[merged["NK_group"] == "NK_low"]["venetoclax_auc"]
            _, p_mw = stats.mannwhitneyu(hi, lo, alternative="two-sided")
            log.info("NK_high vs NK_low venetoclax AUC: p=%.4f", p_mw)
            results.append({"analysis": "NK_high_vs_low_venetoclax_AUC",
                             "n": len(merged), "p_mannwhitney": p_mw,
                             "mean_nk_high": hi.mean(), "mean_nk_low": lo.mean()})

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.boxplot([lo, hi], labels=["NK low", "NK high"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#e0e0e0"),
                        medianprops=dict(color="black", linewidth=2))
            p_str2 = f"p={p_mw:.3f}" if p_mw >= 0.001 else "p<0.001"
            ax.set_ylabel("Venetoclax ex vivo AUC", fontsize=10)
            ax.set_title(f"Venetoclax sensitivity by NK score\n({p_str2})", fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "beat_aml_nk_group_boxplot.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    else:
        log.warning("No venetoclax column found in drug sensitivity file.")
        log.info("Available drugs (first 20): %s",
                 [c for c in drug.columns if c != sample_col][:20])
else:
    log.warning("Drug sensitivity file not found — running expression-only analysis.")
    # Fallback: test NK score distribution and check gene availability
    log.info("NK score distribution: mean=%.3f median=%.3f std=%.3f",
             score_df["NK_cytotoxic_score"].mean(),
             score_df["NK_cytotoxic_score"].median(),
             score_df["NK_cytotoxic_score"].std())

# ── Save results ───────────────────────────────────────────────────────────────
if results:
    pd.DataFrame(results).to_csv(
        os.path.join(TAB_DIR, "beat_aml_correlation_results.tsv"),
        sep="\t", index=False)

score_df.to_csv(os.path.join(TAB_DIR, "beat_aml_nk_scores.tsv"), sep="\t", index=False)
log.info("=== BEAT AML validation complete ===")
