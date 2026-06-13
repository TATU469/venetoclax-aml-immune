"""
GSEA — venetoclax+AZA response, per cell type.

Preranked GSEA using MSigDB Hallmark gene sets, ranked by
sign(log2FC) × -log10(p-value). Run on all cell types with ≥5
significant DGE genes.

Outputs:
  results/tables/gsea/gsea_{cell_type}.tsv
  results/figures/gsea/gsea_dotplot_{cell_type}.png
  results/tables/gsea_summary.tsv
"""

import sys, os, logging, warnings, glob
import numpy as np
import pandas as pd
import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "."
DGE_DIR  = os.path.join(PROJECT, "results/tables/dge")
GSEA_DIR = os.path.join(PROJECT, "results/tables/gsea")
FIG_DIR  = os.path.join(PROJECT, "results/figures/gsea")
for d in [GSEA_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

GENE_SETS   = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"]
MIN_SIZE    = 10
MAX_SIZE    = 500
N_PERMS     = 1000
FDR_THRESH  = 0.05

summary_records = []

dge_files = glob.glob(os.path.join(DGE_DIR, "dge_*_post_vs_pre.tsv"))
log.info("Found %d DGE files", len(dge_files))

for dge_file in sorted(dge_files):
    cell_type = os.path.basename(dge_file).replace("dge_", "").replace("_post_vs_pre.tsv", "")
    log.info("--- GSEA: %s ---", cell_type)

    res = pd.read_csv(dge_file, sep="\t")
    res = res[res["gene_symbol"].notna() & res["log2FoldChange"].notna() & res["pvalue"].notna()]
    res["rank_metric"] = (np.sign(res["log2FoldChange"]) *
                          -np.log10(res["pvalue"].clip(lower=1e-300)))
    res = res.sort_values("rank_metric", ascending=False)

    if len(res) < 50:
        log.warning("  Skipping %s — only %d ranked genes", cell_type, len(res))
        continue

    ranked = res[["gene_symbol", "rank_metric"]].drop_duplicates("gene_symbol")

    for gene_set in GENE_SETS:
        gs_label = gene_set.split("_")[0]  # "MSigDB" or "GO"
        try:
            pre_res = gp.prerank(
                rnk=ranked, gene_sets=gene_set,
                min_size=MIN_SIZE, max_size=MAX_SIZE,
                permutation_num=N_PERMS, seed=42, verbose=False,
                outdir=None,
            )
            gsea_df = pre_res.res2d
            gsea_df["cell_type"] = cell_type
            gsea_df["gene_set_library"] = gene_set

            sig = gsea_df[gsea_df["FDR q-val"] < FDR_THRESH]
            n_sig = len(sig)
            n_up  = (sig["NES"] > 0).sum()
            n_dn  = (sig["NES"] < 0).sum()
            log.info("  %s / %s: %d sig pathways (↑%d ↓%d)",
                     cell_type, gs_label, n_sig, n_up, n_dn)

            out_path = os.path.join(GSEA_DIR, f"gsea_{cell_type}_{gs_label}.tsv")
            gsea_df.sort_values("FDR q-val").to_csv(out_path, sep="\t", index=False)

            summary_records.append({
                "cell_type": cell_type, "gene_set": gene_set,
                "n_sig": n_sig, "n_up": n_up, "n_down": n_dn,
                "top_up": sig[sig["NES"] > 0].nsmallest(1, "FDR q-val")["Term"].values[0]
                          if n_up > 0 else "",
                "top_dn": sig[sig["NES"] < 0].nsmallest(1, "FDR q-val")["Term"].values[0]
                          if n_dn > 0 else "",
            })

            # Dotplot of top significant pathways
            if n_sig > 0:
                top_paths = pd.concat([
                    sig[sig["NES"] > 0].nsmallest(5, "FDR q-val"),
                    sig[sig["NES"] < 0].nsmallest(5, "FDR q-val"),
                ]).copy()
                top_paths["-log10_fdr"] = -np.log10(top_paths["FDR q-val"].clip(lower=1e-10))
                top_paths = top_paths.sort_values("NES")

                fig, ax = plt.subplots(figsize=(8, max(3, len(top_paths) * 0.45)))
                colors = ["#e74c3c" if n > 0 else "#2980b9" for n in top_paths["NES"]]
                bars = ax.barh(range(len(top_paths)), top_paths["NES"],
                               color=colors, alpha=0.85)
                ax.set_yticks(range(len(top_paths)))
                ax.set_yticklabels(
                    [t[:55] for t in top_paths["Term"]], fontsize=7
                )
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("NES", fontsize=10)
                ax.set_title(f"{cell_type} — {gs_label} GSEA\n(post vs pre venetoclax+AZA, FDR<0.05)",
                             fontsize=9)
                plt.tight_layout()
                fig.savefig(os.path.join(FIG_DIR, f"gsea_barplot_{cell_type}_{gs_label}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

        except Exception as e:
            log.error("  GSEA failed for %s / %s: %s", cell_type, gene_set, e)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(os.path.join(PROJECT, "results/tables"), "gsea_summary.tsv"),
                  sep="\t", index=False)
log.info("\n=== GSEA SUMMARY ===\n%s",
         summary_df[["cell_type", "gene_set", "n_sig", "n_up", "n_down",
                      "top_up", "top_dn"]].to_string(index=False))
log.info("=== GSEA complete ===")
