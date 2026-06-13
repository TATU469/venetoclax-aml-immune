# Venetoclax AML Immune Response — Single-Cell Reanalysis

**Target journal:** Leukemia (IF ~12)

## Overview

Reanalysis of two public scRNA-seq datasets to characterise how venetoclax + azacitidine reshapes the immune microenvironment and malignant progenitor transcriptional state in TP53-mutated AML, contextualised against the established AML cell hierarchy.

## Datasets

| GEO | Description | Patients | Cells | Format |
|-----|-------------|----------|-------|--------|
| GSE306339 | TP53-mutated AML, pre/post venetoclax+AZA | 3 | ~6 samples | MTX/TSV |
| GSE116256 | van Galen 2019 AML hierarchy reference | 16 + 5 HBM | ~36k | TXT |

## Analysis Plan

1. **QC + preprocessing** — per-sample MAD-based filtering, doublet removal (Scrublet)
2. **Integration** — Harmony batch correction by patient_id
3. **Cell type annotation** — curated AML marker panels (10 populations)
4. **Composition analysis** — paired Wilcoxon pre/post venetoclax; bootstrap CIs
5. **DGE** — PyDESeq2 pseudobulk, malignant progenitor populations
6. **GSEA** — MSigDB Hallmark, ranked by sign(LFC) × −log10(p)
7. **Reference projection** — map venetoclax-treated cells onto van Galen 2019 hierarchy

## Project Structure

```
data/
  raw/          GSE116256_RAW/, GSE306339_RAW/
  processed/    h5ad per dataset, integrated h5ad
scripts/
  qc/           01_qc_gse306339.py, 02_qc_gse116256.py
  annotation/   03_annotate.py
  composition/  04_composition.py
  dge/          05_dge.py
  gsea/         06_gsea.py
  figures/      07_figures.py
  pbs/          PBS job wrappers
results/
  figures/
  tables/
docs/           manuscript
```

## Environment

Conda env: `/srv/scratch/z5530616/projects/pediatric_aml_relapse_multiome/envs/sc_multiome`

## Code availability

https://github.com/TATU469/venetoclax-aml-immune
