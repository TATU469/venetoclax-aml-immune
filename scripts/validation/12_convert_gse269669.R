#!/usr/bin/env Rscript
#
# Convert GSE269669 Seurat RDS files → per-sample MTX + metadata
# for downstream loading by 13_qc_gse269669.py (Python/Scanpy).
#
# Strategy: export each sample as 10x-style MTX triplet so the Python
# QC pipeline can load them the same way as GSE306339 / GSE311458.
#
# Outputs (one directory per sample):
#   data/raw/GSE269669/converted/<sample_id>/
#     matrix.mtx.gz   barcodes.tsv.gz   features.tsv.gz   metadata.tsv.gz
#
# Usage:
#   Rscript 12_convert_gse269669.R <project_dir>

# Writable user library — prepend so installs go here, system libs still visible
user_lib <- path.expand("~/R/library")
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
message("Library paths: ", paste(.libPaths(), collapse = "; "))

CRAN <- "https://cloud.r-project.org"

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing ", pkg, "...")
    install.packages(pkg, repos = CRAN, lib = user_lib, quiet = TRUE)
  }
}

# Matrix ships with R but may be absent in Spack builds — install from CRAN
install_if_missing("Matrix")
install_if_missing("Seurat")

suppressPackageStartupMessages({
  library(Matrix)
  library(methods)
  library(Seurat)
})
message("Seurat version: ", packageVersion("Seurat"))

# ── Paths ─────────────────────────────────────────────────────────────────────
args    <- commandArgs(trailingOnly = TRUE)
PROJECT <- if (length(args) >= 1) args[1] else "."
RAW_DIR <- file.path(PROJECT, "data/raw/GSE269669")
OUT_DIR <- file.path(RAW_DIR, "converted")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Helper: parse patient_id and timepoint from filename ─────────────────────
parse_sample_info <- function(fname) {
  base <- tools::file_path_sans_ext(tools::file_path_sans_ext(basename(fname)))
  # Remove GSM accession prefix (GSMxxxxxxx_...)
  core <- sub("^GSM[0-9]+_", "", base)
  # Detect timepoint keywords
  timepoint <- if (grepl("post|after|d[0-9]+[^0]|cycle|treat", core,
                          ignore.case = TRUE)) "post" else "pre"
  # Strip timepoint keywords to get patient id
  patient <- gsub("[-_](pre|post|baseline|after|before|treatment|treated|cycle[0-9]*|D[0-9]+|d[0-9]+)",
                  "", core, ignore.case = TRUE)
  patient <- gsub("[-_]+$", "", patient)
  list(patient = patient, timepoint = timepoint, core = core)
}

# ── Helper: export one Seurat object to MTX ───────────────────────────────────
export_sample <- function(seu, sample_id, out_base) {
  dir.create(out_base, recursive = TRUE, showWarnings = FALSE)

  # Prefer RNA assay; fall back to first available
  assay <- if ("RNA" %in% names(seu@assays)) "RNA" else names(seu@assays)[1]
  message("  Using assay: ", assay)

  # Raw counts — handle Seurat v4 and v5 slot differences
  counts <- tryCatch(
    GetAssayData(seu, assay = assay, layer  = "counts"),   # Seurat v5
    error = function(e)
    GetAssayData(seu, assay = assay, slot   = "counts"))   # Seurat v4

  if (ncol(counts) == 0 || nrow(counts) == 0) {
    message("  WARNING: empty counts matrix for ", sample_id, " — skipping")
    return(invisible(NULL))
  }

  # Write matrix.mtx.gz
  mtx_path <- file.path(out_base, "matrix.mtx")
  writeMM(counts, mtx_path)
  system2("gzip", c("-f", mtx_path))

  # Write barcodes.tsv.gz
  bc_path <- file.path(out_base, "barcodes.tsv")
  writeLines(colnames(counts), bc_path)
  system2("gzip", c("-f", bc_path))

  # Write features.tsv.gz (gene_id, gene_name, feature_type)
  feat_path <- file.path(out_base, "features.tsv")
  genes <- rownames(counts)
  writeLines(paste(genes, genes, "Gene Expression", sep = "\t"), feat_path)
  system2("gzip", c("-f", feat_path))

  # Write metadata.tsv.gz (all obs columns)
  meta <- seu@meta.data
  meta$barcode <- rownames(meta)
  gz_con <- gzcon(file(file.path(out_base, "metadata.tsv.gz"), "wb"))
  write.table(meta, gz_con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz_con)

  message("  Exported: ", ncol(counts), " cells × ", nrow(counts),
          " genes → ", out_base)
  invisible(out_base)
}

# ── Find RDS files ────────────────────────────────────────────────────────────
# Prefer the TME file (immune cells); also load AML file if present
preferred <- c("GSE269669_avm_tme.rds", "GSE269669_avm_aml.rds")
rds_files <- file.path(RAW_DIR, preferred)
rds_files <- rds_files[file.exists(rds_files)]

if (length(rds_files) == 0) {
  rds_files <- list.files(RAW_DIR, pattern = "\\.rds$|\\.RDS$",
                          full.names = TRUE, recursive = TRUE)
}

if (length(rds_files) == 0) {
  stop("No RDS files found in ", RAW_DIR,
       "\nRun 13_download_gse269669.pbs first.")
}
message("Found ", length(rds_files), " RDS file(s):")
message(paste(" ", rds_files, collapse = "\n"))

# ── Process each RDS file ─────────────────────────────────────────────────────
for (rds_path in rds_files) {
  message("\n=== Loading: ", basename(rds_path), " ===")
  info <- parse_sample_info(rds_path)

  obj <- tryCatch(readRDS(rds_path),
                  error = function(e) { message("ERROR loading: ", e$message); NULL })
  if (is.null(obj)) next

  # Case 1: single merged Seurat object (all samples in one RDS)
  if (inherits(obj, "Seurat") && ncol(obj) > 10000) {
    message("Large object (", ncol(obj), " cells) — treating as merged dataset")

    # Find patient/timepoint columns in metadata
    meta_cols <- tolower(colnames(obj@meta.data))
    patient_col <- colnames(obj@meta.data)[
      which(meta_cols %in% c("patient", "patient_id", "sample", "sample_id",
                             "donor", "case", "subject"))[1]]
    time_col <- colnames(obj@meta.data)[
      which(meta_cols %in% c("timepoint", "time_point", "time", "treatment",
                             "condition", "group", "stage", "visit"))[1]]

    message("  Patient column: ", ifelse(is.na(patient_col), "NOT FOUND", patient_col))
    message("  Timepoint column: ", ifelse(is.na(time_col), "NOT FOUND", time_col))
    message("  Metadata columns: ", paste(colnames(obj@meta.data), collapse = ", "))

    if (!is.na(patient_col) && !is.na(time_col)) {
      samples <- unique(paste(obj@meta.data[[patient_col]],
                               obj@meta.data[[time_col]], sep = "_"))
      message("  Splitting into ", length(samples), " samples: ", paste(samples, collapse = ", "))

      for (samp in samples) {
        pid   <- obj@meta.data[[patient_col]]
        tp    <- obj@meta.data[[time_col]]
        cells <- colnames(obj)[pid == strsplit(samp, "_")[[1]][1] &
                               tp  == strsplit(samp, "_")[[1]][2]]
        if (length(cells) < 10) next
        sub_obj   <- subset(obj, cells = cells)
        sample_id <- gsub("[^A-Za-z0-9_]", "_", samp)
        export_sample(sub_obj, sample_id, file.path(OUT_DIR, sample_id))
      }
    } else {
      # No clear sample columns — try to infer from orig.ident
      if ("orig.ident" %in% colnames(obj@meta.data)) {
        message("  Splitting by orig.ident")
        for (ident in unique(obj@meta.data$orig.ident)) {
          cells     <- colnames(obj)[obj@meta.data$orig.ident == ident]
          sample_id <- gsub("[^A-Za-z0-9_]", "_", ident)
          sub_obj   <- subset(obj, cells = cells)
          export_sample(sub_obj, sample_id, file.path(OUT_DIR, sample_id))
        }
      } else {
        # Export as single block — Python script will handle splitting
        export_sample(obj, "merged", file.path(OUT_DIR, "merged"))
      }
    }

  } else if (inherits(obj, "Seurat")) {
    # Case 2: one Seurat object per sample file
    sample_id <- paste(info$patient, info$timepoint, sep = "_")
    sample_id <- gsub("[^A-Za-z0-9_]", "_", sample_id)
    export_sample(obj, sample_id, file.path(OUT_DIR, sample_id))

  } else {
    message("  Object class: ", class(obj), " — not a Seurat object, skipping")
  }

  rm(obj); gc()
}

# ── Summary ───────────────────────────────────────────────────────────────────
converted <- list.dirs(OUT_DIR, recursive = FALSE, full.names = FALSE)
message("\n=== Conversion complete ===")
message("Exported ", length(converted), " sample directories:")
for (d in converted) {
  files <- list.files(file.path(OUT_DIR, d))
  message("  ", d, ": ", paste(files, collapse = ", "))
}
