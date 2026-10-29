# ==========================
# Seurat V6 SVG Detection (Moran's I) — 7 Real Datasets
# ==========================
library(Seurat)
library(Matrix)
library(anndata)
library(stats)

# --------------------------
# Path Settings (7 real datasets)
# --------------------------
INPUT_DIR  <- "D:/My_Project_ST/benchmarking of ST platforms/Datasets_Standardized"
OUTPUT_ROOT <- "D:/My_Project_ST/benchmarking of ST platforms/SVGs_Clustering_Method/SeuratV6"
dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)

# --------------------------
# 7 Datasets (actual filenames in Datasets_Standardized)
# --------------------------
DATASET_CONFIG <- list(
  "10X_Visium_Lung" = list(file = "10X_Visium_Lung.h5ad", n_cluster = 6, platform = "10X Visium"),
  "DLPFC_151507"    = list(file = "DLPFC_151507.h5ad",    n_cluster = 7, platform = "DLPFC"),
  "Slide-seq2"      = list(file = "Slide-seq2.h5ad",      n_cluster = 7, platform = "Slide-seq2"),
  "SNARE-seq2"      = list(file = "SNARE-seq2.h5ad",      n_cluster = 5, platform = "SNARE-seq2"),
  "ST"              = list(file = "ST.h5ad",              n_cluster = 4, platform = "ST"),
  "STARmap"         = list(file = "STARmap.h5ad",         n_cluster = 3, platform = "STARmap"),
  "Xenium_BC"       = list(file = "Xenium_BC.h5ad",       n_cluster = 8, platform = "Xenium")
)

# --------------------------
# Helper 1: Build KNN spatial weights matrix (row-normalized)
# --------------------------
build_spatial_knn_weights <- function(coords, k = 12) {
  n <- nrow(coords)
  d <- as.matrix(dist(coords))
  # For each cell, find k nearest neighbors (exclude self)
  diag(d) <- Inf
  W <- matrix(0, nrow = n, ncol = n)
  for (i in seq_len(n)) {
    nn_idx <- order(d[i, ])[1:k]
    W[i, nn_idx] <- 1
  }
  # Row-normalize
  row_sums <- rowSums(W)
  row_sums[row_sums == 0] <- 1
  W <- W / row_sums
  return(W)
}

# --------------------------
# Helper 2: Compute Moran's I for all genes (vectorized)
# --------------------------
compute_moran_i_all <- function(expr_mat, W) {
  n <- ncol(expr_mat)  # cells
  n_genes <- nrow(expr_mat)
  # Row-center each gene
  gene_means <- rowMeans(expr_mat)
  z <- expr_mat - gene_means
  # z * Wz for all genes at once: Wz = W %*% t(z), then rowSums(z * t(Wz))
  Wz <- W %*% t(z)          # n_cells x n_genes
  zWz <- colSums(t(z) * Wz) # n_genes
  S0 <- sum(W)
  z2_sum <- rowSums(z^2)    # n_genes
  moran_i <- (n / S0) * (zWz / z2_sum)
  # Expected value and variance under null
  E_I <- -1 / (n - 1)
  # Simplified variance (randomization assumption)
  S1 <- 0.5 * sum((W + t(W))^2)
  S2 <- sum((rowSums(W) + colSums(W))^2)
  k <- (rowSums(z^4) / n) / (z2_sum / n)^2  # kurtosis per gene
  var_I_num <- n * ((n^2 - 3*n + 3)*S1 - n*S2 + 3*S0^2) - k * ((n^2 - n)*S1 - 2*n*S2 + 6*S0^2)
  var_I_den <- (n-1) * (n-2) * (n-3) * S0^2
  var_I <- var_I_num / var_I_den - E_I^2
  var_I[var_I < 0] <- 1e-10
  z_score <- (moran_i - E_I) / sqrt(var_I)
  p_value <- 2 * pnorm(-abs(z_score))  # two-tailed
  return(list(I = moran_i, p = p_value, z = z_score))
}

# --------------------------
# Main: SeuratV6 SVG pipeline for one dataset
# --------------------------
run_seuratv6_svg <- function(dataset_name, cfg, output_dir) {
  file_path <- file.path(INPUT_DIR, cfg$file)
  cat("\n==== Processing:", dataset_name, "====\n")
  cat("File:", file_path, "\n")
  
  if (!file.exists(file_path)) {
    cat("  WARNING: File missing, skip\n")
    return(NULL)
  }
  
  # 1. Read h5ad
  adata <- anndata::read_h5ad(file_path)
  counts_mat <- t(as.matrix(adata$X))  # genes x cells
  rownames(counts_mat) <- rownames(adata$var)
  colnames(counts_mat) <- rownames(adata$obs)
  cat("  Dimensions:", nrow(counts_mat), "genes x", ncol(counts_mat), "cells\n")
  
  # 2. Extract spatial coordinates
  if ("spatial" %in% names(adata$obsm)) {
    coords <- as.matrix(adata$obsm$spatial)
  } else if ("spatial_coords" %in% names(adata$obsm)) {
    coords <- as.matrix(adata$obsm$spatial_coords)
  } else if (all(c("x","y") %in% colnames(adata$obs))) {
    coords <- as.matrix(adata$obs[, c("x","y")])
  } else {
    stop("No spatial coordinates found")
  }
  cat("  Spatial coords:", paste(dim(coords), collapse = " x "), "\n")
  
  # 3. Create Seurat object
  seurat_obj <- CreateSeuratObject(counts = counts_mat, project = dataset_name)
  
  # 4. Normalize (SCTransform with fallback)
  seurat_obj <- tryCatch({
    SCTransform(seurat_obj, verbose = FALSE)
  }, error = function(e) {
    message("  SCTransform failed -> using NormalizeData fallback")
    seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
    seurat_obj <- FindVariableFeatures(seurat_obj, verbose = FALSE)
    seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
    seurat_obj
  })
  
  # 5. Get normalized expression (use all genes, not just variable)
  #    Use data layer (log-normalized) for Moran's I
  expr_mat <- GetAssayData(seurat_obj, layer = "data")
  if (is(expr_mat, "dgCMatrix") || is(expr_mat, "sparseMatrix")) {
    expr_mat <- as.matrix(expr_mat)
  }
  # Filter genes with zero variance
  gene_vars <- apply(expr_mat, 1, var)
  valid_genes <- gene_vars > 1e-10
  expr_mat <- expr_mat[valid_genes, , drop = FALSE]
  cat("  Genes with non-zero variance:", nrow(expr_mat), "\n")
  
  # 6. Build spatial KNN weights (k=12, consistent with simulation SeuratV6)
  W <- build_spatial_knn_weights(coords, k = 12)
  
  # 7. Compute Moran's I for all genes
  cat("  Computing Moran's I for", nrow(expr_mat), "genes...\n")
  moran_res <- compute_moran_i_all(expr_mat, W)
  
  # 8. BH-FDR correction
  fdr_p <- p.adjust(moran_res$p, method = "BH")
  
  # 9. Build result table (ALL genes)
  svg_df <- data.frame(
    gene_name = rownames(expr_mat),
    Moran_I_score = moran_res$I,
    z_score = moran_res$z,
    raw_p_value = moran_res$p,
    FDR_BH_p_adjust = fdr_p,
    is_SVG_FDR0.05 = fdr_p < 0.05,
    stringsAsFactors = FALSE
  )
  svg_df <- svg_df[order(-svg_df$Moran_I_score), ]
  rownames(svg_df) <- NULL
  
  # 10. Save outputs
  sig_count <- sum(svg_df$is_SVG_FDR0.05)
  cat("  Significant SVGs (FDR<0.05):", sig_count, "\n")
  
  write.csv(svg_df, file.path(output_dir, paste0(dataset_name, "_SeuratV6_SVG_AllGenes.csv")), row.names = FALSE)
  write.csv(svg_df[svg_df$is_SVG_FDR0.05, ], file.path(output_dir, paste0(dataset_name, "_Significant_SVG_FDR0.05.csv")), row.names = FALSE)
  write.csv(head(svg_df, 50), file.path(output_dir, paste0(dataset_name, "_Top50_SVG.csv")), row.names = FALSE)
  
  summary_row <- data.frame(
    platform = cfg$platform,
    dataset = dataset_name,
    n_cluster = cfg$n_cluster,
    total_genes = nrow(svg_df),
    significant_SVG_FDR0.05 = sig_count
  )
  write.csv(summary_row, file.path(output_dir, paste0(dataset_name, "_summary.csv")), row.names = FALSE)
  
  # Cleanup
  rm(adata, counts_mat, seurat_obj, expr_mat, W, moran_res)
  gc()
  
  return(summary_row)
}

# --------------------------
# Batch Run (7 datasets, with try/except)
# --------------------------
all_summary <- list()
idx <- 0
for (ds_name in names(DATASET_CONFIG)) {
  cfg <- DATASET_CONFIG[[ds_name]]
  ds_out_dir <- file.path(OUTPUT_ROOT, ds_name)
  dir.create(ds_out_dir, recursive = TRUE, showWarnings = FALSE)
  idx <- idx + 1
  cat("\n==================================================\n")
  cat("  [", idx, "/7] Dataset:", ds_name, "| Platform:", cfg$platform, "\n")
  cat("==================================================\n")
  result <- tryCatch({
    run_seuratv6_svg(ds_name, cfg, ds_out_dir)
  }, error = function(e) {
    cat("  FAILED:", ds_name, "-", e$message, "\n")
    data.frame(platform = cfg$platform, dataset = ds_name, n_cluster = cfg$n_cluster,
               total_genes = "NA", significant_SVG_FDR0.05 = "Failed")
  })
  if (!is.null(result)) all_summary[[ds_name]] <- result
}

# --------------------------
# Final Global Summary
# --------------------------
if (length(all_summary) > 0) {
  global_df <- do.call(rbind, all_summary)
  rownames(global_df) <- NULL
  write.csv(global_df, file.path(OUTPUT_ROOT, "SeuratV6_7Datasets_Global_SVG_Summary.csv"), row.names = FALSE)
  cat("\n====== ALL 7 DATASETS FINISHED ======\n")
  cat("Global summary saved to:", file.path(OUTPUT_ROOT, "SeuratV6_7Datasets_Global_SVG_Summary.csv"), "\n")
  print(global_df)
} else {
  cat("\nNo datasets were successfully processed.\n")
}