library(iSpatial)
library(anndata)
library(dplyr)
library(Matrix)
library(cluster)

# --------------------------
# SVG permutation function
# --------------------------
calc_cluster_SVG <- function(adata, cluster_key = "iSpatial_leiden", perm_times = 1000L, seed = 0L) {
  set.seed(seed)
  cell_labels <- as.integer(adata$obs[[cluster_key]])
  n_cells <- adata$n_obs
  n_genes <- adata$n_vars
  unique_clusters <- sort(unique(cell_labels))
  n_cluster <- length(unique_clusters)
  expr_mat_sp <- adata$X
  
  if (inherits(expr_mat_sp, "sparseMatrix")) {
    global_gene_mean <- as.vector(Matrix::colMeans(expr_mat_sp))
  } else {
    global_gene_mean <- colMeans(expr_mat_sp)
  }
  
  cluster_mean_matrix <- matrix(0, nrow = n_cluster, ncol = n_genes)
  for (idx in seq_along(unique_clusters)) {
    c <- unique_clusters[idx]
    cell_mask <- cell_labels == c
    if (inherits(expr_mat_sp, "sparseMatrix")) {
      cluster_mean_matrix[idx, ] <- as.vector(Matrix::colMeans(expr_mat_sp[cell_mask, , drop = FALSE]))
    } else {
      cluster_mean_matrix[idx, ] <- colMeans(expr_mat_sp[cell_mask, , drop = FALSE])
    }
  }
  obs_score <- rowSums(t((t(cluster_mean_matrix) - global_gene_mean)^2)) / n_cluster
  
  perm_score_set <- matrix(0, nrow = perm_times, ncol = n_genes)
  for (p in 1:perm_times) {
    shuffle_idx <- sample.int(n_cells, size = n_cells, replace = FALSE)
    perm_cluster_mean <- matrix(0, nrow = n_cluster, ncol = n_genes)
    for (idx in seq_along(unique_clusters)) {
      c <- unique_clusters[idx]
      cell_mask <- cell_labels == c
      perm_sub <- expr_mat_sp[shuffle_idx, , drop = FALSE][cell_mask, , drop = FALSE]
      if (inherits(perm_sub, "sparseMatrix")) {
        perm_cluster_mean[idx, ] <- as.vector(Matrix::colMeans(perm_sub))
      } else {
        perm_cluster_mean[idx, ] <- colMeans(perm_sub)
      }
    }
    perm_score_set[p, ] <- rowSums(t((t(perm_cluster_mean) - global_gene_mean)^2)) / n_cluster
  }
  
  raw_p <- (colSums(perm_score_set >= obs_score) + 1) / (perm_times + 1)
  fdr_res <- p.adjust(raw_p, method = "BH")
  sig_flag <- fdr_res < 0.05
  
  svg_result_df <- data.frame(
    gene_name = adata$var_names,
    iSpatial_SVG_score = obs_score,
    raw_p_value = raw_p,
    fdr_BH_p_adjust = fdr_res,
    is_SVG_FDR0.05 = sig_flag
  ) %>% arrange(desc(iSpatial_SVG_score))
  gc()
  return(svg_result_df)
}

# --------------------------
# iSpatial clustering runner
# --------------------------
run_iSpatial_clustering <- function(adata_raw, n_cluster) {
  adata <- adata_raw$copy()
  keep_genes <- Matrix::rowSums(t(adata$X) > 0) >= 5
  adata <- adata[, keep_genes]
  
  size_factors <- Matrix::rowSums(adata$X)
  adata$X <- adata$X / size_factors * 10000
  adata$X <- log1p(adata$X)
  
  gene_mean <- Matrix::colMeans(adata$X)
  gene_sd <- apply(adata$X, 2, sd)
  gene_sd[gene_sd == 0] <- 1
  adata$X <- t((t(adata$X) - gene_mean)/gene_sd)
  
  expr_mat <- as.matrix(adata$X)
  res_list <- rev(seq(from = 0.2, to = 2.5, by = 0.02))
  final_cluster <- NULL
  embed_out <- NULL
  final_res <- NULL
  for (res in res_list) {
    isp_out <- iSpatial(expr_mat, resolution = res, random.seed = 0)
    pred_clusters <- isp_out$cluster
    if (length(unique(pred_clusters)) == n_cluster) {
      final_cluster <- pred_clusters
      embed_out <- isp_out$embedding
      final_res <- res
      break
    }
  }
  if (is.null(final_cluster)) {
    stop("Cannot find resolution to match target cluster count!")
  }
  
  # Assign back to original adata object safely
  adata_raw$obs[["iSpatial_leiden"]] <- as.character(final_cluster)
  adata_raw$obsm[["iSpatial_embedding"]] <- embed_out
  rm(adata)
  gc()
  return(adata_raw)
}

# --------------------------
# Config
# --------------------------
DATA_INPUT_DIR <- r"(D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized)"
OUTPUT_ROOT_DIR <- r"(D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\iSpatial\R_package)"

DATA_FILE_LIST <- c(
  "Xenium_BC_NCBI784_standardized.h5ad",
  "Xenium_Lung_NCBI856_standardized.h5ad",
  "10X_BC_TENX13_standardized.h5ad",
  "10X_Lung_TENX62_standardized.h5ad",
  "ST_BC_SPA130_standardized.h5ad",
  "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
  "STARmap_BOOST_HMI_standardized.h5ad"
)

CLUSTER_NUM_DICT <- list(
  "Xenium_BC_NCBI784" = 8,
  "Xenium_Lung_NCBI856" = 6,
  "10X_BC_TENX13" = 5,
  "10X_Lung_TENX62" = 6,
  "ST_BC_SPA130" = 4,
  "Slide-seq2_Kedney_HBM965.PSNC.855" = 7,
  "STARmap_BOOST_HMI" = 3
)

PLATFORM_LABEL <- list(
  "Xenium_BC_NCBI784_standardized.h5ad" = "Xenium",
  "Xenium_Lung_NCBI856_standardized.h5ad" = "Xenium",
  "10X_BC_TENX13_standardized.h5ad" = "10X Visium",
  "10X_Lung_TENX62_standardized.h5ad" = "10X Visium",
  "ST_BC_SPA130_standardized.h5ad" = "ST",
  "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad" = "Slide-seq",
  "STARmap_BOOST_HMI_standardized.h5ad" = "STARmap"
)

# --------------------------
# Main loop
# --------------------------
dir.create(OUTPUT_ROOT_DIR, recursive = TRUE, showWarnings = FALSE)
global_summary_df <- tibble()

cat(strrep("=",80),"\n")
cat("iSpatial pipeline | Separate folder per dataset\n")
cat(strrep("=",80),"\n")

for (file_name in DATA_FILE_LIST) {
  file_full_path <- file.path(DATA_INPUT_DIR, file_name)
  dataset_name <- sub("_standardized.h5ad", "", file_name)
  dataset_save_dir <- file.path(OUTPUT_ROOT_DIR, dataset_name)
  dir.create(dataset_save_dir, recursive = TRUE, showWarnings = FALSE)
  
  cat(sprintf("\n==== Processing: %s ====\n", dataset_name))
  if (!file.exists(file_full_path)) {
    cat(sprintf("  WARNING: File missing -> %s, skipped\n", file_full_path))
    next
  }
  
  adata <- read_h5ad(file_full_path)
  adata$var_names <- make.unique(adata$var_names)
  cat(sprintf("  adata loaded: n_obs=%d, n_vars=%d\n", adata$n_obs, adata$n_vars))
  
  if (!"label" %in% colnames(adata$obs)){
    stop(sprintf("Dataset %s lacks ground truth 'label' column", dataset_name))
  }
  
  target_cluster <- CLUSTER_NUM_DICT[[dataset_name]]
  cat(sprintf("  Target clusters: %d\n", target_cluster))
  
  clustered_adata <- run_iSpatial_clustering(adata, n_cluster = target_cluster)
  cat("  Clustering finished\n")
  
  svg_full_df <- calc_cluster_SVG(clustered_adata, cluster_key = "iSpatial_leiden", perm_times = 1000L, seed = 0L)
  cat(sprintf("  SVG table rows: %d\n", nrow(svg_full_df)))
  
  # Metrics: convert labels to numeric for NMI
  pred <- as.integer(factor(clustered_adata$obs$iSpatial_leiden))
  true_lab <- as.integer(factor(clustered_adata$obs$label))
  ari_score <- adjustedRandIndex(true_lab, pred)
  nmi_score <- NMI(true_lab, pred)
  sig_svg_count <- sum(svg_full_df$is_SVG_FDR0.05)
  
  # File paths
  h5ad_out <- file.path(dataset_save_dir, sprintf("iSpatial_%s_clustered.h5ad", dataset_name))
  csv_full <- file.path(dataset_save_dir, sprintf("%s_iSpatial_SVG_full_list.csv", dataset_name))
  csv_sig <- file.path(dataset_save_dir, sprintf("%s_Significant_SVG_FDR0.05.csv", dataset_name))
  csv_top50 <- file.path(dataset_save_dir, sprintf("%s_Top50_SVG.csv", dataset_name))
  csv_single_sum <- file.path(dataset_save_dir, sprintf("%s_summary.csv", dataset_name))
  
  # Write files
  write_h5ad(clustered_adata, h5ad_out)
  write.csv(svg_full_df, csv_full, row.names = FALSE)
  svg_sig <- svg_full_df %>% filter(is_SVG_FDR0.05)
  write.csv(svg_sig, csv_sig, row.names = FALSE)
  write.csv(head(svg_full_df, 50), csv_top50, row.names = FALSE)
  
  single_record <- tibble(
    platform = PLATFORM_LABEL[[file_name]],
    dataset = dataset_name,
    cluster_number = target_cluster,
    ARI = round(ari_score, 4),
    NMI = round(nmi_score, 4),
    total_gene_num = adata$n_vars,
    significant_SVG_FDR0.05 = sig_svg_count
  )
  
  global_summary_df <- bind_rows(global_summary_df, single_record)
  write.csv(single_record, csv_single_sum, row.names = FALSE)
  
  # Debug check
  cat(sprintf("  DONE | ARI: %.4f, NMI: %.4f, Significant SVGs: %d\n", ari_score, nmi_score, sig_svg_count))
  cat(sprintf("  Check files: h5ad exists? %s, svg csv exists? %s\n", file.exists(h5ad_out), file.exists(csv_full)))
  
  rm(adata, clustered_adata, svg_full_df, svg_sig)
  gc()
}

# Export merged global table
global_csv <- file.path(OUTPUT_ROOT_DIR, "iSpatial_All_Datasets_Global_Summary.csv")
write.csv(global_summary_df, global_csv, row.names = FALSE)
cat(sprintf("\nGlobal summary file exists? %s\n", file.exists(global_csv)))

cat("\n", strrep("=",80), "\n")
cat("ALL 7 DATASETS COMPLETED\n")
cat(sprintf("Global summary saved to: %s\n", OUTPUT_ROOT_DIR))
cat(strrep("=",80), "\n")