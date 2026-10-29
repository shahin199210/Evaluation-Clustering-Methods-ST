# ---------------------------
# Required Core Packages Only
# ---------------------------
library(SpaTalk)
library(Seurat)
library(dplyr)
library(anndata)
library(Matrix)
library(stats)

# ---------------------------
# Global Fixed Path Configuration
# ---------------------------
INPUT_DIR  <- "D:/My_Project_ST/benchmarking of ST platforms/Datasets_Standardized"
OUTPUT_DIR <- "D:/My_Project_ST/benchmarking of ST platforms/SVGs_Clustering_Method/SpaTalk"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------
# Predefined Dataset List & Target Cluster Number Settings
# ---------------------------
# Standardized h5ad file list
DATASET_FILE_LIST <- c(
  "Xenium_BC_NCBI784_standardized.h5ad",
  "Xenium_Lung_NCBI856_standardized.h5ad",
  "10X_BC_TENX13_standardized.h5ad",
  "10X_Lung_TENX62_standardized.h5ad",
  "ST_BC_SPA130_standardized.h5ad",
  "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
  "STARmap_BOOST_HMI_standardized.h5ad"
)

# Target cluster count for each dataset
CLUSTER_CONFIG <- list(
  "Xenium_BC_NCBI784" = 8,
  "Xenium_Lung_NCBI856" = 6,
  "10X_BC_TENX13" = 5,
  "10X_Lung_TENX62" = 6,
  "ST_BC_SPA130" = 4,
  "Slide-seq2_Kedney_HBM965.PSNC.855" = 7,
  "STARmap_BOOST_HMI" = 3
)

# Platform category mapping
PLATFORM_MAPPING <- list(
  "Xenium_BC_NCBI784_standardized.h5ad" = "Xenium",
  "Xenium_Lung_NCBI856_standardized.h5ad" = "Xenium",
  "10X_BC_TENX13_standardized.h5ad" = "10X Visium",
  "10X_Lung_TENX62_standardized.h5ad" = "10X Visium",
  "ST_BC_SPA130_standardized.h5ad" = "ST",
  "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad" = "Slide-seq",
  "STARmap_BOOST_HMI_standardized.h5ad" = "STARmap"
)

# ---------------------------
# Helper Function 1: Auto search Leiden resolution for fixed cluster number
# ---------------------------
search_fixed_resolution <- function(seu_obj, target_cluster_num, start_res = 2.5, end_res = 0.1, step = -0.01){
  resolution_seq <- seq(from = start_res, to = end_res, by = step)
  final_res <- 1.0
  for(res in resolution_seq){
    tryCatch({
      seu_temp <- FindClusters(seu_obj, resolution = res, verbose = FALSE)
      current_clust_num <- length(unique(seu_temp$seurat_clusters))
      if(current_clust_num == target_cluster_num){
        final_res <- res
        break
      }
    }, error = function(e){next})
  }
  return(final_res)
}

# ---------------------------
# Helper Function 2: Permutation SVG calculation (FDR-BH correction, consistent with stRGAT standard)
# ---------------------------
calc_cluster_SVG <- function(expr_matrix, cluster_label_vec, perm_times = 1000, seed = 0){
  set.seed(seed)
  cell_num <- nrow(expr_matrix)
  gene_num <- ncol(expr_matrix)
  unique_cluster <- unique(cluster_label_vec)
  clust_count <- length(unique_cluster)
  
  # Global gene mean expression
  global_mean <- colMeans(expr_matrix)
  
  # Observed SVG score
  clust_mean_mat <- matrix(0, nrow = clust_count, ncol = gene_num)
  for(i in seq_along(unique_cluster)){
    cell_mask <- cluster_label_vec == unique_cluster[i]
    clust_mean_mat[i, ] <- colMeans(expr_matrix[cell_mask, , drop = FALSE])
  }
  obs_score <- colSums((clust_mean_mat - matrix(global_mean, nrow = clust_count, ncol = gene_num, byrow = TRUE)) ^ 2) / clust_count
  
  # Permutation null distribution
  perm_score_mat <- matrix(0, nrow = perm_times, ncol = gene_num)
  for(p in 1:perm_times){
    shuffle_idx <- sample(1:cell_num, replace = FALSE)
    perm_expr <- expr_matrix[shuffle_idx, ]
    perm_clust_mean <- matrix(0, nrow = clust_count, ncol = gene_num)
    for(i in seq_along(unique_cluster)){
      cell_mask <- cluster_label_vec == unique_cluster[i]
      perm_clust_mean[i, ] <- colMeans(perm_expr[cell_mask, , drop = FALSE])
    }
    perm_score_mat[p, ] <- colSums((perm_clust_mean - matrix(global_mean, nrow = clust_count, ncol = gene_num, byrow = TRUE)) ^ 2) / clust_count
  }
  
  # P-value calculation & FDR-BH correction
  raw_p <- (colSums(perm_score_mat >= matrix(obs_score, nrow = perm_times, ncol = gene_num, byrow = TRUE)) + 1) / (perm_times + 1)
  fdr_result <- p.adjust(raw_p, method = "BH")
  sig_flag <- fdr_result < 0.05
  
  # Assemble SVG result table
  svg_df <- data.frame(
    gene_name = colnames(expr_matrix),
    SpaTalk_SVG_score = obs_score,
    raw_p_value = raw_p,
    fdr_BH_adjust_p = fdr_result,
    is_SVG_FDR0.05 = sig_flag
  )
  svg_df <- svg_df[order(-svg_df$SpaTalk_SVG_score), ]
  rownames(svg_df) <- NULL
  return(svg_df)
}

# ---------------------------
# Core Pipeline: SpaTalk Construction + Clustering + SVG Identification
# ---------------------------
run_SpaTalk_SVG <- function(h5ad_file_path, dataset_name, target_clust_num, save_folder){
  dir.create(save_folder, recursive = TRUE, showWarnings = FALSE)
  # 1. Read standardized h5ad file
  adata <- read_h5ad(h5ad_file_path)
  expr_raw <- as.matrix(adata$X)
  meta_df <- as.data.frame(adata$obs)
  gene_names <- adata$var_names
  colnames(expr_raw) <- gene_names
  rownames(expr_raw) <- rownames(meta_df)
  
  # 2. Build Seurat object & standard preprocessing
  seu_obj <- CreateSeuratObject(counts = t(expr_raw), meta.data = meta_df, project = dataset_name)
  seu_obj <- tryCatch({
    SCTransform(seu_obj, verbose = FALSE)
  }, error = function(e){
    seu_obj <- NormalizeData(seu_obj, verbose = FALSE)
    seu_obj <- FindVariableFeatures(seu_obj, verbose = FALSE)
    seu_obj <- ScaleData(seu_obj, verbose = FALSE)
    return(seu_obj)
  })
  seu_obj <- RunPCA(seu_obj, verbose = FALSE)
  seu_obj <- FindNeighbors(seu_obj, dims = 1:30, verbose = FALSE)
  
  # 3. Fixed-number Leiden clustering
  opt_res <- search_fixed_resolution(seu_obj, target_clust_num)
  seu_obj <- FindClusters(seu_obj, resolution = opt_res, verbose = FALSE)
  cluster_labels <- seu_obj$seurat_clusters
  
  # 4. Build SpaTalk object
  if(all(c("x", "y") %in% colnames(meta_df))){
    st_meta <- data.frame(
      spot = rownames(meta_df),
      x = meta_df$x,
      y = meta_df$y
    )
  }else{
    st_meta <- data.frame(
      spot = rownames(meta_df),
      x = seq_len(nrow(meta_df)),
      y = seq_len(nrow(meta_df))
    )
  }
  spatalk_obj <- createSpaTalk(
    st_data = t(expr_raw),
    st_meta = st_meta,
    species = "Human",
    if_st_is_sc = FALSE,
    spot_max_cell = 1
  )
  spatalk_obj@meta$SpaTalk_leiden_cluster <- cluster_labels
  
  # 5. Calculate SVG with cluster labels
  svg_result <- calc_cluster_SVG(expr_raw, cluster_labels, perm_times = 1000, seed = 0)
  
  # 6. Export all output files
  saveRDS(spatalk_obj, file.path(save_folder, paste0(dataset_name, "_SpaTalk_Object.rds")))
  write.csv(svg_result, file.path(save_folder, paste0(dataset_name, "_SpaTalk_Full_SVG_List.csv")), row.names = FALSE)
  sig_svg <- subset(svg_result, is_SVG_FDR0.05 == TRUE)
  write.csv(sig_svg, file.path(save_folder, paste0(dataset_name, "_Significant_SVG_FDR0.05.csv")), row.names = FALSE)
  write.csv(head(svg_result, 50), file.path(save_folder, paste0(dataset_name, "_Top50_SVG.csv")), row.names = FALSE)
  
  # Single dataset summary record
  summary_record <- data.frame(
    dataset = dataset_name,
    target_cluster = target_clust_num,
    total_gene = length(gene_names),
    significant_SVG_count = nrow(sig_svg)
  )
  write.csv(summary_record, file.path(save_folder, paste0(dataset_name, "_Dataset_Summary.csv")), row.names = FALSE)
  
  return(list(svg_data = svg_result, summary = summary_record))
}

# ---------------------------
# Main Batch Execution
# ---------------------------
global_summary_list <- list()
cat("=============================\n")
cat("SpaTalk Cluster-Based SVG Detection Pipeline Start\n")
cat(paste0("Input Path: ", INPUT_DIR, "\n"))
cat(paste0("Output Path: ", OUTPUT_DIR, "\n"))
cat("=============================\n")

for(file_name in DATASET_FILE_LIST){
  file_full_path <- file.path(INPUT_DIR, file_name)
  dataset_name <- gsub("_standardized.h5ad", "", file_name)
  dataset_output_folder <- file.path(OUTPUT_DIR, dataset_name)
  
  # Check file existence
  if(!file.exists(file_full_path)){
    cat(paste0("Warning: File ", file_name, " missing, skip this dataset\n"))
    next
  }
  
  target_cluster <- CLUSTER_CONFIG[[dataset_name]]
  current_platform <- PLATFORM_MAPPING[[file_name]]
  cat(paste0("\nStart processing dataset: ", dataset_name, " | Target cluster: ", target_cluster, "\n"))
  
  # Run core pipeline
  run_out <- run_SpaTalk_SVG(file_full_path, dataset_name, target_cluster, dataset_output_folder)
  single_summary <- run_out$summary
  single_summary$platform <- current_platform
  global_summary_list[[dataset_name]] <- single_summary
  
  cat(paste0("Finished ", dataset_name, " | Significant SVG(FDR0.05): ", single_summary$significant_SVG_count, "\n"))
}

# Export global summary table for all datasets
global_summary_df <- do.call(rbind, global_summary_list)
global_summary_df <- global_summary_df[, c("platform", "dataset", "target_cluster", "total_gene", "significant_SVG_count")]
write.csv(global_summary_df, file.path(OUTPUT_DIR, "SpaTalk_All_Datasets_Global_SVG_Summary.csv"), row.names = FALSE)

cat("\n=============================\n")
cat("All datasets finished successfully!\n")
cat(paste0("Global summary saved at: ", file.path(OUTPUT_DIR, "SpaTalk_All_Datasets_Global_SVG_Summary.csv"), "\n"))
cat("=============================\n")
