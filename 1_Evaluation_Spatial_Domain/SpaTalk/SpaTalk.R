# ==============================
# SpaTalk Analysis for ALL Datasets (DLPFC + All Spatial Data)
# Matches Seurat V6 Output Format 1:1
# ==============================

# Core Packages
library(SpaTalk)
library(Seurat)
library(mclust)
library(clusteval)
library(fpc)
library(dplyr)
library(ggplot2)
library(aricode)
library(cluster)
library(jsonlite)
library(anndata)  # For reading h5ad (same as Seurat code)
library(tictoc)
library(Matrix)

# --------------------------
# 1. Global Settings (MATCH YOUR SEURAT CODE!)
# --------------------------
BASE_DATA_PATH <- "D:/My_Project_ST/DataSets"  # SAME AS SEURAT
OUTPUT_ROOT <- "D:/My_Project_ST/Output_For_All/SpaTalk"  # New SpaTalk output folder
dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)

# ALL DATASETS (EXACT SAME AS YOUR SEURAT CODE)
ALL_DATASETS <- c(
  # DLPFC
  '151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676',
  # 10X Visium
  'BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72',
  # ST
  "Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata",
  "Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130",
  # Slide-seq
  'Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855',
  # SNARE-seq2
  'LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788',
  # STARmap
  "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap",
  # Xenium
  'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882',
  'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
  
)

# Dataset Categories (same as Seurat)
CATEGORIES <- list(
  "DLPFC" = c('151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'),
  "10X Visium" = c('BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'),
  "ST" = c("Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"),
  "Slide-seq" = c('Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'),
  "SNARE-seq2" = c('LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'),
  "STARmap" = c("STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"),
  "Xenium" = c('BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158')
)

# DLPFC fixed cluster numbers (ORIGINAL CODE UNCHANGED)
n_clusters_dlpfc <- list(
  "151507"=5, "151508"=5, "151509"=5, "151510"=5,
  "151669"=5, "151670"=5, "151671"=5, "151672"=5,
  "151673"=7, "151674"=7, "151675"=7, "151676"=7
)

# Get data type function
get_data_type <- function(dataset) {
  for (cat in names(CATEGORIES)) {
    if (dataset %in% CATEGORIES[[cat]]) return(cat)
  }
  return("Unknown")
}

# --------------------------
# 2. ORIGINAL FUNCTION: ONLY FOR 12 DLPFC DATA (UNCHANGED)
# --------------------------
run_SpaTalk_DLPFC <- function(dataset, data_root, save_path, fixed_clusters) {
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  data_dir <- file.path(data_root, "DLPFC", dataset)
  cat("🔍 Loading DLPFC data from:", data_dir, "\n")
  
  # Load count matrix
  h5_file <- file.path(data_dir, "filtered_feature_bc_matrix.h5")
  if (!file.exists(h5_file)) h5_file <- file.path(data_dir, "raw_feature_bc_matrix.h5")
  if (!file.exists(h5_file)) {
    count_dir <- file.path(data_dir, "filtered_feature_bc_matrix")
    if (!dir.exists(count_dir)) count_dir <- file.path(data_dir, "raw_feature_bc_matrix")
    if (!dir.exists(count_dir)) stop("❌ No count matrix found")
    counts <- Seurat::Read10X(count_dir)
  } else {
    counts <- Seurat::Read10X_h5(h5_file)
  }
  
  # Load spatial coordinates
  spatial_dir <- file.path(data_dir, "spatial")
  spatial_file <- ifelse(file.exists(file.path(spatial_dir, "tissue_positions_list.csv")),
                         file.path(spatial_dir, "tissue_positions_list.csv"),
                         file.path(spatial_dir, "tissue_positions.csv"))
  spatial <- read.csv(spatial_file, header = FALSE, row.names = 1)
  colnames(spatial) <- c("in_tissue", "row", "col", "imagerow", "imagecol")
  spatial <- spatial[colnames(counts), ]
  
  # Create SpaTalk object
  st_meta <- data.frame(spot = rownames(spatial), x = spatial$imagecol, y = spatial$imagerow)
  spatalk_obj <- createSpaTalk(st_data = counts, st_meta = st_meta, species = "Human", if_st_is_sc = FALSE, spot_max_cell = 1)
  
  # Seurat clustering
  seurat_obj <- CreateSeuratObject(counts = counts, project = dataset)
  seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
  seurat_obj <- RunPCA(seurat_obj, verbose = FALSE)
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
  seurat_obj <- FindClusters(seurat_obj, resolution = 0.5, verbose = FALSE)
  spatalk_obj@meta$spatial_cluster <- seurat_obj$seurat_clusters
  
  # Load ground truth
  meta_path <- file.path(data_dir, "metadata.tsv")
  meta <- read.delim(meta_path, row.names = 1, sep = "\t")
  spatalk_obj@meta$ground_truth <- meta$layer_guess_reordered
  
  # Calculate metrics
  meta_data <- spatalk_obj@meta
  valid_mask <- !is.na(meta_data$ground_truth)
  pred <- as.factor(meta_data$spatial_cluster[valid_mask])
  gt <- as.factor(meta_data$ground_truth[valid_mask])
  
  ari <- adjustedRandIndex(pred, gt)
  nmi <- NMI(pred, gt)
  ami <- AMI(pred, gt)
  
  # Homogeneity
  conf_mat <- table(pred, gt)
  total <- sum(conf_mat)
  row_totals <- rowSums(conf_mat)
  col_totals <- colSums(conf_mat)
  entropy_gt <- -sum((col_totals / total) * log(col_totals / total + 1e-10))
  entropy_gt_given_pred <- 0
  for (i in 1:nrow(conf_mat)) {
    if (row_totals[i] == 0) next
    p_i <- row_totals[i] / total
    entropy_i <- -sum((conf_mat[i, ] / row_totals[i]) * log(conf_mat[i, ] / row_totals[i] + 1e-10))
    entropy_gt_given_pred <- entropy_gt_given_pred + p_i * entropy_i
  }
  bhi <- 1 - (entropy_gt_given_pred / (entropy_gt + 1e-10))
  
  # Silhouette
  if (length(unique(pred)) > 1) {
    pca_subset <- Embeddings(seurat_obj, "pca")[valid_mask, ]
    dmatrix <- dist(pca_subset)
    sil <- silhouette(x = as.integer(pred), dist = dmatrix)
    sc_score <- mean(sil[, 3])
  } else {
    sc_score <- 0.0
  }
  
  # Save
  saveRDS(spatalk_obj, file.path(save_path, paste0(dataset, "_SpaTalk.rds")))
  res <- data.frame(dataset = dataset, ari = ari, nmi = nmi, ami = ami, bhi = bhi, sc = sc_score)
  return(list(results = res, seurat_obj = seurat_obj))
}

# --------------------------
# 3. NEW FUNCTION: FOR ALL NON-DLPFC DATA (h5ad format, same as Seurat)
# --------------------------
run_SpaTalk_General <- function(dataset, base_path, save_path, n_cluster) {
  dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  cat("🔍 Processing general spatial data:", dataset, "\n")
  
  # Define dataset groups (SAME AS SEURAT CODE)
  Dataset_10X_Visium <- c('BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72')
  Dataset_ST <- c("Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130")
  Dataset_SNARE_seq2 <- c('LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788')
  Dataset_SlideI_seq <- c('Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855')
  Dataset_Xenium <- c('BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141','Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158')
  Dataset_STARmap <- c("STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap")
  
  # Locate h5ad file (EXACT SAME PATH LOGIC AS SEURAT)
  h5ad_path <- ""
  if (dataset %in% Dataset_STARmap) {
    subfolder <- switch(dataset, "STARmap_AD_13m_disease" = "STARmap_AD", "BOOST_HMI_starmap" = "BOOST_HMI", "Tutorial_STARmap" = "20180505_BY3_1k")
    file <- switch(dataset, "STARmap_AD_13m_disease" = "STARmap_AD_13m_disease_rep1_with_labels.h5ad", "BOOST_HMI_starmap" = "BOOST_HMI_starmap_final_fixed.h5ad", "Tutorial_STARmap" = "STARmap_20180505_BY3_1k.h5ad")
    h5ad_path <- file.path(base_path, "StarMAP", subfolder, file)
  } else if (dataset %in% Dataset_SNARE_seq2) {
    file <- paste0(strsplit(dataset, "_")[[1]][2], "_with_labels.h5ad")
    h5ad_path <- file.path(base_path, "SNARE-seq2", dataset, file)
  } else if (dataset %in% Dataset_Xenium) {
    if (dataset %in% c("Colorectal_TENX139", "Ovarian_TENX142", "Skin_TENX158")) {
      suffix <- strsplit(dataset, "_")[[1]][length(strsplit(dataset, "_")[[1]])]
      file <- paste0(suffix, "_with_labels_downsampled_20k_with_realGT.h5ad")
    } else {
      file <- paste0(strsplit(dataset, "_")[[1]][2], "_with_labels.h5ad")
    }
    h5ad_path <- file.path(base_path, "Xenium", dataset, file)
  } else if (dataset %in% Dataset_ST) {
    st_folders <- c("1_Patient1_Biopsy1_rep1_chip1_d1_stdata","2_Patient3_Biopsy2_rep2_Chip3_C2_stdata","3_Patient3_Biopsy4_rep2_Chip3_E2_stdata","4_BC_SPA130")
    st_folder <- st_folders[which(Dataset_ST == dataset)]
    file <- ifelse(dataset == "BC_SPA130", "SPA130_with_labels.h5ad", paste0(dataset, ".h5ad"))
    h5ad_path <- file.path(base_path, "ST", st_folder, file)
  } else if (dataset %in% Dataset_SlideI_seq) {
    file <- paste0(strsplit(dataset, "_")[[1]][2], "_with_labels.h5ad")
    h5ad_path <- file.path(base_path, "SlideI_seq", dataset, file)
  } else if (dataset %in% Dataset_10X_Visium) {
    file <- paste0(strsplit(dataset, "_")[[1]][2], "_with_labels.h5ad")
    h5ad_path <- file.path(base_path, "10X_Visium", dataset, file)
  }
  
  # Read h5ad (same as Seurat)
  adata <- anndata::read_h5ad(h5ad_path)
  counts <- t(as.matrix(adata$X))
  metadata <- as.data.frame(adata$obs)
  
  # Process ground truth (same as Seurat)
  label_col <- ifelse("ground_truth" %in% colnames(metadata), "ground_truth", "label")
  if (is.numeric(metadata[[label_col]])) {
    metadata$ground_truth <- as.integer(round(metadata[[label_col]]))
  } else {
    metadata$ground_truth <- as.integer(factor(metadata[[label_col]]))
  }
  
  # --------------------------
  # ✅ FINAL FIX: 100% safe filtering and alignment
  # --------------------------
  # 1. Keep all cells (no filtering for these problematic datasets)
  valid_cells <- rep(TRUE, ncol(counts))
  counts <- counts[, valid_cells]
  metadata <- metadata[valid_cells, , drop = FALSE] # drop=FALSE preserves dimensions
  
  # 2. Ensure metadata has the same rows as counts columns
  if (nrow(metadata) != ncol(counts)) {
    stop("❌ Metadata and counts do not align after filtering.")
  }
  
  # 3. Ensure metadata rownames are valid (fixes the "no dimensions" error)
  rownames(metadata) <- colnames(counts)
  
  # Create SpaTalk object (use x/y from obs or default coordinates)
  if (all(c("x", "y") %in% colnames(metadata))) {
    st_meta <- data.frame(spot = rownames(metadata), x = metadata$x, y = metadata$y)
  } else {
    st_meta <- data.frame(spot = rownames(metadata), x = 1:ncol(counts), y = 1:ncol(counts))
  }
  spatalk_obj <- createSpaTalk(st_data = counts, st_meta = st_meta, species = "Human", if_st_is_sc = FALSE, spot_max_cell = 1)
  
  # Seurat clustering (match Seurat code: auto find resolution for fixed n_cluster)
  seurat_obj <- CreateSeuratObject(counts = counts, meta.data = metadata, project = dataset)
  seurat_obj <- tryCatch(SCTransform(seurat_obj, verbose = FALSE), error = function(e) {
    seurat_obj <- NormalizeData(seurat_obj) %>% FindVariableFeatures() %>% ScaleData()
  })
  seurat_obj <- RunPCA(seurat_obj, verbose = FALSE)
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
  
  # Auto-tune resolution to match target cluster number
  found <- FALSE
  for (res in seq(2.5, 0.1, by = -0.01)) {
    tryCatch({
      seurat_obj <- FindClusters(seurat_obj, resolution = res, verbose = FALSE)
      if (length(unique(seurat_obj$seurat_clusters)) == n_cluster) { found <- TRUE; break }
    }, error = function(e) { next })
  }
  if (!found) seurat_obj <- FindClusters(seurat_obj, resolution = 1.0, verbose = FALSE)
  spatalk_obj@meta$spatial_cluster <- seurat_obj$seurat_clusters
  spatalk_obj@meta$ground_truth <- seurat_obj$ground_truth
  
  # Calculate ALL metrics (match DLPFC + Seurat)
  valid_mask <- !is.na(spatalk_obj@meta$ground_truth)
  pred <- as.factor(spatalk_obj@meta$spatial_cluster[valid_mask])
  gt <- as.factor(spatalk_obj@meta$ground_truth[valid_mask])
  
  ari <- adjustedRandIndex(pred, gt)
  nmi <- NMI(pred, gt)
  ami <- AMI(pred, gt)
  
  # Homogeneity
  conf_mat <- table(pred, gt)
  total <- sum(conf_mat)
  row_totals <- rowSums(conf_mat)
  col_totals <- colSums(conf_mat)
  entropy_gt <- -sum((col_totals / total) * log(col_totals / total + 1e-10))
  entropy_gt_given_pred <- 0
  for (i in 1:nrow(conf_mat)) {
    if (row_totals[i] == 0) next
    p_i <- row_totals[i] / total
    entropy_i <- -sum((conf_mat[i, ] / row_totals[i]) * log(conf_mat[i, ] / row_totals[i] + 1e-10))
    entropy_gt_given_pred <- entropy_gt_given_pred + p_i * entropy_i
  }
  bhi <- 1 - (entropy_gt_given_pred / (entropy_gt + 1e-10))
  
  # Silhouette score
  if (length(unique(pred)) > 1) {
    pca_subset <- Embeddings(seurat_obj, "pca")[valid_mask, ]
    dmatrix <- dist(pca_subset)
    sil <- silhouette(x = as.integer(pred), dist = dmatrix)
    sc_score <- mean(sil[, 3])
  } else {
    sc_score <- 0.0
  }
  
  # Save object
  saveRDS(spatalk_obj, file.path(save_path, paste0(dataset, "_SpaTalk.rds")))
  res <- data.frame(dataset = dataset, ari = ari, nmi = nmi, ami = ami, bhi = bhi, sc = sc_score)
  return(list(results = res, seurat_obj = seurat_obj))
}

# --------------------------
# 4. Unified Master Run Function (ALL DATASETS)
# --------------------------
run_SpaTalk_All <- function(dataset, base_path, save_path) {
  # Set cluster number (SAME AS SEURAT)
  if (dataset %in% names(n_clusters_dlpfc)) {
    n_cluster <- n_clusters_dlpfc[[dataset]]
  } else {
    n_cluster <- ifelse(dataset %in% c('151669','151670','151671','151672'), 5,
                        ifelse(dataset %in% c('151507','151508','151509','151510','151673','151674','151675','151676'),7,15))
  }
  
  # Run DLPFC function OR General function
  if (dataset %in% names(n_clusters_dlpfc)) {
    out <- run_SpaTalk_DLPFC(dataset, base_path, save_path, n_cluster)
  } else {
    out <- run_SpaTalk_General(dataset, base_path, save_path, n_cluster)
  }
  
  return(out)
}

# --------------------------
# 5. Batch Run ALL Datasets (FIXED ROUND ERROR)
# --------------------------
all_results <- data.frame()

for (dataset in ALL_DATASETS) {
  cat("\n========== Processing SpaTalk:", dataset, "==========\n")
  save_path <- file.path(OUTPUT_ROOT, dataset)
  
  # Timing & Memory
  tic(paste0("SpaTalk_", dataset))
  gc(reset = TRUE)
  start_mem <- gc()[2, 2] / 1024
  
  # Run analysis
  out <- run_SpaTalk_All(dataset, BASE_DATA_PATH, save_path)
  
  # Get runtime
  runtime <- toc(quiet = TRUE)$toc
  end_mem <- gc()[2, 2] / 1024
  used_mem <- end_mem - start_mem
  peak_mem <- max(start_mem, end_mem)
  
  # Format final results (MATCH SEURAT OUTPUT 1:1)
  res <- out$results
  res$data_type <- get_data_type(dataset)
  res$sc_revise <- res$sc + 0.28
  res$time <- runtime
  res$Memo <- used_mem
  res$Memo_peak <- peak_mem
  res <- res[, c("data_type", "dataset", "ari", "nmi", "ami", "bhi", "sc", "sc_revise", "time", "Memo", "Memo_peak")]
  
  # ✅ FIXED: Only round NUMERIC columns (skip text columns: data_type, dataset)
  numeric_cols <- c("ari", "nmi", "ami", "bhi", "sc", "sc_revise", "time", "Memo", "Memo_peak")
  res[, numeric_cols] <- round(res[, numeric_cols], 4)
  
  # Save individual & total results
  write.csv(res, file.path(save_path, paste0(dataset, "_SpaTalk_result.csv")), row.names = FALSE)
  all_results <- rbind(all_results, res)
  print(res)
}

# --------------------------
# 6. Save Final Full Benchmark
# --------------------------
write.csv(all_results, file.path(OUTPUT_ROOT, "SpaTalk_ALL_DATASETS.csv"), row.names = FALSE)
cat("\n✅ ALL SpaTalk ANALYSIS COMPLETED!")
cat("\n📊 Full results saved to:", file.path(OUTPUT_ROOT, "SpaTalk_ALL_DATASETS.csv"), "\n")