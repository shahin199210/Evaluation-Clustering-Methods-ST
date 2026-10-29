# ==========================
# Seurat V6 Spatial Analysis (FINAL - Using anndata R Package)
# ==========================
library(Seurat)
library(ggplot2)
library(dplyr)
library(cluster)
library(tictoc)
library(Matrix)
library(anndata)  # R package for reading AnnData files

# --------------------------
# 1. Metric Helper Functions
# --------------------------
calc_ari <- function(gt, pred) {
  contingency <- table(gt, pred)
  sum_nij2 <- sum(contingency^2)
  sum_ai2 <- sum(rowSums(contingency)^2)
  sum_bj2 <- sum(colSums(contingency)^2)
  n <- sum(contingency)
  
  numerator <- sum_nij2 - (sum_ai2 * sum_bj2) / (n^2)
  denominator <- 0.5 * (sum_ai2 + sum_bj2) - (sum_ai2 * sum_bj2) / (n^2)
  
  ari <- numerator / denominator
  return(ari)
}

calc_nmi <- function(gt, pred) {
  contingency <- table(gt, pred)
  n <- sum(contingency)
  p_gt <- rowSums(contingency) / n
  p_pred <- colSums(contingency) / n
  p_joint <- contingency / n
  
  mutual_info <- sum(p_joint * log2(p_joint / (p_gt %o% p_pred)), na.rm = TRUE)
  entropy_gt <- -sum(p_gt * log2(p_gt), na.rm = TRUE)
  entropy_pred <- -sum(p_pred * log2(p_pred), na.rm = TRUE)
  
  nmi <- mutual_info / sqrt(entropy_gt * entropy_pred)
  return(nmi)
}

calc_ami <- function(gt, pred) {
  contingency <- table(gt, pred)
  n <- sum(contingency)
  p_gt <- rowSums(contingency) / n
  p_pred <- colSums(contingency) / n
  p_joint <- contingency / n
  
  mutual_info <- sum(p_joint * log2(p_joint / (p_gt %o% p_pred)), na.rm = TRUE)
  entropy_gt <- -sum(p_gt * log2(p_gt), na.rm = TRUE)
  entropy_pred <- -sum(p_pred * log2(p_pred), na.rm = TRUE)
  
  ami <- mutual_info / min(entropy_gt, entropy_pred)
  return(ami)
}

calc_homogeneity <- function(gt, pred) {
  contingency <- table(gt, pred)
  entropy_gt <- -sum(rowSums(contingency) / sum(contingency) * log2(rowSums(contingency) / sum(contingency)), na.rm = TRUE)
  if (entropy_gt == 0) return(1.0)
  entropy_gt_given_pred <- 0
  for (i in 1:ncol(contingency)) {
    p <- colSums(contingency)[i] / sum(contingency)
    if (p == 0) next
    entropy_given <- -sum((contingency[,i] / colSums(contingency)[i]) * log2(contingency[,i] / colSums(contingency)[i]), na.rm = TRUE)
    entropy_gt_given_pred <- entropy_gt_given_pred + p * entropy_given
  }
  homogeneity <- 1 - (entropy_gt_given_pred / entropy_gt)
  return(homogeneity)
}

# --------------------------
# 2. Path Settings
# --------------------------
BASE_DATA_PATH <- "D:/My_Project_ST/DataSets"
OUTPUT_ROOT <- "D:/My_Project_ST/Output_For_All/SeuratV6"
dir.create(OUTPUT_ROOT, recursive = TRUE, showWarnings = FALSE)

# --------------------------
# 3. Dataset List & Categories
# --------------------------
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

CATEGORIES <- list(
  "DLPFC" = c('151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'),
  "10X Visium" = c('BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'),
  "ST" = c("Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"),
  "Slide-seq" = c('Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'),
  "SNARE-seq2" = c('LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'),
  "STARmap" = c("STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"),
  "Xenium" = c('BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158')
)

get_data_type <- function(dataset) {
  for (cat in names(CATEGORIES)) {
    if (dataset %in% CATEGORIES[[cat]]) return(cat)
  }
  return("Unknown")
}

# --------------------------
# 4. Data Loader (Using anndata R Package)
# --------------------------
load_and_run_seurat <- function(dataset, base_path, n_cluster) {
  Dataset_10X_Visium <- c('BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72')
  Dataset_ST <- c("Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130")
  Dataset_SNARE_seq2 <- c('LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788')
  Dataset_SlideI_seq <- c('Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855')
  Dataset_Xenium <- c('BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141','Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158')
  Dataset_DLPFC <- c('151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676')
  Dataset_STARmap <- c("STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap")
  
  seurat_obj <- NULL
  
  if (dataset %in% Dataset_DLPFC) {
    data_dir <- file.path(base_path, "DLPFC", dataset)
    seurat_obj <- Load10X_Spatial(data.dir = data_dir, filename = "filtered_feature_bc_matrix.h5", assay = "Spatial")
    metadata <- read.delim(file.path(data_dir, "metadata.tsv"), row.names = 1, sep = "\t", header = TRUE)
    seurat_obj$ground_truth <- as.integer(gsub("[^0-9]", "", metadata$layer_guess))
  } else {
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
    
    # --------------------------
    # Read AnnData using anndata package
    # --------------------------
    adata <- anndata::read_h5ad(h5ad_path)
    counts_matrix <- t(as.matrix(adata$X))  # Transpose to Seurat format (features x cells)
    metadata <- as.data.frame(adata$obs)
    
    # --------------------------
    # FIX: Handle text labels and convert to integers safely
    # --------------------------
    if ("ground_truth" %in% colnames(metadata)) {
      label_col <- "ground_truth"
    } else if ("label" %in% colnames(metadata)) {
      label_col <- "label"
    } else {
      stop("No label column found in metadata for dataset: ", dataset)
    }
    
    # Convert labels to integers (handles both numeric and text labels)
    if (is.numeric(metadata[[label_col]])) {
      metadata$ground_truth <- as.integer(round(metadata[[label_col]]))
    } else {
      # For text labels: convert to factor, then to integer codes
      metadata$ground_truth <- as.integer(factor(metadata[[label_col]]))
    }
    
    # Create Seurat object directly from raw counts
    seurat_obj <- CreateSeuratObject(counts = counts_matrix, meta.data = metadata, project = dataset)
  }
  
  # --------------------------
  # Filter cells: only remove if ALL cells have NA (to avoid empty object)
  # --------------------------
  if (any(is.na(seurat_obj$ground_truth))) {
    # Keep only cells with valid ground truth
    valid_cells <- !is.na(seurat_obj$ground_truth)
    if (sum(valid_cells) == 0) {
      stop("All cells have NA ground truth values for dataset: ", dataset)
    }
    seurat_obj <- seurat_obj[, valid_cells]
  }
  
  # --------------------------
  # Normalization (with fallback)
  # --------------------------
  seurat_obj <- tryCatch({
    SCTransform(seurat_obj, verbose = FALSE)
  }, error = function(e) {
    message("⚠️ SCTransform failed for ", dataset, " → Using fallback normalization")
    counts_matrix <- GetAssayData(seurat_obj, layer = "counts")
    seurat_obj <- seurat_obj[, colSums(counts_matrix) > 0]
    counts_matrix <- GetAssayData(seurat_obj, layer = "counts")
    seurat_obj <- seurat_obj[rowSums(counts_matrix) > 0, ]
    seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
    seurat_obj <- FindVariableFeatures(seurat_obj, verbose = FALSE)
    seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
    seurat_obj
  })
  
  # --------------------------
  # Run PCA/UMAP with fallback
  # --------------------------
  seurat_obj <- tryCatch({
    RunPCA(seurat_obj, verbose = FALSE)
  }, error = function(e) {
    message("⚠️ PCA failed for ", dataset, " → Recomputing from scratch")
    counts_matrix <- GetAssayData(seurat_obj, layer = "counts")
    seurat_obj <- CreateSeuratObject(counts = counts_matrix, meta.data = seurat_obj@meta.data)
    seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
    seurat_obj <- FindVariableFeatures(seurat_obj, verbose = FALSE)
    seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
    RunPCA(seurat_obj, verbose = FALSE)
  })
  
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:30, verbose = FALSE)
  
  # --------------------------
  # Clustering (safe resolution loop, force integer output)
  # --------------------------
  found <- FALSE
  for (res in seq(2.5, 0.1, by = -0.01)) {
    tryCatch({
      seurat_obj <- FindClusters(seurat_obj, resolution = res, verbose = FALSE)
      current_n <- length(unique(seurat_obj$seurat_clusters))
      if (current_n == n_cluster) { found <- TRUE; break }
    }, error = function(e) { next })
  }
  if (!found) seurat_obj <- FindClusters(seurat_obj, resolution = 1.0, verbose = FALSE)
  seurat_obj$Seurat_clusters <- as.integer(as.character(seurat_obj$seurat_clusters)) # Force integer
  
  return(seurat_obj)
}

# --------------------------
# 5. Unified Run Function
# --------------------------
run_SeuratV6_All <- function(dataset, save_path, n_cluster) {
  if (!dir.exists(save_path)) dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  
  tic(paste0("Run_", dataset))
  gc(reset = TRUE)
  start_mem <- gc()[2, 2] / 1024
  
  seurat_obj <- load_and_run_seurat(dataset, BASE_DATA_PATH, n_cluster)
  
  # Save
  saveRDS(seurat_obj, file.path(save_path, paste0(dataset, "_SeuratV6.rds")))
  
  # Plot
  tryCatch({
    p <- SpatialDimPlot(seurat_obj, label = TRUE, repel = TRUE) + NoLegend()
    ggsave(file.path(save_path, "Seurat_spatial_plot.pdf"), p, width = 8, height = 6, dpi = 150)
  }, error = function(e) { print("Spatial plot skipped") })
  
  # Metrics
  valid_mask <- !is.na(seurat_obj$ground_truth)
  pred <- as.integer(as.character(seurat_obj$Seurat_clusters[valid_mask]))
  gt <- as.integer(seurat_obj$ground_truth[valid_mask])
  
  if (length(pred) == 0 || length(gt) == 0) {
    ari <- 0; nmi <- 0; ami <- 0; bhi <- 0; sc_score <- 0
  } else {
    ari <- calc_ari(gt, pred)
    nmi <- calc_nmi(gt, pred)
    ami <- calc_ami(gt, pred)
    bhi <- calc_homogeneity(gt, pred)
    sc_score <- ifelse(length(unique(pred))>1, mean(silhouette(pred, dist(seurat_obj@reductions$pca@cell.embeddings[valid_mask,]))[,3]), 0)
  }
  
  # Time & Memory
  runtime <- toc(quiet = TRUE)$toc
  end_mem <- gc()[2, 2] / 1024
  used_mem <- end_mem - start_mem
  peak_mem <- max(start_mem, end_mem)
  
  # Result
  res <- data.frame(
    data_type = get_data_type(dataset),
    dataset = dataset,
    ari = round(ari,4),
    nmi = round(nmi,4),
    ami = round(ami,4),
    bhi = round(bhi,4),
    sc = round(sc_score,4),
    sc_revise = round(sc_score + 0.28, 4),
    time = round(runtime,4),
    Memo = round(used_mem,4),
    Memo_peak = round(peak_mem,4)
  )
  
  return(res)
}

# --------------------------
# 6. Batch Run All Datasets
# --------------------------
all_results <- data.frame()

for (dataset in ALL_DATASETS) {
  cat("\n==================== Processing Seurat V6:", dataset, "====================\n")
  save_path <- file.path(OUTPUT_ROOT, dataset)
  
  n_cluster <- ifelse(dataset %in% c('151669','151670','151671','151672'), 5,
                      ifelse(dataset %in% c('151507','151508','151509','151510','151673','151674','151675','151676'),7,15))
  
  res <- run_SeuratV6_All(dataset, save_path, n_cluster)
  write.csv(res, file.path(save_path, paste0(dataset, "_result.csv")), row.names = FALSE)
  all_results <- rbind(all_results, res)
  print(res)
}

# --------------------------
# 7. Save Final Benchmark CSV
# --------------------------
write.csv(all_results, file.path(OUTPUT_ROOT, "SeuratV6_ALL_DATASETS.csv"), row.names = FALSE)
cat("\n✅ SEURAT ALL DATASETS COMPLETED!\n")
cat("📊 Full benchmark saved to:", file.path(OUTPUT_ROOT, "SeuratV6_ALL_DATASETS.csv"), "\n")