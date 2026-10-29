# ======================================================
# iSpatial benchmark script (fixed: ground truth = Cluster)
# ======================================================
library(iSpatial)
library(SpatialExperiment)
library(scran)
library(scater)
library(bluster)
library(aricode)
library(cluster)
library(dplyr)
library(readr)
library(ps)
library(stringr)
library(Seurat)
set.seed(0)
# --------------------------
# PATH SETTINGS
# --------------------------
BASE_DATA_PATH  <- "D:/My_Project_ST/DataSets"
OUTPUT_ROOT     <- "D:/My_Project_ST/Output_For_All/iSpatial"
if (!dir.exists(OUTPUT_ROOT)) dir.create(OUTPUT_ROOT, recursive = TRUE)
# --------------------------
# YOUR EXACT DATASET LIST
# --------------------------
CATEGORIES <- list(
  "DLPFC" = c('151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'),
  "10X Visium" = c('BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'),
  "Xenium" = c('BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158','Lung_TENX141'),
  "ST" = c("Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"),
  "Slide-seq" = c('Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'),
  "STARmap" = c("STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"),
  "SNARE-seq2" = c('LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.975','LungR_HBM285.MWNX.594','LungR_HBM753.NSKJ.789')
)
ALL_DATASETS <- unlist(CATEGORIES, use.names = FALSE)
cat("Total datasets to run: ", length(ALL_DATASETS), "\n")
get_data_group <- function(ds_id){
  for(g in names(CATEGORIES)){
    if(ds_id %in% CATEGORIES[[g]]) return(g)
  }
  stop(paste("Dataset", ds_id, "not found in category list"))
}
eval_model <- function(pred, gt) {
  ari <- ARI(pred, gt)
  nmi <- NMI(pred, gt)
  ami <- AMI(pred, gt)
  bhi <- homogeneity(gt, pred)
  list(ari=ari, nmi=nmi, ami=ami, bhi=bhi)
}
load_sample_data <- function(dataset_id){
  group_name <- get_data_group(dataset_id)
  sample_folder <- file.path(BASE_DATA_PATH, group_name, dataset_id)
  if(!dir.exists(sample_folder)){
    stop(paste("Folder not found: ", sample_folder))
  }
  
  if(group_name %in% c("10X Visium", "DLPFC")){
    seu <- Load10X_Spatial(
      data.dir = sample_folder,
      filename = "filtered_feature_bc_matrix.h5"
    )
    mat <- GetAssayData(seu, assay = "Spatial", layer = "counts")
    sp_coords <- Seurat::GetTissueCoordinates(seu)
    sp_coords <- sp_coords[colnames(mat), c("x", "y")]
    sp_coords <- as.matrix(sp_coords)
    
    spe <- SpatialExperiment(
      assays = list(counts = mat),
      spatialCoords = sp_coords,
      sample_id = dataset_id
    )
  } else {
    stop(sprintf("Loader not implemented yet for platform: %s, dataset: %s", group_name, dataset_id))
  }
  
  meta_path <- file.path(sample_folder, "metadata.tsv")
  if(!file.exists(meta_path)){
    stop(paste("metadata.tsv missing in ", sample_folder))
  }
  
  meta <- read.delim(meta_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  rownames(meta) <- meta$barcode
  common_barcodes <- intersect(colnames(spe), rownames(meta))
  meta_sub <- meta[common_barcodes, , drop = FALSE]
  spe <- spe[, common_barcodes]
  colData(spe) <- cbind(colData(spe), meta_sub)
  
  return(spe)
}
# --------------------------
# Modified run_iSpatial: safer, check class explicitly
# --------------------------
run_iSpatial <- function(spe, save_path, dataset, n_cluster){
  if (!dir.exists(save_path)) dir.create(save_path, recursive = TRUE)
  cat("==== Running official iSpatial (R package) ====\n")
  
  # Run iSpatial
  isp_out <- iSpatial::infer(spe)
  cat("iSpatial output class: ", class(isp_out), "\n")
  
  # Branch: two common return formats for iSpatial
  if (inherits(isp_out, "SpatialExperiment")) {
    emb <- reducedDim(isp_out, "iSpatial")
  } else if (is.matrix(isp_out)) {
    emb <- isp_out
  } else {
    stop("Unknown return type from iSpatial::infer(). Check your installed iSpatial version.")
  }
  
  clust_res <- kmeans(emb, centers = n_cluster, nstart=20)
  pred_labels <- as.factor(clust_res$cluster)
  
  # FIXED: ground truth column = Cluster
  gt_labels <- spe$Cluster
  
  saveRDS(isp_out, file.path(save_path, paste0(dataset, "_iSpatial_out.rds")))
  
  valid_mask <- !is.na(gt_labels)
  valid_pred <- pred_labels[valid_mask]
  valid_gt <- gt_labels[valid_mask]
  valid_emb <- emb[valid_mask, , drop=FALSE]
  
  ari <- 0; nmi <-0; ami <-0; bhi <-0
  if(sum(valid_mask) > 0){
    met <- eval_model(valid_pred, valid_gt)
    ari <- met$ari
    nmi <- met$nmi
    ami <- met$ami
    bhi <- met$bhi
  }
  
  sc_score <- 0
  if(length(unique(pred_labels))>1){
    sil <- silhouette(dist(emb), cluster = as.integer(pred_labels))
    sc_score <- mean(sil[,3])
  }
  
  sc_revise <- 0
  if(length(unique(valid_gt))>1 && nrow(valid_emb)>0){
    sil_rev <- silhouette(dist(valid_emb), cluster = as.integer(as.factor(valid_gt)))
    sc_revise <- mean(sil_rev[,3])
  }
  
  cat(sprintf("✅ %s Final Results:\n", dataset))
  cat(sprintf("   ARI=%.4f | NMI=%.4f | AMI=%.4f | BHI=%.4f\n", ari,nmi,ami,bhi))
  cat(sprintf("   SC=%.4f | SC_rev=%.4f\n", sc_score, sc_revise))
  
  out_list <- list(
    dataset=dataset, ari=ari, nmi=nmi, ami=ami, bhi=bhi, sc=sc_score, sc_revise=sc_revise
  )
  return(out_list)
}
# --------------------------
# Main loop
# --------------------------
all_results <- list()
for (dataset in ALL_DATASETS) {
  cat("\n==================================================\n")
  cat("         RUNNING iSpatial: ", dataset, "\n")
  cat("==================================================\n")
  
  group_name <- get_data_group(dataset)
  save_path <- file.path(OUTPUT_ROOT, group_name, dataset)
  if (!dir.exists(save_path)) dir.create(save_path, recursive=TRUE)
  
  if(!group_name %in% c("10X Visium", "DLPFC")){
    cat(sprintf("⚠️ Skip %s (%s): loader not ready\n", dataset, group_name))
    next
  }
  
  spe <- load_sample_data(dataset)
  
  # FIXED: check "Cluster" instead of "ground_truth"
  if (! "Cluster" %in% colnames(colData(spe)) ) {
    stop(paste("Cluster column NOT found in metadata for", dataset))
  }
  n_cluster <- length(unique(spe$Cluster))
  
  p <- ps::ps_handle()
  start_time <- Sys.time()
  start_mem <- ps::ps_memory_info(p)[["rss"]] / 1024^3
  
  res <- run_iSpatial(spe, save_path, dataset, n_cluster = n_cluster)
  
  end_time <- Sys.time()
  end_mem <- ps::ps_memory_info(p)[["rss"]] / 1024^3
  used_time <- as.numeric(difftime(end_time, start_time, units="secs"))
  used_mem <- end_mem - start_mem
  peak_gb <- NA_real_
  
  cat(sprintf('Time: %.2fs | Mem: %.4fGB | Peak: %.4fGB\n', used_time, used_mem, peak_gb))
  
  full_result_tbl <- tibble(
    data_type = group_name,
    dataset = res$dataset,
    ari = res$ari,
    nmi = res$nmi,
    ami = res$ami,
    bhi = res$bhi,
    sc = res$sc,
    sc_revise = res$sc_revise,
    time = used_time,
    Memo = used_mem,
    Memo_peak = peak_gb
  )
  write_csv(full_result_tbl, file.path(save_path, paste0(dataset,"_result.csv")))
  all_results[[dataset]] <- full_result_tbl
}
final_df <- bind_rows(all_results)
write_csv(final_df, file.path(OUTPUT_ROOT, "iSpatial_ALL_DATASETS.csv"))
cat("\n iSpatial Benchmark FINISHED!\n")