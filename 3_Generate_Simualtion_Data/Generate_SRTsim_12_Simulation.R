#------------------------------------------------------------------------------
# Step0: Set options and output folder
#------------------------------------------------------------------------------
options(timeout = max(300, getOption("timeout")))
output_root <- "D:/My_Project_ST/benchmarking of ST platforms/Simulation_data"
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
setwd(output_root)
cache_dir <- "D:/My_Project_ST/ExperimentHub_cache"
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
#------------------------------------------------------------------------------
# Step1: Load libraries
#------------------------------------------------------------------------------
library(SRTsim)
library(spatialLIBD)
library(SpatialExperiment)
library(tidyverse)
library(dbscan)
library(ape)
library(ExperimentHub)
library(gridExtra)
ExperimentHub::setExperimentHubOption("CACHE", cache_dir)
#------------------------------------------------------------------------------
# Step2: Load DLPFC reference from LOCAL file
#------------------------------------------------------------------------------
load(file.path(output_root, "DLPFC_12data.Rdata"))
obj_names <- ls()
spe_obj <- NULL
for (nm in obj_names) {
  obj <- get(nm)
  if (inherits(obj, "SpatialExperiment") ||
      inherits(obj, "SingleCellExperiment") ||
      inherits(obj, "SummarizedExperiment")) {
    spe_obj <- obj
    message("Using object: ", nm)
    break
  }
}
if (is.null(spe_obj)) stop("No suitable object found.")
spe_dlpfc <- spe_obj
cd_all <- as.data.frame(colData(spe_dlpfc))
sample_col <- NULL
for (col in c("sample_name", "sample_id", "Sample", "sample")) {
  if (col %in% colnames(cd_all)) { sample_col <- col; break }
}
if (is.null(sample_col)) stop("No sample column found.")
keep <- grepl("151673", cd_all[[sample_col]], ignore.case = TRUE)
spe_ref <- spe_dlpfc[, keep]
cat("Subset to 151673:", ncol(spe_ref), "spots.\n")
spe_ref <- spe_ref[, !is.na(spe_ref$layer_guess)]
ref_counts <- as.matrix(assay(spe_ref, "counts"))
ref_coldata <- colData(spe_ref)
ref_rowdata <- rowData(spe_ref)
#------------------------------------------------------------------------------
# Step2b: Filter low‑expression genes + build gene map
#------------------------------------------------------------------------------
spot_frac_cutoff <- 0.05
gene_nonzero_frac <- rowSums(ref_counts > 0) / ncol(ref_counts)
keep_mask <- gene_nonzero_frac >= spot_frac_cutoff
ref_counts  <- ref_counts[keep_mask, ]
ref_rowdata <- ref_rowdata[keep_mask, ]
sym_col <- NULL
for (col in c("gene_name", "symbol", "SYMBOL", "gene_symbol")) {
  if (col %in% colnames(as.data.frame(ref_rowdata))) { sym_col <- col; break }
}
if (is.null(sym_col)) stop("No gene symbol column found.")
cat("Using symbol column:", sym_col, "\n")
gene_map <- data.frame(
  ensembl_id  = rownames(ref_rowdata),
  gene_symbol = as.character(as.data.frame(ref_rowdata)[[sym_col]]),
  stringsAsFactors = FALSE
)
gene_map <- gene_map[!is.na(gene_map$gene_symbol) & gene_map$gene_symbol != "", ]
gene_map <- gene_map[!duplicated(gene_map$ensembl_id), ]
gene_map <- gene_map[gene_map$ensembl_id %in% rownames(ref_counts), ]

## --------------------------
## Keep your original gene sampling setting: 1800 genes (unchanged from your first script)
## --------------------------
n_genes_target <- 1800
all_gene_ids <- rownames(ref_counts)
set.seed(123)
keep_gene_ids <- sample(all_gene_ids, size = n_genes_target, replace = FALSE)
ref_counts  <- ref_counts[keep_gene_ids, ]
ref_rowdata <- ref_rowdata[keep_gene_ids, ]
gene_map    <- gene_map[gene_map$ensembl_id %in% keep_gene_ids, ]
cat("After filter + sampling:", nrow(ref_counts), "genes\n")

## --------------------------
## MODIFIED: use REAL original DLPFC‑151673 spatial coordinates (3611 spots), NO synthetic random spots
## --------------------------
cd <- as.data.frame(colData(spe_ref))
if ("col" %in% colnames(cd) && "row" %in% colnames(cd)) {
  spatial_coords <- as.matrix(cd[, c("col", "row")]); colnames(spatial_coords) <- c("x", "y")
} else if ("imagecol" %in% colnames(cd) && "imagerow" %in% colnames(cd)) {
  spatial_coords <- as.matrix(cd[, c("imagecol", "imagerow")]); colnames(spatial_coords) <- c("x", "y")
} else { stop("No coordinate columns found.") }
## keep original spot IDs from spe_ref
rownames(spatial_coords) <- rownames(cd)
## ref_counts column names must exactly match real spot IDs
colnames(ref_counts) <- rownames(cd)
cat("Real DLPFC‑151673 coords dim:", dim(spatial_coords), "\n")
cat("Number of spots =", nrow(spatial_coords), "\n")
#------------------------------------------------------------------------------
# Step4: Scenarios + helper (completely unchanged from your first script)
#------------------------------------------------------------------------------
generate_domains <- function(coords_mat, n_domains, seed) {
  set.seed(seed)
  km <- kmeans(coords_mat, centers = n_domains, nstart = 30)
  raw_labels <- km$cluster
  knn <- dbscan::kNN(coords_mat, k = 18)
  smoothed_labels <- vapply(
    X = seq_len(nrow(coords_mat)),
    FUN = function(i) {
      neigh_idx <- knn$id[i, ]
      tab <- table(raw_labels[c(i, neigh_idx)])
      as.integer(names(tab)[which.max(tab)])
    },
    FUN.VALUE = integer(1)
  )
  return(factor(smoothed_labels))
}
# Scenario grid: 12 simulation datasets, true SVG proportion fixed at 0.30
domain_vector <- c(5,6,7,8,9,10, 5,6,7,8,9,10)
snr_vector    <- c(rep("high",6), rep("moderate",6))
scenario_grid <- data.frame(
  n_domains   = domain_vector,
  svg_prop    = rep(0.30, 12),
  snr_level   = snr_vector,
  stringsAsFactors = FALSE
)
scenario_grid$scenario_id <- paste0("sim_", sprintf("%02d", 1:nrow(scenario_grid)))
scenario_grid$seed <- 1000 + 1:nrow(scenario_grid)
print(scenario_grid)
#------------------------------------------------------------------------------
# HIGH‑STRENGTH amplification for target Moran I: 0.35‑0.52
# multiplicative gain + additive offset, hard clamp maximum count
#------------------------------------------------------------------------------
amplify_spatial_signal <- function(count_matrix, gene_types, domain_labels, ampl_factor = 9.0, mult_gain = 2.8) {
  svg_gene_idx <- which(gene_types != "none")
  if (length(svg_gene_idx) == 0) return(count_matrix)
  
  domains <- as.character(domain_labels)
  unique_domains <- unique(domains)
  
  for (g in svg_gene_idx) {
    gene_expr <- as.numeric(count_matrix[g, ])
    global_mean <- mean(gene_expr, na.rm = TRUE)
    if (global_mean < 0.2) next
    
    domain_means <- tapply(gene_expr, domains, mean, na.rm = TRUE)
    domain_effect <- domain_means - global_mean
    
    scaled_effect <- (domain_effect / global_mean) * ampl_factor * global_mean
    for (d in names(scaled_effect)) {
      spot_mask <- domains == d
      if(scaled_effect[d] > 0){
        gene_expr[spot_mask] <- gene_expr[spot_mask] * mult_gain
      }
      gene_expr[spot_mask] <- gene_expr[spot_mask] + scaled_effect[d]
    }
    
    gene_expr <- pmax(0, round(gene_expr))
    gene_expr[gene_expr > 1200] <- 1200
    count_matrix[g, ] <- gene_expr
  }
  return(count_matrix)
}
#------------------------------------------------------------------------------
# Step5: Main loop
#------------------------------------------------------------------------------
out_dir <- file.path(output_root, "SRTsim_12_datasets")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
qc_summary <- list()
for (i in 1:nrow(scenario_grid)) {
  scen <- scenario_grid[i, ]
  sid  <- scen$scenario_id
  cat("\n=== Generating", sid, "| domains =", scen$n_domains,
      "| svg_prop =", scen$svg_prop, "| snr =", scen$snr_level, "===\n")
  
  true_domains <- generate_domains(spatial_coords, scen$n_domains, scen$seed)
  
  loc_in <- data.frame(x = spatial_coords[, "x"], y = spatial_coords[, "y"], label = true_domains)
  rownames(loc_in) <- rownames(spatial_coords)
  sim_base <- createSRT(count_in = ref_counts, loc_in = loc_in)
  
  if (scen$snr_level == "high") {
    gene_type_probs <- c("none"=1 - scen$svg_prop, "both"=scen$svg_prop*0.85,
                         "plus"=scen$svg_prop*0.075, "minus"=scen$svg_prop*0.075)
    ampl_factor <- 9.0
    mult_gain   <- 2.8
  } else {
    gene_type_probs <- c("none"=1 - scen$svg_prop, "both"=scen$svg_prop*0.70,
                         "plus"=scen$svg_prop*0.15, "minus"=scen$svg_prop*0.15)
    ampl_factor <- 7.2
    mult_gain   <- 2.2
  }
  probs <- gene_type_probs / sum(gene_type_probs)
  
  num_genes <- nrow(ref_counts)
  set.seed(scen$seed)
  gene_types <- sample(c("none","both","plus","minus"), size = num_genes, replace = TRUE, prob = probs)
  
  rd <- ref_rowdata
  rd$gene_type <- gene_types
  slot(sim_base, "refrowData") <- rd
  
  sim_base <- srtsim_fit(sim_base, sim_scheme = "domain")
  
  set.seed(scen$seed)
  sim_obj <- srtsim_count(simsrt = sim_base, breaktie = "random",
                          nn_num = 5, nn_func = "mean", numCores = 1, verbose = FALSE)
  
  sim_counts <- as.matrix(simCounts(sim_obj))
  gt_gene_type <- refrowData(sim_obj)$gene_type
  if (is.null(gt_gene_type)) gt_gene_type <- gene_types
  
  sim_counts <- amplify_spatial_signal(
    count_matrix = sim_counts,
    gene_types   = gt_gene_type,
    domain_labels = true_domains,
    ampl_factor  = ampl_factor,
    mult_gain    = mult_gain
  )
  
  true_svg_genes <- rownames(sim_counts)[gt_gene_type != "none"]
  
  true_domain_df <- data.frame(spot_id = colnames(sim_counts), true_domain = as.character(true_domains))
  coords_df <- as.data.frame(spatial_coords)
  coords_df$spot_id <- rownames(coords_df)
  coords_df <- coords_df[, c("spot_id", "x", "y")]
  ds_dir <- file.path(out_dir, sid)
  dir.create(ds_dir, showWarnings = FALSE)
  
  saveRDS(sim_counts, file.path(ds_dir, "counts.rds"))
  write.csv(sim_counts, file.path(ds_dir, "counts.csv"), row.names = TRUE)
  write.csv(coords_df, file.path(ds_dir, "spatial_coords.csv"), row.names = FALSE)
  write.csv(true_domain_df, file.path(ds_dir, "true_domains.csv"), row.names = FALSE)
  
  gene_gt_df <- data.frame(
    ensembl_id = rownames(sim_counts), gene_type = gt_gene_type,
    is_true_svg = gt_gene_type != "none", stringsAsFactors = FALSE
  )
  gene_gt_symbol_df <- merge(gene_gt_df, gene_map, by = "ensembl_id", all.x = TRUE)
  gene_gt_symbol_df <- gene_gt_symbol_df[, c("gene_symbol","is_true_svg","ensembl_id","gene_type")]
  write.csv(gene_gt_symbol_df, file.path(ds_dir, "gene_ground_truth_symbol.csv"), row.names = FALSE)
  
  true_svg_symbols <- gene_gt_symbol_df$gene_symbol[gene_gt_symbol_df$is_true_svg == TRUE]
  writeLines(true_svg_symbols, file.path(ds_dir, "true_svg_genes_symbol.txt"))
  write.csv(gene_gt_df, file.path(ds_dir, "gene_ground_truth.csv"), row.names = FALSE)
  writeLines(true_svg_genes, file.path(ds_dir, "true_svg_genes.txt"))
  
  meta_df <- data.frame(
    scenario_id = sid, n_domains = scen$n_domains, svg_prop = scen$svg_prop,
    snr_level = scen$snr_level, ampl_factor = ampl_factor, mult_gain = mult_gain,
    note = "Target True‑SVG median Moran I:0.35‑0.52; n_spots=3611 (real DLPFC‑151673 spots), total_genes=1800, true SVG=30%",
    seed = scen$seed, n_spots = ncol(sim_counts), n_genes = nrow(sim_counts),
    n_true_svgs = length(true_svg_symbols),
    true_svg_pct = round(length(true_svg_symbols)/nrow(sim_counts)*100, 2)
  )
  write.csv(meta_df, file.path(ds_dir, "metadata.csv"), row.names = FALSE)
  qc_summary[[sid]] <- meta_df
  
  cat("  -> Saved", ds_dir, "| spots =", ncol(sim_counts),
      "| genes =", nrow(sim_counts), "| true SVGs =", length(true_svg_symbols), "\n")
}
qc_all <- do.call(rbind, qc_summary)
n_generated <- nrow(qc_all)
write.csv(qc_all, file.path(out_dir, paste0("ALL_",n_generated,"_datasets_QC_summary.csv")), row.names = FALSE)
print(qc_all)
cat("\n=== All", n_generated, "datasets generated successfully. ===\n")

#------------------------------------------------------------------------------
# Step6: QC & Visual inspection on one example dataset (uses SYMBOL files)
#------------------------------------------------------------------------------

simdata <- c("sim_01","sim_02","sim_03","sim_04","sim_05","sim_06",
             "sim_07","sim_08","sim_09","sim_10","sim_11","sim_12")

# Loop over each simulation dataset
for(sim in simdata){
  
  example_dir <- file.path(out_dir, sim)
  
  # Check whether dataset folder exists
  if (file.exists(file.path(example_dir, "counts.rds"))) {
    
    cat("\n=====================================\n")
    cat("Processing dataset: ", sim, "\n")
    cat("=====================================\n")
    
    # Load files
    counts   <- readRDS(file.path(example_dir, "counts.rds"))
    coords   <- read.csv(file.path(example_dir, "spatial_coords.csv"))
    domains  <- read.csv(file.path(example_dir, "true_domains.csv"))
    gt_sym   <- read.csv(file.path(example_dir, "gene_ground_truth_symbol.csv"))
    meta     <- read.csv(file.path(example_dir, "metadata.csv"))
    
    cat("Dataset:", meta$scenario_id,
        "| spots:", meta$n_spots,
        "| genes:", meta$n_genes,
        "| true SVGs:", meta$n_true_svgs, "\n")
    
    # Sanity check: TRUE count match metadata
    cat("Sanity check — CSV TRUE count:", sum(gt_sym$is_true_svg),
        "| metadata n_true_svgs:", meta$n_true_svgs, "\n")
    
    true_svgs_df <- gt_sym[gt_sym$is_true_svg == TRUE, ]
    example_gene_ensembl <- true_svgs_df$ensembl_id[1]
    example_gene_symbol  <- true_svgs_df$gene_symbol[1]
    
    # Plot true spatial domains
    plot_df_domains <- merge(coords, domains, by = "spot_id")
    plot_df_domains$true_domain <- as.factor(plot_df_domains$true_domain)
    
    p_domains <- ggplot() +
      geom_point(data = plot_df_domains,
                 aes(x = x, y = y, color = true_domain), size = 1.5) +
      scale_color_brewer(palette = "Set1") +
      ggtitle(paste0(meta$scenario_id, " : True spatial domains (k=", meta$n_domains, ")")) +
      theme_bw() + coord_fixed()
    
    gene_expr <- data.frame(
      spot_id = colnames(counts),
      expression = counts[example_gene_ensembl, ]
    )
    
    # Plot example SVG gene
    p_gene <- ggplot() +
      geom_point(data = merge(coords, gene_expr, by = "spot_id"),
                 aes(x = x, y = y, color = expression), size = 1.5) +
      scale_color_viridis_c() +
      ggtitle(paste0("Example true SVG: ", example_gene_symbol)) +
      theme_bw() + coord_fixed()
    
    p_combined <- gridExtra::grid.arrange(p_domains, p_gene, ncol = 2)
    
    ggsave(file.path(example_dir, paste0("QC_spatial_domains_and_SVG_",sim,".png")),
           p_combined, width = 14, height = 6)
    
    # Moran's I calculation
    d_mat <- as.matrix(dist(coords[, c("x", "y")]))
    w_mat <- 1 / (d_mat + 1)
    diag(w_mat) <- 0
    
    set.seed(42)
    sample_svgs_df <- true_svgs_df[sample(nrow(true_svgs_df), min(20, nrow(true_svgs_df))), ]
    non_svgs_df <- gt_sym[gt_sym$is_true_svg == FALSE, ]
    sample_nonsvgs_df <- non_svgs_df[sample(nrow(non_svgs_df), min(20, nrow(non_svgs_df))), ]
    
    safe_moran <- function(expr_vec, weight_mat) {
      res <- tryCatch({
        out <- ape::Moran.I(expr_vec, weight_mat)
        out$observed
      }, error = function(e) {
        return(NA_real_)
      })
      return(res)
    }
    
    moran_svg <- sapply(sample_svgs_df$ensembl_id, function(g) safe_moran(counts[g, ], w_mat))
    moran_nonsvg <- sapply(sample_nonsvgs_df$ensembl_id, function(g) safe_moran(counts[g, ], w_mat))
    
    moran_df <- data.frame(
      moran_I = c(moran_svg, moran_nonsvg),
      group = c(rep("True SVG", length(moran_svg)), rep("Non‑SVG", length(moran_nonsvg)))
    )
    moran_df <- moran_df[!is.na(moran_df$moran_I), ]
    
    p_moran <- ggplot(moran_df, aes(x = group, y = moran_I, fill = group)) +
      geom_boxplot() + geom_jitter(width = 0.1) +
      ggtitle(paste0("Moran's I check: ", sim)) +
      theme_bw()
    
    ggsave(file.path(example_dir, paste0("QC_moran_I_check_",sim,".png")),
           p_moran, width = 6, height = 5)
    
    cat("\nMoran's I summary:\n")
    print(tapply(moran_df$moran_I, moran_df$group, summary))
    
  } else {
    cat("\nSkip", sim, ": folder or counts.rds not found\n")
  }
}

cat("\n=== All dataset QC finished ===\n")