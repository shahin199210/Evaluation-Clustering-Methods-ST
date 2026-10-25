# ============================================================
library(tidyverse)
library(readxl)
# ---- ADDED: Adjustable font sizes for Plot 1 (F1 boxplot) ----
TITLE_SIZE          <- 36   # <-- INCREASED (was 20)
X_AXIS_TITLE_SIZE   <- 36   # <-- INCREASED (was 19)
Y_AXIS_TITLE_SIZE   <- 36   # <-- INCREASED (was 19)
X_AXIS_TICK_SIZE    <- 28   # <-- INCREASED (was 18)
Y_AXIS_TICK_SIZE    <- 28   # <-- INCREASED (was 18)
# -------------------------------------------------------------
# ---- ADDED: Adjustable sizes for Plot 4 (Aggregated Sens vs Spec) ----
AGG_FIG_WIDTH       <- 13   # <-- INCREASED (was 10)
AGG_FIG_HEIGHT      <- 10   # <-- INCREASED (was 7)
AGG_TITLE_SIZE      <- 36   # <-- INCREASED
AGG_AXIS_TITLE_SIZE <- 30   # <-- INCREASED
AGG_AXIS_TEXT_SIZE  <- 30   # <-- INCREASED
AGG_LEGEND_SIZE     <- 28   # <-- INCREASED
AGG_LEGEND_SPACING  <- 4  # <-- CHANGED to 1.5 (6.0 is very large, but you can change it back)
# ----------------------------------------------------------------------
# ============================================================
#  STEP 0: SET FOLDER PATHS
# ============================================================
input_dir     <- "D:/My_Project_ST/benchmarking of ST platforms/SVGs_Clustering_Method/All_Scores_Cluster_Methods"
consensus_dir <- "D:/My_Project_ST/benchmarking of ST platforms/Global_SVGs_Methods/All_Scores_Global_Methods/Consensus_BarPlot"
output_dir    <- "D:/My_Project_ST/benchmarking of ST platforms/SVGs_Clustering_Method/All_Scores_Cluster_Methods/Accuracy_Check"
if (!dir.exists(input_dir)) stop("Clustering folder NOT found: ", input_dir)
if (!dir.exists(consensus_dir)) stop("Consensus folder NOT found: ", consensus_dir)
clustering_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(clustering_files) == 0) stop("No .csv files in clustering folder")
consensus_files <- list.files(consensus_dir, pattern = "consensus.*\\.csv$", full.names = TRUE)
if (length(consensus_files) == 0) stop("No consensus .csv files found in:\n", consensus_dir)
cat("Clustering files:", length(clustering_files), "\n")
cat("Consensus files :", length(consensus_files), "\n")
print(basename(consensus_files))
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
# ============================================================
#  27-METHOD KNOWN ORDER (for consistent plotting)
# ============================================================
KNOWN_ORDER <- c(
  "CCST", "conST", "DeepGFT", "DeepSpaCE", "DSTN", "GraphST",
  "IRIS", "iSpatial", "Leiden", "Louvain", "SCGDL", "SEDR",
  "SeuratV6", "SGAE", "SpaGCN", "SpaNCMG", "SpaRSE", "SpaTalk",
  "SpaceFlow", "SpatialMGCN", "STAGATE", "STAlign", "STAligner",
  "stCluster", "stFormer", "STMGCN", "stRGAT"
)
# ============================================================
#  STEP 1: Read clustering method results
# ============================================================
all_results <- map_dfr(clustering_files, function(f) {
  base  <- str_remove(basename(f), "\\.csv$")
  parts <- str_split(base, "_", n = 2)[[1]]
  method  <- parts[1]
  dataset <- ifelse(length(parts) > 1, parts[2], "unknown")
  
  df <- read_csv(f, show_col_types = FALSE)
  
  # FIX 1: expanded gene column detection
  gene_candidates <- c("gene_name","gene_id","gene_symbol","gene","Gene","genes","GENE")
  gene_col <- intersect(gene_candidates, colnames(df))[1]
  if (is.na(gene_col)) {
    cat("  SKIP (no gene column):", basename(f), "| cols:", paste(colnames(df), collapse=", "), "\n")
    return(NULL)
  }
  
  # FIX 2: expanded SVG flag column detection
  svg_candidates <- c("is_SVG_FDR0.05","is_SVG","is_SVG_FDR0.01","is_svg","SVG","significant")
  svg_col <- intersect(svg_candidates, colnames(df))[1]
  if (is.na(svg_col)) {
    candidates <- grep("SVG|significant|is_svg", colnames(df), ignore.case = TRUE, value = TRUE)
    if (length(candidates) > 0) {
      svg_col <- candidates[1]
    } else {
      cat("  SKIP (no SVG column):", basename(f), "| cols:", paste(colnames(df), collapse=", "), "\n")
      return(NULL)
    }
  }
  
  # FIX 3: robust TRUE detection (handles "TRUE","True","true","1","T", logical)
  svg_vals <- as.character(df[[svg_col]])
  is_svg_flag <- svg_vals %in% c("TRUE","True","true","1","T") |
    (!is.na(as.logical(svg_vals)) & as.logical(svg_vals))
  
  df %>%
    dplyr::select(gene = all_of(gene_col)) %>%
    mutate(
      Method  = method,
      Dataset = dataset,
      is_SVG_FDR0.05 = is_svg_flag
    )
})
if (nrow(all_results) == 0) stop("No valid clustering files found.")
cat("\nClustering data loaded:\n")
cat("  Methods :", length(unique(all_results$Method)), "\n")
print(unique(all_results$Method))
cat("  Datasets:", length(unique(all_results$Dataset)), "\n")
print(unique(all_results$Dataset))
# ============================================================
#  STEP 2: Load external consensus SVGs from CSV files
# ============================================================
consensus <- map_dfr(consensus_files, function(f) {
  fname <- basename(f)
  dataset <- str_split(fname, "_consensus")[[1]][1]
  
  df <- read_csv(f, show_col_types = FALSE)
  
  gene_candidates <- c("gene_name","gene_id","gene_symbol","gene","Gene","genes","GENE","X__1")
  gene_col <- intersect(gene_candidates, colnames(df))[1]
  if (is.na(gene_col)) {
    gene_col <- colnames(df)[1]
    cat("  Warning: no gene column in", fname, "- using first column:", gene_col, "\n")
  }
  
  df %>%
    dplyr::select(gene = all_of(gene_col)) %>%
    mutate(Dataset = dataset, is_consensus = TRUE)
})
cat("\nConsensus SVGs loaded:\n")
consensus %>% count(Dataset, name = "n_consensus_SVGs") %>% print()
# ============================================================
#  STEP 3: Compute accuracy metrics per Method x Dataset
# ============================================================
metrics <- map_dfr(unique(all_results$Dataset), function(ds) {
  all_genes <- all_results %>% filter(Dataset == ds) %>% pull(gene) %>% unique()
  true_svg  <- consensus %>% filter(Dataset == ds) %>% pull(gene) %>% unique()
  
  if (length(true_svg) == 0) {
    cat("  WARNING: no consensus SVGs for dataset", ds, "- skipping\n")
    return(NULL)
  }
  
  map_dfr(unique(all_results$Method), function(mt) {
    pred_svg <- all_results %>%
      filter(Dataset == ds, Method == mt, is_SVG_FDR0.05 == TRUE) %>%
      pull(gene) %>% unique()
    
    TP <- sum(pred_svg %in% true_svg)
    FP <- length(setdiff(pred_svg, true_svg))
    FN <- length(setdiff(true_svg, pred_svg))
    TN <- length(setdiff(all_genes, union(pred_svg, true_svg)))
    total <- TP + FP + TN + FN
    
    tibble(
      Dataset = ds, Method = mt,
      n_called = length(pred_svg),
      n_true_consensus = length(true_svg),
      TP, FP, TN, FN,
      Accuracy    = (TP + TN) / total,
      Sensitivity = ifelse(TP + FN > 0, TP / (TP + FN), NA),
      Specificity = ifelse(TN + FP > 0, TN / (TN + FP), NA),
      Precision   = ifelse(TP + FP > 0, TP / (TP + FP), NA),
      F1          = ifelse(TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), NA)
    )
  })
})
write_csv(metrics, file.path(output_dir, "Accuracy_Metrics_per_Method_Dataset.csv"))
# ============================================================
#  STEP 4: Rank methods by mean F1
# ============================================================
ranking <- metrics %>%
  group_by(Method) %>%
  summarise(
    across(c(Accuracy, Sensitivity, Specificity, Precision, F1),
           ~mean(.x, na.rm = TRUE), .names = "mean_{.col}"),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_F1))
write_csv(ranking, file.path(output_dir, "Method_Ranking_by_F1.csv"))
cat("\n=== Method Ranking (by mean F1) ===\n")
print(ranking, n = Inf)
# ============================================================
#  STEP 5: Plots (auto-scaled for 27 methods)
# ============================================================
method_order <- ranking$Method
metrics$Method_f <- factor(metrics$Method, levels = method_order)
n_methods <- length(unique(metrics$Method))
cat("\nPlotting", n_methods, "methods\n")
# --- Plot 1: F1 boxplot ---
p_f1 <- ggplot(metrics, aes(x = Method_f, y = F1, fill = Method_f)) +
  geom_boxplot(width = 0.7, outlier.shape = NA, alpha = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(x = "Clustering Methods", y = "F1-score",
       title = paste0("Clustering method accuracy (F1) vs consensus")) +
  theme_bw(base_size = 16) +
  theme(
    plot.title    = element_text(size = TITLE_SIZE, face = "bold", hjust = 0.5, color = "black"),
    axis.title.x  = element_text(size = X_AXIS_TITLE_SIZE, color = "black", face = "bold"),
    axis.title.y  = element_text(size = Y_AXIS_TITLE_SIZE, color = "black", face = "bold"),
    axis.text.x   = element_text(angle = 45, hjust = 1, size = X_AXIS_TICK_SIZE, color = "black"),
    axis.text.y   = element_text(size = Y_AXIS_TICK_SIZE, color = "black"),
    legend.position = "none"
  )
# --- Plot 2: Sensitivity vs Specificity (raw) ---
p_senspec <- ggplot(metrics, aes(x = Specificity, y = Sensitivity, color = Method)) +
  geom_vline(aes(xintercept = mean(Specificity, na.rm = TRUE)),
             linetype = "dashed", color = "grey50", linewidth = 0.3) +
  geom_hline(aes(yintercept = mean(Sensitivity, na.rm = TRUE)),
             linetype = "dashed", color = "grey50", linewidth = 0.3) +
  geom_point(size = 2.5, alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", size = 4, shape = 18) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(x = "Specificity", y = "Sensitivity", color = "Method") +
  theme_bw(base_size = 12) +
  theme(legend.position = "right", legend.text = element_text(size = 8), legend.spacing.y = unit(AGG_LEGEND_SPACING, "cm"))
# --- Plot 3: F1 heatmap ---
p_heat <- ggplot(metrics, aes(x = Method_f, y = Dataset, fill = F1)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(F1, 2)), size = 3) +
  scale_fill_viridis_c(option = "plasma", limits = c(0, 1)) +
  labs(x = "", y = "", fill = "F1", title = paste0("F1 heatmap (", n_methods, " methods x Dataset)")) +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9))
# --- Plot 4: Aggregated Sensitivity vs Specificity ---
agg_summary <- metrics %>%
  group_by(Method) %>%
  summarise(
    mean_Sensitivity = mean(Sensitivity, na.rm = TRUE),
    sd_Sensitivity   = sd(Sensitivity, na.rm = TRUE),
    mean_Specificity = mean(Specificity, na.rm = TRUE),
    sd_Specificity   = sd(Specificity, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(Method_f = factor(Method, levels = method_order))

# ========= NEW BLOCK: Identify top5 sensitivity and top5 specificity =========
top5_sens <- agg_summary %>% slice_max(mean_Sensitivity, n = 5) %>% pull(Method)
top5_spec <- agg_summary %>% slice_max(mean_Specificity, n = 5) %>% pull(Method)

# Add marked legend label: add * for top5 Sens, † for top5 Spec
agg_summary <- agg_summary %>%
  mutate(
    Method_label = case_when(
      Method %in% intersect(top5_sens, top5_spec) ~ paste0(Method, " *,†"),
      Method %in% top5_sens ~ paste0(Method, " *"),
      Method %in% top5_spec ~ paste0(Method, " †"),
      TRUE ~ Method
    )
  )
# =============================================================================

write_csv(agg_summary, file.path(output_dir, "Aggregated_Sens_Spec_per_Method.csv"))
p_senspec_agg <- ggplot(agg_summary,
                        aes(x = mean_Specificity, y = mean_Sensitivity,
                            color = Method_f)) +
  geom_errorbar(aes(ymin = mean_Sensitivity - sd_Sensitivity,
                    ymax = mean_Sensitivity + sd_Sensitivity),
                width = 0.003, linewidth = 0.4, alpha = 0.7) +
  geom_errorbarh(aes(xmin = mean_Specificity - sd_Specificity,
                     xmax = mean_Specificity + sd_Specificity),
                 height = 0.003, linewidth = 0.4, alpha = 0.7) +
  geom_point(size = 3.5, alpha = 0.9) +
  
  # ---- ADDED: Forces vertical spacing in multi-column legends ----
guides(color = guide_legend(
  byrow = TRUE,
  label.theme = element_text(family = "Times New Roman"),
  label.hjust = 0
)) +
  # ----------------------------------------------------------------
scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(x = "Specificity", y = "Sensitivity",
       color = "Method\n*:Top5 Sensitivity; †:Top5 Specificity", # add footnote in legend title
       title = paste0("Aggregated sensitivity vs specificity")) +
  theme_bw(base_size = 16) +
  theme(
    plot.title    = element_text(size = AGG_TITLE_SIZE, face = "bold", hjust = 0.5, color = "black"),
    plot.title.position = "plot",
    plot.margin = margin(t = 30, r = 10, b = 10, l = 10),
    axis.title.x  = element_text(size = AGG_AXIS_TITLE_SIZE, color = "black", face = "bold"),
    axis.title.y  = element_text(size = AGG_AXIS_TITLE_SIZE, color = "black", face = "bold"),
    axis.text.x   = element_text(size = AGG_AXIS_TEXT_SIZE, color = "black"),
    axis.text.y   = element_text(size = AGG_AXIS_TEXT_SIZE, color = "black"),
    legend.position = "right", 
    # ---- ADDED: Increase vertical space between legend rows ----
    legend.spacing.y = unit(AGG_LEGEND_SPACING, "cm"),
    legend.key.height = unit(1.25, "cm"),  # <-- ADDED: Makes each legend item taller
    # -----------------------------------------------------------
    legend.title = element_text(size = AGG_LEGEND_SIZE, face = "bold", color = "black"),
    legend.text = element_text(size = AGG_LEGEND_SIZE, color = "black")
  ) +
  # Override legend labels with our annotated Method_label
  scale_color_discrete(labels = agg_summary$Method_label)
# ============================================================
#  STEP 6: Save all plots (width auto-scaled for 27 methods)
# ============================================================
fig_width <- max(13, n_methods * 0.8)
ggsave(file.path(output_dir, "F1_boxplot.png"), p_f1, width = fig_width, height = 8, dpi = 300)
ggsave(file.path(output_dir, "Sensitivity_vs_Specificity.png"), p_senspec, width = 9, height = 6, dpi = 300)
ggsave(file.path(output_dir, "F1_heatmap.png"), p_heat, width = fig_width, height = 5, dpi = 300)
ggsave(file.path(output_dir, "Sensitivity_vs_Specificity_Aggregated.png"),
       p_senspec_agg, width = AGG_FIG_WIDTH, height = AGG_FIG_HEIGHT, dpi = 300)
ggsave(file.path(output_dir, "Sensitivity_vs_Specificity_Aggregated.pdf"),
       p_senspec_agg, width = AGG_FIG_WIDTH, height = AGG_FIG_HEIGHT)
cat("\n=== DONE ===\n")
cat("Outputs saved to:\n", output_dir, "\n")
cat("Total methods analyzed:", n_methods, "\n")