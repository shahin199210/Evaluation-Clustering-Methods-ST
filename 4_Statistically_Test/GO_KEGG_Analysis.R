# ---- GO / KEGG Enrichment ----
library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
library(enrichplot)
library(ggplot2)
library(stringr) # ADDED for text wrap/truncate

# ---- FIX: Increase timeout and set download method ----
options(timeout = 99999)
options(clusterProfiler.download.method = "auto")
# If "auto" fails, try: options(clusterProfiler.download.method = "wininet")
# --------------------------------------------------------
output_dir <- "D:/My_Project_ST/benchmarking of ST platforms/SVGs_Clustering_Method/All_Scores_Cluster_Methods/GO_KEGG"
load_data  <- "D:/My_Project_ST/benchmarking of ST platforms/SVGs_Clustering_Method/All_Scores_Cluster_Methods/GO_KEGG/Top_Common_SVGs"
top_70 <- read.csv(file.path(load_data, "Top_70_SVGs.csv"))$gene_name
# Convert gene symbols to Entrez IDs
gene_df <- bitr(top_70, fromType = "SYMBOL",
                toType   = "ENTREZID",
                OrgDb    = org.Hs.eg.db)
entrez_ids <- gene_df$ENTREZID
# ---- GO enrichment (BP, MF, CC) ----
ego_bp <- enrichGO(gene          = entrez_ids,
                   OrgDb         = org.Hs.eg.db,
                   ont           = "BP",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.05,
                   qvalueCutoff  = 0.05,
                   readable      = TRUE,
                   minGSSize     = 3)
ego_mf <- enrichGO(gene          = entrez_ids,
                   OrgDb         = org.Hs.eg.db,
                   ont           = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.05,
                   qvalueCutoff  = 0.05,
                   readable      = TRUE,
                   minGSSize     = 3)
ego_cc <- enrichGO(gene          = entrez_ids,
                   OrgDb         = org.Hs.eg.db,
                   ont           = "CC",
                   pAdjustMethod = "BH",
                   qvalueCutoff  = 0.05,
                   readable      = TRUE,
                   minGSSize     = 3)
# ---- KEGG enrichment (with timeout fix) ----
kk <- enrichKEGG(gene          = entrez_ids,
                 organism      = "hsa",
                 pvalueCutoff  = 0.05,
                 pAdjustMethod = "BH",
                 minGSSize     = 3)
# ---- Save results ----
#write.csv(as.data.frame(ego_bp), file.path(output_dir, "GO_BP_results.csv"), row.names = FALSE)
#write.csv(as.data.frame(ego_mf), file.path(output_dir, "GO_MF_results.csv"), row.names = FALSE)
#write.csv(as.data.frame(ego_cc), file.path(output_dir, "GO_CC_results.csv"), row.names = FALSE)
#write.csv(as.data.frame(kk),     file.path(output_dir, "KEGG_results.csv"),     row.names = FALSE)
# ---- Summary ----
cat("=== GO / KEGG Enrichment Summary ===\n")
cat("GO BP significant terms:", nrow(as.data.frame(ego_bp)), "\n")
cat("GO MF significant terms:", nrow(as.data.frame(ego_mf)), "\n")
cat("GO CC significant terms:", nrow(as.data.frame(ego_cc)), "\n")
cat("KEGG significant terms :", nrow(as.data.frame(kk)),     "\n")
# ---- Top enriched terms ----
cat("\nTop 10 GO BP terms:\n")
print(head(as.data.frame(ego_bp)[, c("Description", "p.adjust", "Count")], 10))

# ==============================
# NEW CUSTOM LABEL FUNCTION: wrap max 2 lines, add ... if exceed two lines
# ==============================
label_2line_ellipsis <- function(x) {
  sapply(x, function(txt) {
    wrapped <- str_wrap(txt, width = 32)
    lines <- str_split(wrapped, "\n")[[1]]
    if (length(lines) >2) {
      combined_two <- paste(lines[1:2], collapse = "\n")
      combined_two <- str_trunc(combined_two, width = 42, ellipsis = "...")
      return(combined_two)
    } else {
      return(wrapped)
    }
  })
}

# ==============================
# PLOTTING BLOCK: DOTPLOTS (BUBBLE PLOTS)
# ==============================
# Set global large font theme for ALL plots
theme_large_font <- theme_bw() +
  theme(
    plot.title      = element_text(size = 32, face = "bold", colour = "black", family = "Times New Roman", hjust = 0.25, margin = margin(b = 1)),
    axis.title.x    = element_text(size = 28, colour = "black", family = "Times New Roman"),
    axis.title.y    = element_text(size = 28, colour = "black", family = "Times New Roman"),
    axis.text.x     = element_text(size = 28, colour = "black", family = "Times New Roman"),
    axis.text.y     = element_text(size = 28, colour = "black", family = "Times New Roman"),
    legend.title    = element_text(size = 26, colour = "black", family = "Times New Roman"),
    legend.text     = element_text(size = 24, colour = "black", family = "Times New Roman")
  )
# showCategory = 10: show top 10 most significant terms for each
# Use label_format to apply custom 2-line+ellipsis formatter
p_bp <- dotplot(ego_bp, showCategory = 5, title = "GO Biological Process", label_format = label_2line_ellipsis) + theme_large_font
 

p_mf <- dotplot(ego_mf, showCategory = 5, title = "GO Molecular Function", label_format = label_2line_ellipsis) + theme_large_font
p_cc <- dotplot(ego_cc, showCategory = 5, title = "GO Cellular Component", label_format = label_2line_ellipsis) + theme_large_font
p_kegg <- dotplot(kk, showCategory = 5, title = "KEGG Pathway", label_format = label_2line_ellipsis) + theme_large_font

# Short X-axis GeneRatio tick labels (optional cleaner ratio formatting)
p_bp <- p_bp + scale_x_continuous(labels = scales::label_number(accuracy = 0.01))
p_mf <- p_mf + scale_x_continuous(labels = scales::label_number(accuracy = 0.01))
p_cc <- p_cc + scale_x_continuous(labels = scales::label_number(accuracy = 0.01))
p_kegg <- p_kegg + scale_x_continuous(labels = scales::label_number(accuracy = 0.01))

# Save individual high-resolution figures
#ggsave(file.path(output_dir, "GO_BP_dotplot.pdf"), p_bp, width = 10, height = 7, dpi = 300)
#ggsave(file.path(output_dir, "GO_MF_dotplot.pdf"), p_mf, width = 10, height = 7, dpi = 300)
#ggsave(file.path(output_dir, "GO_CC_dotplot.pdf"), p_cc, width = 10, height = 7, dpi = 300)
#ggsave(file.path(output_dir, "KEGG_dotplot.pdf"), p_kegg, width = 10, height = 7, dpi = 300)
# Optional: combine all 4 plots into ONE single figure (2×2 panel for manuscript)
library(patchwork)
p_combined <- (p_bp + p_mf) / (p_cc + p_kegg) +
  plot_layout(widths = c(0.75,0.75))
#ggsave(file.path(output_dir, "GO_KEGG_combined_2x2.pdf"), p_combined, width = 18, height = 10, dpi = 300)
ggsave(file.path(output_dir, "GO_KEGG_combined_2x2.png"), p_combined, width = 18, height = 12, dpi = 300)
cat("\nFigures saved to output folder!\n")