import sys
import os
import time
import psutil
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import glob
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

# --------------------------
# Global Basic Configuration
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Global device:', device)

# Import STAGATE framework
import STAGATE_pyG as STAGATE

# --------------------------
# Basic Utility Function
# --------------------------
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# --------------------------
# Unified SVG Calculation Module (Consistent benchmark standard)
# --------------------------
def calculate_STAGATE_SVG(
    adata,
    n_permutation=1000,
    seed=0
):
    """
    Calculate spatial variable genes using STAGATE spatial graph smoothness
    Score: Pearson correlation between raw expression and graph-smoothed expression
    Permutation test + BH-FDR correction
    """
    np.random.seed(seed)
    n_cells, n_genes = adata.shape
    expr_mat = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
    adj_norm = adata.uns['Spatial_Net_Normalized']

    # Compute spatial smooth correlation score for all genes
    spatial_corr_score = np.zeros(n_genes)
    for g in range(n_genes):
        gene_raw = expr_mat[:, g]
        gene_smooth = adj_norm @ gene_raw
        cov = np.cov(gene_raw, gene_smooth)[0, 1]
        var_raw = np.var(gene_raw)
        var_smooth = np.var(gene_smooth)
        if var_raw < 1e-8 or var_smooth < 1e-8:
            spatial_corr_score[g] = 0.0
        else:
            spatial_corr_score[g] = cov / np.sqrt(var_raw * var_smooth)

    # Permutation test to get empirical p-value
    perm_score_mat = np.zeros((n_permutation, n_genes))
    for p in range(n_permutation):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_smooth = adj_norm @ perm_expr
        for g in range(n_genes):
            g_raw = perm_expr[:, g]
            g_smooth = perm_smooth[:, g]
            cov_p = np.cov(g_raw, g_smooth)[0, 1]
            vr = np.var(g_raw)
            vs = np.var(g_smooth)
            if vr < 1e-8 or vs < 1e-8:
                perm_score_mat[p, g] = 0.0
            else:
                perm_score_mat[p, g] = cov_p / np.sqrt(vr * vs)

    raw_pval = np.zeros(n_genes)
    for g in range(n_genes):
        obs_score = spatial_corr_score[g]
        count_extreme = np.sum(np.abs(perm_score_mat[:, g]) >= np.abs(obs_score))
        raw_pval[g] = (count_extreme + 1) / (n_permutation + 1)

    reject, fdr_padj, _, _ = multipletests(raw_pval, alpha=0.05, method='fdr_bh')

    svg_result_df = pd.DataFrame({
        "gene_ids": adata.var.index.values,
        "spatial_correlation_score": spatial_corr_score,
        "raw_pvalue": raw_pval,
        "fdr_bh_adjust_p": fdr_padj,
        "is_SVG_FDR0.05": reject
    })
    svg_result_df = svg_result_df.sort_values("spatial_correlation_score", ascending=False).reset_index(drop=True)
    return svg_result_df

# --------------------------
# STAGATE Graph Construction Function 1: DLPFC Dataset
# --------------------------
def run_STAGATE_Graph_DLPFC(adata_h5, random_seed=0, rad_cutoff=150):
    adata = adata_h5.copy()
    # Standard preprocessing
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Build spatial neighbor graph
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    # Normalize adjacency matrix for SVG calculation
    adj_df = adata.uns['Spatial_Net']
    n_cell = adata.n_obs
    adj_mat = np.zeros((n_cell, n_cell))
    for i, j in zip(adj_df['Cell1'], adj_df['Cell2']):
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
    row_sum = adj_mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1e-8
    adj_norm = adj_mat / row_sum
    adata.uns['Spatial_Net_Normalized'] = adj_norm
    # Train STAGATE embedding
    adata = STAGATE.train_STAGATE(adata, device=device, random_seed=random_seed)
    return adata

# --------------------------
# STAGATE Graph Construction Function 2: General All Datasets (FULL FIX)
# --------------------------
def run_STAGATE_Graph_General(adata_h5, random_seed=0, rad_cutoff=50):
    adata = adata_h5.copy()
    n_total = adata.n_obs
    target_min, target_max = 1000, 1200
    # Dynamic downsampling to avoid OOM
    if n_total > target_max:
        target_cell = (target_min + target_max) // 2
        frac = target_cell / n_total
        print(f"Memory optimization: downsample {n_total} cells -> {target_cell} cells")
        adata = adata[adata.obs.sample(frac=frac, random_state=42).index].copy()

    # Fix spatial coordinate format
    if 'spatial' not in adata.obsm or adata.obsm['spatial'].shape[1] != 2:
        if 'spatial' in adata.obsm:
            adata.obsm['spatial'] = adata.obsm['spatial'][:, :2]
        else:
            n = adata.n_obs
            side = int(np.ceil(np.sqrt(n)))
            xx, yy = np.meshgrid(np.arange(side), np.arange(side))
            dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
            adata.obsm['spatial'] = dummy_coords[:n]
    # Lightweight preprocessing
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1500)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Build spatial graph on DOWN-SAMPLED adata
    print("------Calculating spatial graph...")
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    adj_df = adata.uns['Spatial_Net']
    n_cell = adata.n_obs
    adj_mat = np.zeros((n_cell, n_cell))

    # Core Fix: Map string cell labels to integer positional indices
    cell_pos_map = {cell_name: idx for idx, cell_name in enumerate(adata.obs.index)}
    for cell1_name, cell2_name in zip(adj_df['Cell1'], adj_df['Cell2']):
        i = cell_pos_map[cell1_name]
        j = cell_pos_map[cell2_name]
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    row_sum = adj_mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1e-8
    adj_norm = adj_mat / row_sum
    adata.uns['Spatial_Net_Normalized'] = adj_norm
    # Train embedding on CPU to avoid GPU OOM
    adata = STAGATE.train_STAGATE(adata, device=torch.device("cpu"), random_seed=random_seed)
    return adata

# --------------------------
# Main Execution Pipeline
# --------------------------
if __name__ == '__main__':
    # Fixed file paths
    DATA_ROOT = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
    OUTPUT_ROOT = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\STAGATE"
    mk_dir(OUTPUT_ROOT)

    # Auto classify sequencing platform from filename prefix
    def get_platform_tag(filename: str):
        fn = filename.lower()
        if fn.startswith("10x") or fn.startswith("10_x"):
            return "10X Visium"
        elif fn.startswith("slide-seq"):
            return "Slide-seq"
        elif fn.startswith("st_"):
            return "ST"
        elif fn.startswith("starmap"):
            return "STARmap"
        elif fn.startswith("xenium"):
            return "Xenium"
        elif fn.startswith("15"):
            return "DLPFC"
        else:
            return "Unknown"

    # Scan all standardized h5ad files
    h5ad_file_list = glob.glob(os.path.join(DATA_ROOT, "*_standardized.h5ad"))
    if len(h5ad_file_list) == 0:
        raise FileNotFoundError(f"No standardized h5ad files found in {DATA_ROOT}")

    global_summary_list = []

    # Iterate each dataset
    for file_path in h5ad_file_list:
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace("_standardized.h5ad", "")
        platform = get_platform_tag(file_name)
        dataset_save_dir = mk_dir(os.path.join(OUTPUT_ROOT, dataset_name))

        print(f"\n==================================================")
        print(f"STAGATE SVG Detection | Dataset: {dataset_name} | Platform: {platform}")
        print(f"File Path: {file_path}")
        print(f"==================================================")

        # Load standardized adata
        adata_raw = sc.read_h5ad(file_path)
        adata_raw.var_names_make_unique()

        # Initialize time & memory monitor
        tracemalloc.start()
        start_time = time.time()
        start_rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

        # Select graph construction function
        if dataset_name.startswith("15"):
            adata_processed = run_STAGATE_Graph_DLPFC(adata_raw, random_seed=0, rad_cutoff=150)
        else:
            adata_processed = run_STAGATE_Graph_General(adata_raw, random_seed=0, rad_cutoff=50)

        # Calculate SVG
        svg_df = calculate_STAGATE_SVG(adata_processed, n_permutation=1000, seed=0)

        # End resource statistics
        run_seconds = time.time() - start_time
        end_rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        mem_increase_gb = end_rss_gb - start_rss_gb
        _, peak_byte = tracemalloc.get_traced_memory()
        peak_gb = peak_byte / (1024 ** 3)
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Basic statistics
        total_gene = adata_processed.n_vars
        sig_svg_count = int(svg_df["is_SVG_FDR0.05"].sum())
        top10_svg = svg_df.head(10)["gene_ids"].tolist()

        # Append summary record
        record = {
            "platform": platform,
            "dataset": dataset_name,
            "total_gene_number": total_gene,
            "significant_SVG_FDR0.05": sig_svg_count,
            "runtime_seconds": round(run_seconds, 2),
            "ram_increment_GB": round(mem_increase_gb, 3),
            "peak_memory_GB": round(peak_gb, 3)
        }
        global_summary_list.append(record)

        # Export all SVG result files
        full_csv = os.path.join(dataset_save_dir, f"{dataset_name}_full_SVG_table.csv")
        svg_df.to_csv(full_csv, index=False)

        sig_csv = os.path.join(dataset_save_dir, f"{dataset_name}_significant_SVG_FDR0.05.csv")
        svg_df[svg_df["is_SVG_FDR0.05"]].to_csv(sig_csv, index=False)

        top50_csv = os.path.join(dataset_save_dir, f"{dataset_name}_Top50_SVG.csv")
        svg_df.head(50).to_csv(top50_csv, index=False)

        # Save processed adata, embed SVG results into adata.uns
        adata_processed.uns["STAGATE_SVG_Result"] = svg_df
        adata_processed.write_h5ad(os.path.join(dataset_save_dir, f"{dataset_name}_processed.h5ad"))

        # Console print log
        print(f"✅ Completed {dataset_name}:")
        print(f"Total Genes: {total_gene} | Significant SVG(FDR<0.05): {sig_svg_count}")
        print(f"Top10 Spatial Variable Genes: {top10_svg}")
        print(f"Runtime: {run_seconds:.2f}s | Peak Memory: {peak_gb:.3f}GB\n")

    # Save global benchmark summary table
    summary_df = pd.DataFrame(global_summary_list)
    summary_path = os.path.join(OUTPUT_ROOT, "STAGATE_Global_SVG_Benchmark_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("==================== ALL DATASETS FINISHED ====================")
    print(f"Global summary saved to: {summary_path}")
    print("Each dataset folder contains: full SVG table, significant SVG table, Top50 SVG table, processed h5ad file")