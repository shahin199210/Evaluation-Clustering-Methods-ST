import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import psutil, tracemalloc
import time
import warnings
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import glob

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# ===================== Unified SVG Calculation Function (Consistent with GraphST logic) =====================
def calculate_SpaGCN_SVG(
    adata,
    adj_matrix: np.ndarray,
    n_permutation=1000,
    seed=0
):
    np.random.seed(seed)
    n_cells, n_genes = adata.shape
    expr_mat = adata.X.A if issparse(adata.X) else adata.X.copy()
    # Symmetric normalization of adjacency matrix
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    row_sum = adj_matrix.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1e-8
    adj_norm = adj_matrix / row_sum

    graph_spatial_score = np.zeros(n_genes)
    for g in range(n_genes):
        gene_expr = expr_mat[:, g]
        smooth_expr = adj_norm @ gene_expr
        cov = np.cov(gene_expr, smooth_expr)[0, 1]
        var_raw = np.var(gene_expr)
        var_smooth = np.var(smooth_expr)
        if var_raw < 1e-8 or var_smooth < 1e-8:
            graph_spatial_score[g] = 0.0
        else:
            graph_spatial_score[g] = cov / np.sqrt(var_raw * var_smooth)

    # Permutation test
    perm_scores = np.zeros((n_permutation, n_genes))
    for p in range(n_permutation):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_smooth = adj_norm @ perm_expr
        for g in range(n_genes):
            g_expr = perm_expr[:, g]
            g_smooth = perm_smooth[:, g]
            cov_p = np.cov(g_expr, g_smooth)[0, 1]
            vr = np.var(g_expr)
            vs = np.var(g_smooth)
            if vr < 1e-8 or vs < 1e-8:
                perm_scores[p, g] = 0.0
            else:
                perm_scores[p, g] = cov_p / np.sqrt(vr * vs)

    raw_pval = np.zeros(n_genes)
    for g in range(n_genes):
        obs_s = graph_spatial_score[g]
        cnt = np.sum(np.abs(perm_scores[:, g]) >= np.abs(obs_s))
        raw_pval[g] = (cnt + 1) / (n_permutation + 1)

    # FIX: Change method='bh' to method='fdr_bh' (valid official parameter)
    reject, fdr_padj, _, _ = multipletests(raw_pval, alpha=0.05, method='fdr_bh')

    svg_df = pd.DataFrame({
        "gene_ids": adata.var.index.values,
        "spatial_correlation_score": graph_spatial_score,
        "raw_pvalue": raw_pval,
        "fdr_bh_adjust_p": fdr_padj,
        "is_SVG_FDR0.05": reject
    })
    svg_df = svg_df.sort_values("spatial_correlation_score", ascending=False).reset_index(drop=True)
    return svg_df

# ===================== SpaGCN Graph Construction Function (No clustering) =====================
def run_SpaGCN_BuildGraph(adata, dataset, random_seed = np.random.randint(100),
                device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
    import SpaGCN_raw as spg
    import random

    # Standard SpaGCN preprocessing
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    # Auto extract spatial coordinates
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x','y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x','y']].values
    elif {'spatial_x','spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x','spatial_y']].values
    else:
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        adata.obsm['spatial'] = spatial_coords

    adata.obs['x_array'] = spatial_coords[:, 0]
    adata.obs['y_array'] = spatial_coords[:, 1]
    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()

    adj = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    return adata, adj

# ===================== MAIN EXECUTION =====================
if __name__ == '__main__':
    # -------------------------- Custom Path Definition --------------------------
    DATASET_FOLDER = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
    OUTPUT_ROOT    = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\spaGCN"

    mk_dir(OUTPUT_ROOT)
    warnings.filterwarnings('ignore')

    # Platform classification rule (match filename prefix)
    def get_platform_label(filename):
        fn = filename.lower()
        if fn.startswith("10x") or fn.startswith("10_x"):
            return "10X Visium"
        elif fn.startswith("slide-seq") or fn.startswith("slide-seq2"):
            return "Slide-seq"
        elif fn.startswith("st_"):
            return "ST"
        elif fn.startswith("starmap"):
            return "STARmap"
        elif fn.startswith("xenium"):
            return "Xenium"
        else:
            return "Unknown"

    # Scan all standardized h5ad files
    h5ad_file_list = glob.glob(os.path.join(DATASET_FOLDER, "*_standardized.h5ad"))
    if len(h5ad_file_list) == 0:
        raise FileNotFoundError(f"No standardized h5ad found in {DATASET_FOLDER}")

    # Global fixed random seed
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    global_summary = []

    # Iterate each dataset file
    for file_path in h5ad_file_list:
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace("_standardized.h5ad", "")
        platform_type = get_platform_label(file_name)

        print(f"\n==================================================")
        print(f"         SpaGCN SVG Detection | Dataset: {dataset_name} | Platform: {platform_type}")
        print(f"==================================================")

        # Create independent folder for each dataset
        dataset_out_dir = os.path.join(OUTPUT_ROOT, dataset_name)
        mk_dir(dataset_out_dir)

        # Load standardized adata directly
        adata_raw = sc.read_h5ad(file_path)
        adata_raw.var_names_make_unique()
        random_seed = np.random.randint(100)

        # Start time & memory record
        start_time = time.time()
        tracemalloc.start()
        start_rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3

        # Step1: SpaGCN preprocess + build spatial adjacency matrix
        adata_processed, adj_matrix = run_SpaGCN_BuildGraph(adata_raw.copy(), dataset_name, random_seed)
        # Step2: Compute SVG with permutation & FDR correction
        svg_df = calculate_SpaGCN_SVG(adata_processed, adj_matrix, n_permutation=1000, seed=random_seed)

        # End resource statistics
        run_sec = time.time() - start_time
        end_rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        mem_increment_gb = end_rss_gb - start_rss_gb
        _, peak_byte = tracemalloc.get_traced_memory()
        peak_gb = peak_byte / 1024**3
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Basic statistical information
        total_genes = adata_processed.n_vars
        sig_svg_count = int(svg_df["is_SVG_FDR0.05"].sum())
        top10_svg = svg_df.head(10)["gene_ids"].tolist()

        # Append global summary
        record = {
            "platform": platform_type,
            "dataset_name": dataset_name,
            "total_gene_count": total_genes,
            "significant_SVG_FDR0.05": sig_svg_count,
            "runtime_seconds": round(run_sec, 2),
            "ram_increase_GB": round(mem_increment_gb, 3),
            "peak_memory_GB": round(peak_gb, 3)
        }
        global_summary.append(record)

        # Export all SVG result files
        full_svg_path = os.path.join(dataset_out_dir, f"{dataset_name}_full_SVG_table.csv")
        svg_df.to_csv(full_svg_path, index=False)

        sig_svg_path = os.path.join(dataset_out_dir, f"{dataset_name}_significant_SVG(FDR0.05).csv")
        svg_df[svg_df["is_SVG_FDR0.05"]].to_csv(sig_svg_path, index=False)

        top50_path = os.path.join(dataset_out_dir, f"{dataset_name}_Top50_SVG.csv")
        svg_df.head(50).to_csv(top50_path, index=False)

        # Save processed adata with SVG results stored in uns
        adata_processed.uns["SpaGCN_SVG_Result"] = svg_df
        adata_processed.write_h5ad(os.path.join(dataset_out_dir, f"{dataset_name}_processed.h5ad"))

        # Console log
        print(f"✅ Completed {dataset_name}:")
        print(f"Total Genes: {total_genes} | Significant SVG(FDR<0.05): {sig_svg_count}")
        print(f"Top10 Spatial Variable Genes: {top10_svg}")
        print(f"All outputs saved: {dataset_out_dir}\n")

    # Save global benchmark summary table
    summary_df = pd.DataFrame(global_summary)
    summary_save_path = os.path.join(OUTPUT_ROOT, "SpaGCN_Global_SVG_Benchmark_Summary.csv")
    summary_df.to_csv(summary_save_path, index=False)
    print("==================== ALL DATASETS FINISHED ====================")
    print(f"Global summary table saved to: {summary_save_path}")
    print("Each dataset folder contains: full SVG table, significant SVG table, Top50 SVG table, processed h5ad")