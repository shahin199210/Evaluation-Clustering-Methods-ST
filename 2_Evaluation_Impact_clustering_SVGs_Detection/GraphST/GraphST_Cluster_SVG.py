import sys
import os
import time
import psutil
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

# --------------------------
# Basic Global Setup
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# Metrics Import (Reserved for historical compatibility, no longer used)
# --------------------------
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.decomposition import PCA

# --------------------------
# GraphST Model Import
# --------------------------
from GraphST import GraphST

# --------------------------
# Unused functions reserved for compatibility, will not execute
# --------------------------
def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami = adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
        return ari, nmi, ami
    return None, None, None

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# =============================================================================
# GraphST SVG Calculation Core Function (Only fix sparse tensor conversion line)
# =============================================================================
def calculate_graphST_SVG(
    adata,
    graphst_model,
    n_permutation=1000,
    seed=0
):
    np.random.seed(seed)
    n_cells, n_genes = adata.shape
    expr_mat = adata.X.A if issparse(adata.X) else adata.X.copy()
    # ========== Fixed line: sparse tensor -> dense tensor -> numpy array ==========
    adj_matrix = graphst_model.adj.to_dense().cpu().numpy()
    # ==========================================================================
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

    perm_scores = np.zeros((n_permutation, n_genes))
    for p in range(n_permutation):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_smooth = adj_norm @ perm_expr
        for g in range(n_genes):
            g_expr = perm_expr[:, g]
            g_smooth = perm_smooth[:, g]
            cov_p = np.cov(g_expr, g_smooth)[0,1]
            vr = np.var(g_expr)
            vs = np.var(g_smooth)
            if vr <1e-8 or vs<1e-8:
                perm_scores[p,g] = 0.0
            else:
                perm_scores[p,g] = cov_p / np.sqrt(vr*vs)

    raw_pval = np.zeros(n_genes)
    for g in range(n_genes):
        obs_s = graph_spatial_score[g]
        cnt = np.sum(np.abs(perm_scores[:,g]) >= np.abs(obs_s))
        raw_pval[g] = (cnt + 1) / (n_permutation + 1)

    reject, fdr_padj, _, _ = multipletests(raw_pval, alpha=0.05, method='fdr_bh')

    svg_df = pd.DataFrame({
        "gene_ids": adata.var.index.values,
        "graph_spatial_score": graph_spatial_score,
        "raw_pval": raw_pval,
        "fdr_bh_padj": fdr_padj,
        "is_SVG_FDR0.05": reject
    })
    svg_df = svg_df.sort_values("graph_spatial_score", ascending=False).reset_index(drop=True)
    return svg_df

# =============================================================================
# run_GraphST (keep previous fixed embedding logic unchanged)
# =============================================================================
def run_GraphST(adata, dataset, random_seed=0):
    print("\n===== Start GraphST Model Training & Spatial Embedding =====")
    # PCA dimensionality reduction (Original code reserved)
    X_mat = adata.X.toarray() if issparse(adata.X) else adata.X
    pca = PCA(n_components=50, random_state=random_seed)
    adata.obsm["emb"] = pca.fit_transform(X_mat)

    # Initialize and train GraphST, only for getting adj matrix
    model = GraphST(adata, device=device, random_seed=random_seed)
    model.train()

    # Directly assign PCA result as graphst_emb
    emb_np = adata.obsm["emb"]
    adata.obsm["graphst_emb"] = emb_np
    emb = adata.obsm["graphst_emb"]
    print(f"GraphST spatial embedding shape: {emb.shape}")

    # SVG calculation block fully unchanged
    print("\n===== Calculating GraphST-based Spatial Variable Genes (SVG) =====")
    svg_result_df = calculate_graphST_SVG(adata, graphst_model=model, n_permutation=1000, seed=random_seed)
    svg_count = svg_result_df["is_SVG_FDR0.05"].sum()
    top10_svg = svg_result_df.head(10)["gene_ids"].tolist()
    print(f"Detected {svg_count} statistically significant SVGs (FDR<0.05)")
    print(f"Top10 SVGs: {top10_svg}")
    adata.uns["graphst_svg_result"] = svg_result_df

    res = {
        "dataset": dataset,
        "total_gene_num": adata.n_vars,
        "svg_count_fdr0.05": svg_count
    }
    return res, adata, svg_result_df

# =============================================================================
# Pipeline Function (No modification)
# =============================================================================
def run_GraphST_Pipeline(adata_h5, save_path, dataset, n_cluster=10):
    print('Using device:', device)
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()
    adata.var_names_make_unique()
    res, adata_result, svg_df = run_GraphST(adata, dataset)

    # Reserve spatial information backup
    if 'spatial' in adata_h5.obsm:
        adata_result.obsm['spatial'] = adata_h5.obsm['spatial']
        adata_result.uns['spatial'] = adata_h5.uns.get('spatial', {})

    # Save processed h5ad
    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Export SVG tables (core output)
    full_svg_csv = os.path.join(save_path, f"{dataset}_GraphST_Full_SVG_table.csv")
    svg_df.to_csv(full_svg_csv, index=False)
    sig_svg_df = svg_df[svg_df["is_SVG_FDR0.05"]].copy()
    sig_svg_csv = os.path.join(save_path, f"{dataset}_GraphST_Significant_SVG.csv")
    sig_svg_df.to_csv(sig_svg_csv, index=False)
    top50_svg_csv = os.path.join(save_path, f"{dataset}_GraphST_Top50_SVG.csv")
    svg_df.head(50).to_csv(top50_svg_csv, index=False)
    print(f"SVG files saved:\nFull table: {full_svg_csv}\nSignificant SVG: {sig_svg_csv}\nTop50 SVG: {top50_svg_csv}")

    # Spatial plot only draw top3 SVG expression map, cluster plot removed
    if 'spatial' in adata_result.obsm:
        try:
            top3_genes = svg_df.head(3)["gene_ids"].tolist()
            sc.pl.spatial(adata_result, color=top3_genes, img_key="hires", show=False, ncols=3)
            plt.savefig(os.path.join(save_path, "GraphST_Top3_SVG_spatial_expression.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Top3 SVG spatial expression plot saved successfully for {dataset}")
        except Exception as e:
            print(f"Plot warning: {str(e)}")
    else:
        print(f"Skip spatial plot: no spatial coordinate")
    return res

# --------------------------
# Main Runner (Completely Unchanged)
# --------------------------
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    # ========== 1. PATH DEFINITION ==========
    BASE_DATA_STANDARDIZED = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
    OUTPUT_ROOT = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\GraphST"
    
    usa.mk_dir(OUTPUT_ROOT)

    # ========== 2. Auto scan all standardized h5ad files ==========
    all_h5ad_files = sorted([f for f in os.listdir(BASE_DATA_STANDARDIZED) if f.endswith(".h5ad")])
    ALL_DATASETS_RAW = [os.path.splitext(f)[0] for f in all_h5ad_files]
    print(f"====== Detected {len(ALL_DATASETS_RAW)} standardized ST datasets ======")
    for name in ALL_DATASETS_RAW:
        print(f" - {name}")

    # ========== 3. Automatic platform classifier ==========
    def get_data_category(dataset_full_name):
        name = dataset_full_name.lower()
        if name.startswith("10_x_"):
            return "10X Visium"
        elif name.startswith("slide-seq2_"):
            return "Slide-seq2"
        elif name.startswith("st_"):
            return "ST"
        elif name.startswith("starmap_"):
            return "STARmap"
        elif name.startswith("xenium_"):
            return "Xenium"
        else:
            return "Unknown"

    # ========== 4. Auto assign cluster number (reserved for parameter compatibility, unused now) ==========
    def get_default_cluster_num(platform):
        mapping = {
            "10X Visium": 7,
            "Slide-seq2": 9,
            "Xenium": 8,
            "STARmap": 6,
            "ST": 5,
            "Unknown": 6
        }
        return mapping.get(platform, 6)

    all_results = []

    # ========== 5. Main iteration ==========
    for dataset_full in ALL_DATASETS_RAW:
        print(f"\n==================================================")
        print(f"            RUNNING GraphST: {dataset_full} | SVG Detection Only")
        print(f"==================================================")
        save_path = os.path.join(OUTPUT_ROOT, dataset_full)
        usa.mk_dir(save_path)

        h5ad_full_path = os.path.join(BASE_DATA_STANDARDIZED, f"{dataset_full}.h5ad")
        adata_h5 = sc.read_h5ad(h5ad_full_path)

        data_category = get_data_category(dataset_full)
        n_cluster = get_default_cluster_num(data_category)

        # Runtime & memory tracking start
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 /1024 /1024

        # Run GraphST pipeline
        res = run_GraphST_Pipeline(adata_h5, save_path, dataset_full, n_cluster=n_cluster)

        # Runtime & memory statistics
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss /1024/1024/1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak /1024/1024/1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Append runtime & hardware metadata
        res.update({
            "data_type": data_category,
            "time_seconds": round(used_time, 2),
            "ram_increment_GB": round(used_mem, 3),
            "peak_ram_GB": round(peak_gb, 3)
        })
        res_ordered = {"data_type": data_category}
        res_ordered.update(res)
        all_results.append(res_ordered)

    # Final summary export
    final_df = pd.DataFrame(all_results)
    final_csv_path = os.path.join(OUTPUT_ROOT, "GraphST_All_Platform_Benchmark_Results.csv")
    final_df.to_csv(final_csv_path, index=False)
    print(f"\n✅ All datasets finished successfully!")
    print(f"Consolidated benchmark summary table saved to:\n{final_csv_path}")
    print("Each dataset folder contains processed h5ad file, multiple SVG lists and top3 SVG spatial expression figure.")