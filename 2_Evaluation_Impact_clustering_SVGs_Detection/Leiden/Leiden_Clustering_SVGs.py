# -*- coding:utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from scipy.sparse import issparse
import os
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import time
import psutil
import tracemalloc
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
np.random.seed(0)
sc.settings.seed = 0

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique = len(adata.obs['leiden'].unique())
        if count_unique == fixed_clus_count:
            break
    return res

def calculate_leiden_based_SVG_fast(adata, n_permutation=1000, seed=0):
    np.random.seed(seed)
    expr_mat = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
    labels = adata.obs["leiden"].astype(np.int64).values
    n_cells, n_genes = expr_mat.shape
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    global_mean = expr_mat.mean(axis=0, keepdims=True)

    # Vectorized observed SVG score
    cluster_means = np.zeros((n_clusters, n_genes))
    for idx, c in enumerate(clusters):
        mask = labels == c
        cluster_means[idx] = expr_mat[mask].mean(axis=0)
    svg_score = np.sum((cluster_means - global_mean) ** 2, axis=0) / n_clusters

    # Vectorized permutation calculation
    perm_scores = np.zeros((n_permutation, n_genes))
    for p in range(n_permutation):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_cluster_means = np.zeros((n_clusters, n_genes))
        for idx, c in enumerate(clusters):
            mask = labels == c
            perm_cluster_means[idx] = perm_expr[mask].mean(axis=0)
        perm_scores[p] = np.sum((perm_cluster_means - global_mean) ** 2, axis=0) / n_clusters

    raw_pval = (np.sum(perm_scores >= svg_score, axis=0) + 1) / (n_permutation + 1)
    reject, fdr_padj, _, _ = multipletests(raw_pval, alpha=0.05, method='fdr_bh')

    df = pd.DataFrame({
        "gene_ids": adata.var.index.values,
        "leiden_based_SVG_score": svg_score,
        "raw_pvalue": raw_pval,
        "fdr_bh_adjust_p": fdr_padj,
        "is_SVG_FDR0.05": reject
    })
    return df.sort_values("leiden_based_SVG_score", ascending=False).reset_index(drop=True)

def run_leiden_cluster(adata_h5, save_path, dataset_name, n_cluster):
    adata = adata_h5.copy()
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
    target_res = res_search_fixed_clus(adata, n_cluster)
    sc.tl.leiden(adata, resolution=target_res, key_added="cluster", random_state=0)

    adata.write_h5ad(f"{save_path}/Leiden_{dataset_name}_clustered.h5ad")
    svg_df = calculate_leiden_based_SVG_fast(adata, n_permutation=1000, seed=0)
    svg_df.to_csv(f"{save_path}/{dataset_name}_Leiden_Based_SVG_full_table.csv", index=False)
    svg_df[svg_df["is_SVG_FDR0.05"]].to_csv(f"{save_path}/{dataset_name}_Significant_SVG_FDR0.05.csv", index=False)
    svg_df.head(50).to_csv(f"{save_path}/{dataset_name}_Top50_SVG.csv", index=False)

    record = {
        "dataset": dataset_name,
        "target_cluster_num": n_cluster,
        "total_genes": adata.n_vars,
        "significant_SVG_count": int(svg_df["is_SVG_FDR0.05"].sum())
    }
    return record, adata, svg_df

# Path Configuration
BASE_DATA_PATH = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\Leiden"

DATA_FILE_LIST = [
    "Xenium_BC_NCBI784_standardized.h5ad",
    "Xenium_Lung_NCBI856_standardized.h5ad",
    "10X_BC_TENX13_standardized.h5ad",
    "10X_Lung_TENX62_standardized.h5ad",
    "ST_BC_SPA130_standardized.h5ad",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
    "STARmap_BOOST_HMI_standardized.h5ad"
    
]

PLATFORM_MAP = {
    "Xenium_BC_NCBI784_standardized.h5ad": "Xenium",
    "Xenium_Lung_NCBI856_standardized.h5ad": "Xenium",
    "10X_BC_TENX13_standardized.h5ad": "10X Visium",
    "10X_Lung_TENX62_standardized.h5ad": "10X Visium",
    "ST_BC_SPA130_standardized.h5ad": "ST",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad": "Slide-seq",
    "STARmap_BOOST_HMI_standardized.h5ad": "STARmap"
    
}

DATASET_CLUSTER_NUM = {
    "Xenium_BC_NCBI784": 8,
    "Xenium_Lung_NCBI856": 6,
    "10X_BC_TENX13": 5,
    "10X_Lung_TENX62": 6,
    "ST_BC_SPA130": 4,
    "Slide-seq2_Kedney_HBM965.PSNC.855": 7,
    "STARmap_BOOST_HMI": 3
    
}

if __name__ == "__main__":
    mk_dir(OUTPUT_ROOT)
    global_summary_list = []
    print(f"\n{'='*75}")
    print(f"LEIDEN CLUSTERING + SVG IDENTIFICATION START")
    print(f"Folder: {BASE_DATA_PATH}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"{'='*75}")

    for file_name in DATA_FILE_LIST:
        full_path = os.path.join(BASE_DATA_PATH, file_name)
        ds_name = file_name.replace("_standardized.h5ad", "")
        save_dir = mk_dir(os.path.join(OUTPUT_ROOT, ds_name))
        print(f"\n==== Current Dataset Processing: {ds_name} ====")

        adata = sc.read_h5ad(full_path)
        adata.var_names_make_unique()
        print(f"Available obs columns: {list(adata.obs.columns)}")
        n_clu = DATASET_CLUSTER_NUM[ds_name]
        print(f"Manually set target cluster number: {n_clu}")

        start = time.time()
        tracemalloc.start()
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024**3

        rec, adata_out, svg_df = run_leiden_cluster(adata, save_dir, ds_name, n_clu)

        runtime = time.time() - start
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        mem_inc = mem_end - mem_start
        _, peak_mem_byte = tracemalloc.get_traced_memory()
        peak_mem = peak_mem_byte / 1024**3
        tracemalloc.stop()
        tracemalloc.clear_traces()

        platform = PLATFORM_MAP[file_name]
        summary = {
            "platform_category": platform,
            "dataset_name": ds_name,
            "target_cluster_num": n_clu,
            "total_genes": rec["total_genes"],
            "significant_SVG_FDR0.05": rec["significant_SVG_count"],
            "runtime_seconds": round(runtime, 2),
            "ram_increase_GB": round(mem_inc, 3),
            "peak_memory_GB": round(peak_mem, 3)
        }
        global_summary_list.append(summary)
        pd.DataFrame([summary]).to_csv(f"{save_dir}/{ds_name}_single_summary.csv", index=False)

        print(f"✅ {ds_name} finished | Significant SVG: {summary['significant_SVG_FDR0.05']}")
        print(f"Runtime: {runtime:.2f}s | Peak Memory: {peak_mem:.3f}GB\n")

    pd.DataFrame(global_summary_list).to_csv(
        os.path.join(OUTPUT_ROOT, "Leiden_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("="*75)
    print("All datasets completed successfully! Global summary saved.")
    print("="*75)