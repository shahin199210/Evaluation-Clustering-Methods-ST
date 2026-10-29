import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import gc
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import warnings
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

# Keep IRIS source imports (required, original IRIS_train / preprocess modules)
sys.path.insert(0, r"D:\My_Project_ST\1_Evaluation_SRT")
from IRIS_train import IRIS_train
from preprocess import preprocess_data

warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# Safe preprocessing function (keep original error‑handling for IRIS)
# --------------------------
def safe_preprocess(adata, dataset_name):
    try:
        return preprocess_data(adata)
    except Exception as e:
        print(f"⚠️ WARNING: Default preprocessing failed for {dataset_name}. Using safe fallback.")
        print(f"Error: {e}")
        adata = adata.copy()
        adata.var_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=1)
        if adata.n_vars == 0:
            raise ValueError(f"Dataset {dataset_name} has 0 genes left after filtering.")
        if not (adata.X.min() >= 0 and adata.X.max() <= 10):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
        return adata

# --------------------------
# ✅ FIXED IRIS clustering pipeline
# NO LONGER try paste labels back to raw adata (cell count mismatch bug)
# Return preprocessed adata_result directly for SVG calculation
# --------------------------
def run_IRIS_clustering(adata_raw, n_clusters, epochs=200):
    dataset_tmp = "tmp"
    adata_preprocessed = safe_preprocess(adata_raw, dataset_tmp)
    # Run original IRIS training
    adata_result = IRIS_train(adata_preprocessed, n_clusters=n_clusters, device=device, epochs=epochs)

    del adata_preprocessed
    gc.collect()
    torch.cuda.empty_cache()
    return adata_result

# --------------------------
# Shared permutation‑test SVG function (consistent with all other methods)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="IRIS_cluster", perm_times=1000, seed=0):
    np.random.seed(seed)
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = adata.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)

    expr_mat_sp = adata.X
    if issparse(expr_mat_sp):
        global_gene_mean = np.asarray(expr_mat_sp.mean(axis=0)).ravel()
    else:
        global_gene_mean = expr_mat_sp.mean(axis=0)

    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        cell_mask = cell_labels == c
        if issparse(expr_mat_sp):
            cluster_mean_matrix[idx] = np.asarray(expr_mat_sp[cell_mask].mean(axis=0)).ravel()
        else:
            cluster_mean_matrix[idx] = expr_mat_sp[cell_mask].mean(axis=0)

    obs_score = np.sum((cluster_mean_matrix - global_gene_mean) ** 2, axis=0) / n_cluster

    perm_score_set = np.zeros((perm_times, n_genes), dtype=np.float32)
    for p in range(perm_times):
        shuffle_idx = np.random.permutation(n_cells)
        perm_cluster_mean = np.zeros((n_cluster, n_genes), dtype=np.float32)
        for idx, c in enumerate(unique_clusters):
            cell_mask = cell_labels == c
            perm_sub = expr_mat_sp[shuffle_idx, :][cell_mask, :]
            if issparse(perm_sub):
                perm_cluster_mean[idx] = np.asarray(perm_sub.mean(axis=0)).ravel()
            else:
                perm_cluster_mean[idx] = perm_sub.mean(axis=0)
        perm_score_set[p] = np.sum((perm_cluster_mean - global_gene_mean) ** 2, axis=0) / n_cluster

    raw_p = (np.sum(perm_score_set >= obs_score, axis=0) + 1) / (perm_times + 1)
    sig_flag, fdr_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    svg_result_df = pd.DataFrame({
        "gene_name": adata.var.index.values,
        "IRIS_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set
    gc.collect()
    return svg_result_df.sort_values("IRIS_SVG_score", ascending=False).reset_index(drop=True)

# --------------------------
# Global Configuration (7 standardized datasets)
# --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\IRIS"

DATA_FILE_LIST = [
    "Xenium_BC_NCBI784_standardized.h5ad",
    "Xenium_Lung_NCBI856_standardized.h5ad",
    "10X_BC_TENX13_standardized.h5ad",
    "10X_Lung_TENX62_standardized.h5ad",
    "ST_BC_SPA130_standardized.h5ad",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
    "STARmap_BOOST_HMI_standardized.h5ad"
]

CLUSTER_NUM_DICT = {
    "Xenium_BC_NCBI784": 8,
    "Xenium_Lung_NCBI856": 6,
    "10X_BC_TENX13": 5,
    "10X_Lung_TENX62": 6,
    "ST_BC_SPA130": 4,
    "Slide-seq2_Kedney_HBM965.PSNC.855": 7,
    "STARmap_BOOST_HMI": 3
}

PLATFORM_LABEL = {
    "Xenium_BC_NCBI784_standardized.h5ad": "Xenium",
    "Xenium_Lung_NCBI856_standardized.h5ad": "Xenium",
    "10X_BC_TENX13_standardized.h5ad": "10X Visium",
    "10X_Lung_TENX62_standardized.h5ad": "10X Visium",
    "ST_BC_SPA130_standardized.h5ad": "ST",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad": "Slide-seq",
    "STARmap_BOOST_HMI_standardized.h5ad": "STARmap"
}

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    global_summary = []

    print("=" * 80)
    print("IRIS Clustering + Cluster‑based SVG Identification")
    print(f"Input : {DATA_INPUT_DIR}")
    print(f"Output: {OUTPUT_ROOT_DIR}")
    print("=" * 80)

    for file_name in DATA_FILE_LIST:
        file_full_path = os.path.join(DATA_INPUT_DIR, file_name)
        dataset_name = file_name.replace("_standardized.h5ad", "")
        dataset_save_dir = os.path.join(OUTPUT_ROOT_DIR, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)

        print(f"\n==== Processing: {dataset_name} ====")
        if not os.path.exists(file_full_path):
            print(f"  WARNING: File missing -> {file_full_path}, skipped")
            continue

        adata_raw = sc.read_h5ad(file_full_path)
        adata_raw.var_names_make_unique()
        target_cluster = CLUSTER_NUM_DICT[dataset_name]
        print(f"  Target clusters: {target_cluster}")

        # ✅ FIX: get filtered IRIS adata (do NOT use raw adata for SVG)
        adata_iris = run_IRIS_clustering(adata_raw, n_clusters=target_cluster, epochs=200)
        print(f"  After IRIS preprocessing: cells={adata_iris.n_obs}, genes={adata_iris.n_vars}")

        # Compute cluster‑based SVGs directly on IRIS‑processed adata
        svg_full_df = calc_cluster_SVG(adata_iris, cluster_key="IRIS_cluster", perm_times=1000, seed=0)

        # Save outputs (save IRIS‑processed adata, not raw adata)
        adata_iris.write_h5ad(os.path.join(dataset_save_dir, f"IRIS_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_IRIS_SVG_full_list.csv"), index=False)

        svg_sig = svg_full_df[svg_full_df["is_SVG_FDR0.05"]]
        svg_sig.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Significant_SVG_FDR0.05.csv"), index=False)
        svg_full_df.head(50).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Top50_SVG.csv"), index=False)

        sig_svg_count = int(svg_sig.shape[0])
        single_record = {
            "platform": PLATFORM_LABEL[file_name],
            "dataset": dataset_name,
            "cluster_number": target_cluster,
            "total_gene_num_original": adata_raw.n_vars,
            "gene_num_iris_processed": adata_iris.n_vars,
            "cell_num_original": adata_raw.n_obs,
            "cell_num_iris_processed": adata_iris.n_obs,
            "significant_SVG_FDR0.05": sig_svg_count
        }
        global_summary.append(single_record)
        pd.DataFrame([single_record]).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_summary.csv"), index=False)

        print(f"  DONE | Significant SVGs (FDR<0.05): {sig_svg_count}")

        del adata_raw, adata_iris, svg_full_df, svg_sig
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(global_summary).to_csv(
        os.path.join(OUTPUT_ROOT_DIR, "IRIS_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "=" * 80)
    print("ALL 7 DATASETS COMPLETED")
    print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
    print("=" * 80)