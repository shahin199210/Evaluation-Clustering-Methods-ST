# -*- coding:utf-8 -*-
import sys
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from scipy.sparse import issparse
import os
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
np.random.seed(0)

# ==========================
# Utility Functions
# ==========================
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# ==========================
# Louvain clustering + Cluster-based SVG detection
# ==========================
def run_louvain_svg(adata_h5, save_path, dataset):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()

    # Preprocessing pipeline consistent with original clustering code
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)

    # Fixed resolution Louvain (removed fixed-cluster search; no ground_truth available)
    sc.tl.louvain(adata, key_added="louvain_cluster", resolution=0.8, random_state=0)
    cluster_labels = adata.obs["louvain_cluster"].values

    # --------------------------
    # Cluster-based SVG identification (ANOVA across Louvain clusters)
    # --------------------------
    gene_names = adata.var_names.tolist()
    expr_matrix = adata.X  # shape: (n_cells, n_genes)
    p_values = []

    for gene_idx in range(expr_matrix.shape[1]):
        gene_expr = expr_matrix[:, gene_idx]
        groups = [gene_expr[cluster_labels == clu] for clu in np.unique(cluster_labels)]
        stat, p = f_oneway(*groups)
        p_values.append(p)

    # FDR Benjamini–Hochberg correction
    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    # Result table
    svg_df = pd.DataFrame({
        "gene": gene_names,
        "p_value": p_values,
        "p_adj_FDR": p_adj,
        "is_SVG_FDR0.05": reject
    })

    output_csv = os.path.join(save_path, f"{dataset}_Louvain_Cluster_SVG_Result.csv")
    svg_df.to_csv(output_csv, index=False)

    sig_count = np.sum(reject)
    print(f"✅ {dataset} finished | Significant SVGs (FDR<0.05): {sig_count}\n")
    return {"dataset": dataset, "significant_SVG_count": sig_count}

# ==========================
# MAIN RUNNER
# ==========================
if __name__ == "__main__":
    # Input & Output Paths as requested
    INPUT_DIR  = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
    OUTPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\louvain"
    mk_dir(OUTPUT_DIR)

    # Hardcode ONLY your 7 target datasets matching file names in screenshot
    target_file_list = [
        "Xenium_BC_NCBI784_standardized.h5ad",
        "Xenium_Lung_NCBI856_standardized.h5ad",
        "10X_BC_TENX13_standardized.h5ad",
        "10X_Lung_TENX62_standardized.h5ad",
        "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
        "ST_BC_SPA130_standardized.h5ad",
        "STARmap_BOOST_HMI_standardized.h5ad"
    ]

    print(f"Total selected datasets to run: {len(target_file_list)}\n")

    batch_summary = []
    for filename in target_file_list:
        dataset_name = filename.replace("_standardized.h5ad", "")
        full_h5ad_path = os.path.join(INPUT_DIR, filename)
        dataset_out_folder = os.path.join(OUTPUT_DIR, dataset_name)
        mk_dir(dataset_out_folder)

        print(f"==================================================")
        print(f"            PROCESS DATASET: {dataset_name}")
        print(f"==================================================")

        try:
            adata = sc.read_h5ad(full_h5ad_path)
            # No longer require ground_truth column
            res = run_louvain_svg(adata, dataset_out_folder, dataset_name)
            batch_summary.append(res)
        except Exception as err:
            print(f"❌ FAILED: {dataset_name}, Error message: {str(err)}\n")
            batch_summary.append({"dataset": dataset_name, "significant_SVG_count": "Failed"})

    # Save global batch summary
    summary_df = pd.DataFrame(batch_summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Louvain_Batch_SVG_Summary.csv"), index=False)
    print("\n==== ALL TARGET DATASETS PROCESSING FINISHED ====")
    print(f"All SVG outputs saved to: {OUTPUT_DIR}")
    print(summary_df)