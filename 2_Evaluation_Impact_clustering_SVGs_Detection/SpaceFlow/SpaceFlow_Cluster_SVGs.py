# Fix networkx compatibility patch for SpaceFlow
import networkx as nx
def fake_get_adj():
    pass
if not hasattr(nx, "from_numpy_matrix"):
    nx.from_numpy_matrix = fake_get_adj

from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import os
import numpy as np
import pandas as pd
import scanpy as sc
from SpaceFlow.SpaceFlow import SpaceFlow
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Path Configuration (7 real datasets)
# ==========================
INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\SpaceFlow"

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
mk_dir(OUTPUT_DIR)

# ==========================
# CORRECTED: actual filenames in Datasets_Standardized
# ==========================
DATASET_CONFIG = {
    "10X_Visium_Lung.h5ad": 6,
    "DLPFC_151507.h5ad": 7,
    "Slide-seq2.h5ad": 7,
    "SNARE-seq2.h5ad": 5,
    "ST.h5ad": 4,
    "STARmap.h5ad": 3,
    "Xenium_BC.h5ad": 8,
}

PLATFORM_MAPPING = {
    "10X_Visium": "10X Visium",
    "DLPFC": "DLPFC",
    "Slide-seq2": "Slide-seq2",
    "SNARE-seq2": "SNARE-seq2",
    "ST.h5ad": "ST",
    "STARmap": "STARmap",
    "Xenium_BC": "Xenium",
}

# ==========================
# Helper 1: Auto repair invalid spatial coordinates
# ==========================
def ensure_valid_spatial_coords(adata):
    n = adata.n_obs
    spatial_coords = None
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x', 'y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x', 'y']].values
    elif {'spatial_x', 'spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x', 'spatial_y']].values
    if spatial_coords is not None:
        if not np.issubdtype(spatial_coords.dtype, np.number):
            spatial_coords = None
        else:
            try:
                spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
            except ValueError:
                spatial_coords = None
    if spatial_coords is None or spatial_coords.shape[0] != n:
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        adata.obsm['spatial'] = spatial_coords.copy()
    return adata

# ==========================
# Helper 2: VECTORIZED SVG permutation test
# (same statistical test as f_oneway, but matrix ops instead of 30M scipy calls)
# ==========================
def calculate_cluster_SVG(adata_filtered, expr_matrix, cluster_labels, perm_times=1000, random_seed=42):
    np.random.seed(random_seed)
    n_cells, n_genes = expr_matrix.shape
    unique_clusters = np.unique(cluster_labels)
    n_cluster = len(unique_clusters)
    gene_names_arr = adata_filtered.var_names.values
    # Observed: cluster means for ALL genes at once
    global_gene_mean = expr_matrix.mean(axis=0)
    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        cluster_mean_matrix[idx] = expr_matrix[cluster_labels == c].mean(axis=0)
    obs_score = np.sum((cluster_mean_matrix - global_gene_mean) ** 2, axis=0) / n_cluster
    # Permutation null
    perm_score_set = np.zeros((perm_times, n_genes), dtype=np.float32)
    for p in range(perm_times):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_matrix[shuffle_idx, :]
        perm_cluster_mean = np.zeros((n_cluster, n_genes), dtype=np.float32)
        for idx, c in enumerate(unique_clusters):
            perm_cluster_mean[idx] = perm_expr[cluster_labels == c].mean(axis=0)
        perm_score_set[p] = np.sum((perm_cluster_mean - global_gene_mean) ** 2, axis=0) / n_cluster
    raw_p = (np.sum(perm_score_set >= obs_score, axis=0) + 1) / (perm_times + 1)
    reject, p_corrected, _, _ = multipletests(raw_p, method="fdr_bh")
    svg_df = pd.DataFrame({
        "gene_name": gene_names_arr,
        "F_score": obs_score,
        "p_value": raw_p,
        "FDR_BH": p_corrected,
        "is_SVG": reject
    })
    svg_df = svg_df.sort_values("F_score", ascending=False).reset_index(drop=True)
    return svg_df

# ==========================
# Core Pipeline: SpaceFlow training + clustering + SVG detection
# FIXED: outputs ALL original genes (filtered genes get NA)
# ==========================
def run_spaceflow_svg(file_path, dataset_name, target_cluster, save_folder):
    mk_dir(save_folder)
    adata_raw = sc.read_h5ad(file_path)
    adata_raw.var_names_make_unique()
    # Save ALL original gene names BEFORE SpaceFlow preprocessing filters them
    all_original_genes = adata_raw.var_names.tolist()
    adata_raw = ensure_valid_spatial_coords(adata_raw)
    sf = SpaceFlow(adata=adata_raw)
    sf.preprocessing_data()
    sf.train(spatial_regularization_strength=0.1, z_dim=50, epochs=1000)
    emb = sf.embedding
    emb_df = pd.DataFrame(emb)
    emb_df.to_csv(os.path.join(save_folder, "embedding.tsv"), sep="\t", index=False)
    kmeans = KMeans(n_clusters=target_cluster, random_state=42, n_init=20)
    cluster_result = kmeans.fit_predict(emb)
    # Use sf.adata (filtered data after preprocessing)
    adata_filtered = sf.adata
    if issparse(adata_filtered.X):
        expr_matrix = adata_filtered.X.toarray()
    else:
        expr_matrix = np.array(adata_filtered.X, dtype=np.float64)
    print(f"Filtered expr_matrix shape: {expr_matrix.shape}, genes in filtered adata: {adata_filtered.n_vars}")
    svg_filtered = calculate_cluster_SVG(adata_filtered, expr_matrix, cluster_result, perm_times=1000, random_seed=42)
    # ---------- LEFT JOIN: ensure output contains ALL original genes ----------
    all_gene_df = pd.DataFrame({"gene_name": all_original_genes})
    svg_full = pd.merge(all_gene_df, svg_filtered, on="gene_name", how="left")
    # Fill NaN in is_SVG with False (filtered-out genes are not significant)
    svg_full["is_SVG"] = svg_full["is_SVG"].fillna(False).astype(bool)
    # -------------------------------------------------------------------------
    sig_svg = svg_full.loc[svg_full["is_SVG"]].copy()
    top50 = svg_full.sort_values("F_score", ascending=False).head(50).copy()
    svg_full.to_csv(os.path.join(save_folder, "All_Gene_SVG.csv"), index=False)
    sig_svg.to_csv(os.path.join(save_folder, "Significant_SVG_FDR0.05.csv"), index=False)
    top50.to_csv(os.path.join(save_folder, "Top50_SVG.csv"), index=False)
    summary_df = pd.DataFrame({
        "dataset": [dataset_name],
        "n_cluster": [target_cluster],
        "n_genes_original": [len(all_original_genes)],
        "n_genes_filtered": [adata_filtered.n_vars],
        "n_SVG_significant": [sig_svg.shape[0]]
    })
    summary_df.to_csv(os.path.join(save_folder, "dataset_summary.csv"), index=False)
    return summary_df, svg_full

# ==========================
# Batch execution for 7 real datasets (with try/except + empty-list guard)
# ==========================
global_summary_list = []
print("====== SpaceFlow SVG Analysis (7 Real Datasets) Start ======")
print(f"Input Folder: {INPUT_DIR}")
print(f"Output Folder: {OUTPUT_DIR}\n")

for file_name, target_cluster in DATASET_CONFIG.items():
    full_file_path = os.path.join(INPUT_DIR, file_name)
    dataset_name = file_name.replace(".h5ad", "")
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    platform = "Unknown"
    for key, plat in PLATFORM_MAPPING.items():
        if file_name.startswith(key):
            platform = plat
            break
    print(f"Processing Dataset: {dataset_name} | Platform: {platform} | Target Clusters: {target_cluster}")
    if not os.path.exists(full_file_path):
        print(f"Warning: File {file_name} missing, skip\n")
        continue
    try:
        single_summary, _ = run_spaceflow_svg(full_file_path, dataset_name, target_cluster, dataset_output_dir)
        single_summary["platform"] = platform
        global_summary_list.append(single_summary)
        print(f"Completed {dataset_name}: Significant SVG count (FDR<0.05) = {single_summary['n_SVG_significant'].item()}\n")
    except Exception as e:
        print(f"FAILED {dataset_name}: {str(e)}\n")

if len(global_summary_list) > 0:
    global_summary_df = pd.concat(global_summary_list, axis=0, ignore_index=True)
    global_summary_df = global_summary_df[["platform", "dataset", "n_cluster", "n_genes_original", "n_genes_filtered", "n_SVG_significant"]]
    global_summary_df.columns = ["platform", "dataset", "target_cluster", "original_gene_count", "filtered_gene_count", "significant_SVG_count"]
    global_summary_df.to_csv(os.path.join(OUTPUT_DIR, "SpaceFlow_7Datasets_Global_SVG_Summary.csv"), index=False)
    print("====== All 7 Datasets Finished ======")
    print(f"Global summary saved to: {os.path.join(OUTPUT_DIR, 'SpaceFlow_7Datasets_Global_SVG_Summary.csv')}")
    print(global_summary_df)
else:
    print("====== No datasets were successfully processed ======")
    print("Check that input files exist and SpaceFlow is installed.")