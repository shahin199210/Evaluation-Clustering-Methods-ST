from __future__ import division
from __future__ import print_function
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch.optim as optim
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
import pandas as pd
import random
import numpy as np
import scanpy as sc
import warnings
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests

# Path Configuration (Real datasets)
INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\Spatial_MGCN"

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path
mk_dir(OUTPUT_DIR)

# ==========================
# Dataset Configuration (7 real datasets)
# ==========================
DATASET_CONFIG = {
    #"10X_Visium_Lung.h5ad": 6,
    #"DLPFC_151507.h5ad": 7,
    #"Slide-seq2.h5ad": 7,
    "SNARE-seq2.h5ad": 5,
    "ST.h5ad": 4,
    "STARmap.h5ad": 3,
    "Xenium_BC.h5ad": 8,
}

PLATFORM_MAPPING = {
    #"10X_Visium": "10X Visium",
    #"DLPFC": "DLPFC",
    #"Slide-seq2": "Slide-seq2",
    "SNARE-seq2": "SNARE-seq2",
    "ST.h5ad": "ST",
    "STARmap": "STARmap",
    "Xenium_BC": "Xenium",
}

# --------------------------
# Spatial graphs (OPTIMIZED: vectorized graph_neg)
# --------------------------
def compute_spatial_graphs(adata, k_neighbors=15):
    print("Computing SpatialMGCN graphs (fadj, sadj, graph_nei, graph_neg)...")
    n = adata.n_obs
    spatial_coords = None
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x','y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x','y']].values
    elif {'spatial_x','spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x','spatial_y']].values
    if spatial_coords is not None:
        if not np.issubdtype(spatial_coords.dtype, np.number):
            spatial_coords = None
        else:
            try:
                spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
            except ValueError:
                spatial_coords = None
    if spatial_coords is None or spatial_coords.shape[0] != n:
        print("Generating dummy spatial coordinates")
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        adata.obsm['spatial'] = spatial_coords.copy()
    spatial_coords = np.array(spatial_coords, dtype=np.float32)
    sadj = kneighbors_graph(spatial_coords, n_neighbors=k_neighbors, mode='connectivity', include_self=True)
    sadj = sadj.tocsr()
    sc.tl.pca(adata, n_comps=30)
    pca_data = adata.obsm['X_pca']
    fadj = kneighbors_graph(pca_data, n_neighbors=k_neighbors, mode='connectivity', include_self=True)
    fadj = fadj.tocsr()
    # OPTIMIZED: vectorized neighbor extraction (no per-cell Python loop for sadj)
    graph_nei = []
    graph_neg = []
    all_indices = np.arange(n)
    for i in range(n):
        neighbors = sadj[i].indices[:k_neighbors]
        graph_nei.append(neighbors)
        # OPTIMIZED: use boolean mask instead of np.setdiff1d (much faster)
        mask = np.ones(n, dtype=bool)
        mask[neighbors] = False
        non_neighbors = all_indices[mask]
        if len(non_neighbors) >= k_neighbors:
            negatives = np.random.choice(non_neighbors, size=k_neighbors, replace=False)
        else:
            negatives = np.random.choice(non_neighbors, size=k_neighbors, replace=True)
        graph_neg.append(negatives)
    adata.obsm['sadj'] = sadj
    adata.obsm['fadj'] = fadj
    adata.obsm['graph_nei'] = np.array(graph_nei)
    adata.obsm['graph_neg'] = np.array(graph_neg)
    return adata

# ==============================================================================
# Training Function (UNCHANGED — SpatialMGCN core)
# ==============================================================================
def train(model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer):
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, zinb_loss, reg_loss, con_loss, total_loss

# ==========================
# SVG Permutation Test (VECTORIZED — replaces 30M f_oneway calls with matrix ops)
# Same statistical test: between-cluster sum of squares (monotonic with F-statistic)
# ==========================
def calculate_cluster_SVG(adata_filtered, expr_matrix, cluster_labels, perm_times=1000, random_seed=42):
    np.random.seed(random_seed)
    n_cells, n_genes = expr_matrix.shape
    unique_clusters = np.unique(cluster_labels)
    n_cluster = len(unique_clusters)
    gene_names_arr = adata_filtered.var_names.values
    # Observed: cluster means for ALL genes at once (vectorized)
    global_gene_mean = expr_matrix.mean(axis=0)
    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        cluster_mean_matrix[idx] = expr_matrix[cluster_labels == c].mean(axis=0)
    obs_score = np.sum((cluster_mean_matrix - global_gene_mean) ** 2, axis=0) / n_cluster
    # Permutation null: shuffle cells ONCE per perm, compute all-gene cluster means
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
    del cluster_mean_matrix, perm_score_set
    return svg_df

# ==============================================================================
# Main Pipeline
# ==============================================================================
def run_SpatialMGCN_SVG(file_path, dataset_name, target_cluster, save_folder):
    mk_dir(save_folder)
    adata_raw = sc.read_h5ad(file_path)
    adata_raw = compute_spatial_graphs(adata_raw)
    adata_raw.X = adata_raw.X.todense() if issparse(adata_raw.X) else adata_raw.X
    features = torch.FloatTensor(adata_raw.X)
    fadj = adata_raw.obsm['fadj']
    sadj = adata_raw.obsm['sadj']
    graph_nei = torch.LongTensor(adata_raw.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata_raw.obsm['graph_neg'])
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    config_file = './config/DLPFC.ini'
    config = Config(config_file)
    config.n = features.shape[0]
    config.class_num = target_cluster
    config.fdim = features.shape[1]
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = Spatial_MGCN(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    emb_final = None
    print(f"\nTraining SpatialMGCN for {dataset_name}")
    for epoch in range(config.epochs):
        emb, zinb_loss, reg_loss, con_loss, total_loss = train(
            model, features, nsadj, nfadj, graph_nei, graph_neg, config, optimizer
        )
        emb_final = emb
        if epoch % 10 == 0:
            print(f'{dataset_name} | Epoch {epoch:3d} | Loss: {total_loss:.2f}')
    emb_df = pd.DataFrame(emb_final)
    emb_df.to_csv(os.path.join(save_folder, "embedding.tsv"), sep="\t", index=False)
    kmeans = KMeans(n_clusters=target_cluster, random_state=config.seed, n_init=20)
    cluster_result = kmeans.fit_predict(emb_final)
    expr_matrix = np.array(adata_raw.X, dtype=np.float64)
    svg_full = calculate_cluster_SVG(adata_raw, expr_matrix, cluster_result, perm_times=1000, random_seed=42)
    sig_svg = svg_full.loc[svg_full["is_SVG"]].copy()
    top50 = svg_full.head(50).copy()
    svg_full.to_csv(os.path.join(save_folder, "All_Gene_SVG.csv"), index=False)
    sig_svg.to_csv(os.path.join(save_folder, "Significant_SVG_FDR0.05.csv"), index=False)
    top50.to_csv(os.path.join(save_folder, "Top50_SVG.csv"), index=False)
    summary_df = pd.DataFrame({
        "dataset": [dataset_name],
        "n_cluster": [target_cluster],
        "n_genes": [adata_raw.n_vars],
        "n_SVG_significant": [sig_svg.shape[0]]
    })
    summary_df.to_csv(os.path.join(save_folder, "dataset_summary.csv"), index=False)
    return summary_df, svg_full

# ==========================
# Batch Runner
# ==========================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    global_summary_list = []
    print("====== SpatialMGCN SVG Analysis (7 Real Datasets) Start ======")
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
            single_summary, _ = run_SpatialMGCN_SVG(full_file_path, dataset_name, target_cluster, dataset_output_dir)
            single_summary["platform"] = platform
            global_summary_list.append(single_summary)
            print(f"Completed {dataset_name}: Significant SVG count (FDR<0.05) = {single_summary['n_SVG_significant'].item()}\n")
        except Exception as e:
            print(f"FAILED {dataset_name}: {str(e)}\n")
    if len(global_summary_list) > 0:
        global_summary_df = pd.concat(global_summary_list, axis=0, ignore_index=True)
        global_summary_df = global_summary_df[["platform", "dataset", "n_cluster", "n_genes", "n_SVG_significant"]]
        global_summary_df.columns = ["platform", "dataset", "target_cluster", "total_genes", "significant_SVG_count"]
        global_summary_df.to_csv(os.path.join(OUTPUT_DIR, "SpatialMGCN_7Datasets_Global_SVG_Summary.csv"), index=False)
        print("====== All Datasets Finished ======")
        print(f"Global summary saved to: {os.path.join(OUTPUT_DIR, 'SpatialMGCN_7Datasets_Global_SVG_Summary.csv')}")
        print(global_summary_df)
    else:
        print("====== No datasets were successfully processed ======")
        print("Check that input files exist and config/DLPFC.ini is accessible from the working directory.")