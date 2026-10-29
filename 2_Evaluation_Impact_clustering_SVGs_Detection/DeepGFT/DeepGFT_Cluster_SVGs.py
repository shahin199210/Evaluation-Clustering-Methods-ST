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
import torch.nn as nn
import torch.nn.functional as F
import warnings
import scipy.sparse as sp
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
from scipy.sparse import issparse

# --------------------------
# DeepGFT Model Definition (REQUIRED ONLY PARTS)
# --------------------------
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('===== Using device:', device)

class GATv2Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=3, concat=False, dropout=0.2, add_self_loops=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        h = self.lin(x).view(-1, self.heads, self.out_channels)
        row, col = edge_index
        att_score = (h[row] * self.att).sum(dim=-1)
        alpha = F.softmax(att_score, dim=0)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        out = torch.zeros_like(h)
        out.index_add_(0, col, h[row] * alpha.unsqueeze(-1))
        if self.concat:
            return out.view(-1, self.heads * self.out_channels)
        else:
            return out.mean(dim=1)

class GAT_noncluster(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GAT_noncluster, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        num_hidden *= 2
        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv2 = torch.nn.Linear(num_hidden, out_dim, bias=True)
        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv4 = torch.nn.Linear(num_hidden, in_dim, bias=True)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1)
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = self.conv4(h3)
        return h2, h4

class SPOT(nn.Module):
    def __init__(self, dim, dims_spot=[64, 16], dropout=0.1, device='cuda:0', cluster=False):
        super(SPOT, self).__init__()
        self.dropout = dropout
        self.NFeature = dim
        self.device = device
        self.cluster = cluster
        self.GATE_spot = GAT_noncluster(hidden_dims=dims_spot).to(device)

    def forward(self, spot_freq_mtx, spotnet):
        emb_spot, mtx_res_spot = self.GATE_spot(spot_freq_mtx, spotnet)
        return emb_spot, mtx_res_spot

# --------------------------
# Required Utility Functions
# --------------------------
def get_spatial_graph(adata, rad_cutoff=150):
    if 'spatial' not in adata.obsm:
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])[:n]
        adata.obsm['spatial'] = dummy_coords
    coords = adata.obsm['spatial']
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=rad_cutoff)
    if len(pairs) == 0:
        distances, indices = tree.query(coords, k=5)
        pair_list = []
        for i in range(coords.shape[0]):
            for j in indices[i, 1:]:
                pair_list.append((i, j))
        pairs = set(pair_list)
    edges = np.array(list(pairs)).T
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    return torch.tensor(edges, dtype=torch.long).to(device)

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# ✅ FIXED DeepGFT Clustering Pipeline
# --------------------------
def run_DeepGFT_clustering(adata_raw, n_clusters, dataset_name, rad_cutoff=50, epochs=200):
    adata = adata_raw.copy()
    # Preprocessing consistent with original DeepGFT
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_obs, n_vars = adata.shape
    features = torch.tensor(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=torch.float32).to(device)
    edge_index = get_spatial_graph(adata, rad_cutoff=rad_cutoff)

    spot_model = SPOT(dim=64, dims_spot=[n_vars, 64, 16], cluster=False, device=device).to(device)
    optimizer = torch.optim.Adam(spot_model.parameters(), lr=0.001)
    spot_model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        emb_spot, recon = spot_model(features, edge_index)
        loss = F.mse_loss(recon, features)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    spot_model.eval()
    with torch.no_grad():
        emb_spot, _ = spot_model(features, edge_index)

    # --------------------------
    # Fixed line: emb_spot is cell‑level embedding (n_cells, latent_dim)
    # NO gene boolean mask indexing here
    # --------------------------
    emb_np = emb_spot.cpu().numpy()
    adata_raw.obsm["DeepGFT_embedding"] = emb_np

    # Leiden clustering on raw adata using DeepGFT embedding
    sc.pp.neighbors(adata_raw, use_rep='DeepGFT_embedding')
    res = res_search_fixed_clus(adata_raw, n_clusters)
    sc.tl.leiden(adata_raw, key_added="DeepGFT_cluster", resolution=res, random_state=0)

    del adata, spot_model, features, edge_index
    gc.collect()
    torch.cuda.empty_cache()
    return adata_raw

# --------------------------
# Unified Cluster‑based SVG Permutation Test (Consistent with IRIS/stFormer)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="DeepGFT_cluster", perm_times=1000, seed=0):
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
        "DeepGFT_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set
    gc.collect()
    return svg_result_df.sort_values("DeepGFT_SVG_score", ascending=False).reset_index(drop=True)

# --------------------------
# Global Configuration (7 Standardized Datasets)
# --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\DeepGFT"

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
    print("DeepGFT Clustering + Cluster‑based SVG Identification")
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

        adata = sc.read_h5ad(file_full_path)
        adata.var_names_make_unique()
        target_cluster = CLUSTER_NUM_DICT[dataset_name]
        print(f"  Target clusters: {target_cluster}")

        # Run DeepGFT clustering
        clustered_adata = run_DeepGFT_clustering(adata, n_clusters=target_cluster, dataset_name=dataset_name, rad_cutoff=50, epochs=200)
        # Compute SVGs
        svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="DeepGFT_cluster", perm_times=1000, seed=0)

        # Save outputs
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"DeepGFT_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_DeepGFT_SVG_full_list.csv"), index=False)

        svg_sig = svg_full_df[svg_full_df["is_SVG_FDR0.05"]]
        svg_sig.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Significant_SVG_FDR0.05.csv"), index=False)
        svg_full_df.head(50).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Top50_SVG.csv"), index=False)

        sig_svg_count = int(svg_sig.shape[0])
        single_record = {
            "platform": PLATFORM_LABEL[file_name],
            "dataset": dataset_name,
            "cluster_number": target_cluster,
            "total_gene_num": adata.n_vars,
            "significant_SVG_FDR0.05": sig_svg_count
        }
        global_summary.append(single_record)
        pd.DataFrame([single_record]).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_summary.csv"), index=False)

        print(f"  DONE | Significant SVGs (FDR<0.05): {sig_svg_count}")

        del adata, clustered_adata, svg_full_df, svg_sig
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(global_summary).to_csv(
        os.path.join(OUTPUT_ROOT_DIR, "DeepGFT_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "=" * 80)
    print("ALL 7 DATASETS COMPLETED")
    print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
    print("=" * 80)