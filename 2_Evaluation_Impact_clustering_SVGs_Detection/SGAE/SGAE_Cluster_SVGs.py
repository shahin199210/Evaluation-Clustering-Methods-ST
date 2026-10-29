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
import scipy.sparse as sp
import warnings
from scipy.spatial import cKDTree
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------------------
# GCN Layer (original from SGAE / Kipf & Welling GCN paper)
# --------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAE(nn.Module):
    def __init__(self, nfeat, nhid, nemb, dropout):
        super(GAE, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nemb)
        self.dropout = dropout

    def encode(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def forward(self, x, adj):
        emb = self.encode(x, adj)
        adj_recon = self.reconstruct(emb)
        return emb, adj_recon

    def reconstruct(self, emb):
        return torch.sigmoid(torch.mm(emb, emb.t()))

class SpatialGraphAutoEncoder(nn.Module):
    """SGAE with spatial graph constraint"""
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=32, dropout=0.1):
        super(SpatialGraphAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.gcn = GAE(embedding_dim, hidden_dim//2, embedding_dim, dropout)

    def forward(self, x, adj):
        feat_emb = self.encoder(x)
        spatial_emb, adj_recon = self.gcn(feat_emb, adj)
        x_recon = self.decoder(spatial_emb)
        return spatial_emb, x_recon, adj_recon

# --------------------------
# SGAE helper functions
# --------------------------
def normalize_adjacency(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    adj_normalized = torch.sparse_coo_tensor(
        np.vstack((adj_normalized.row, adj_normalized.col)),
        adj_normalized.data,
        adj_normalized.shape,
        dtype=torch.float32
    ).to(device)
    return adj_normalized

def get_spatial_adjacency(adata, rad_cutoff=50):
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
        pairs = set()
        for i in range(coords.shape[0]):
            for j in indices[i, 1:]:
                pairs.add((i, j))
    adj = sp.lil_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
    for i, j in pairs:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj = adj.tocoo()
    return normalize_adjacency(adj)

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# SGAE clustering pipeline (Removed evaluation, ground‑truth, metrics, plotting)
# --------------------------
def run_SGAE_clustering(adata_raw, n_clusters, epochs=200, rad_cutoff=50):
    adata = adata_raw.copy()

    # SGAE preprocessing
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    if "log1p" not in adata.uns:
        sc.pp.log1p(adata)

    n_obs, n_vars = adata.shape
    features = torch.tensor(
        adata.X.todense() if sp.issparse(adata.X) else adata.X,
        dtype=torch.float32
    ).to(device)
    adj = get_spatial_adjacency(adata, rad_cutoff=rad_cutoff)

    model = SpatialGraphAutoEncoder(
        input_dim=n_vars,
        hidden_dim=128,
        embedding_dim=32,
        dropout=0.1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_recon = nn.MSELoss()
    criterion_adj = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        emb, x_recon, adj_recon = model(features, adj)
        loss_recon = criterion_recon(x_recon, features)
        loss_adj = criterion_adj(adj_recon, adj.to_dense())
        loss = loss_recon + 0.1 * loss_adj
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        emb, _, _ = model(features, adj)

    embed_adata = anndata.AnnData(emb.cpu().numpy())
    sc.pp.neighbors(embed_adata, n_neighbors=20)
    eval_res = res_search_fixed_clus(embed_adata, n_clusters)
    sc.tl.leiden(embed_adata, key_added="SGAE_leiden", resolution=eval_res)

    # Attach cluster labels and embedding back to original full adata
    adata_raw.obs["SGAE_leiden"] = embed_adata.obs["SGAE_leiden"].values
    adata_raw.obsm["SGAE_embedding"] = embed_adata.X

    del adata, embed_adata, model, features
    gc.collect()
    torch.cuda.empty_cache()
    return adata_raw

# --------------------------
# Shared permutation‑test SVG function (same as DeepSpaCE / iSpatial / SpaNCMG / SEDR / conST / stCluster)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="SGAE_leiden", perm_times=1000, seed=0):
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
        "SGAE_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set
    gc.collect()
    return svg_result_df.sort_values("SGAE_SVG_score", ascending=False).reset_index(drop=True)

# --------------------------
# Global Configuration (7 standardized datasets)
# --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\SGAE"

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
    print("SGAE Clustering + Cluster‑based SVG Identification")
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

        # Run SGAE clustering
        clustered_adata = run_SGAE_clustering(adata, n_clusters=target_cluster, epochs=200, rad_cutoff=50)
        # Compute cluster‑based SVGs
        svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="SGAE_leiden", perm_times=1000, seed=0)

        # Save outputs
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"SGAE_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_SGAE_SVG_full_list.csv"), index=False)

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
        os.path.join(OUTPUT_ROOT_DIR, "SGAE_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "=" * 80)
    print("ALL 7 DATASETS COMPLETED")
    print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
    print("=" * 80)