import sys
import os
import time
import psutil
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

# -------------------------- Global Environment Configuration --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('===== Using device:', device)

# -------------------------- Original stRGAT Graph Attention Layer (No modification) --------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class stRGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(stRGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

# -------------------------- Reserved Core Utility Functions --------------------------
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.01):
    for res in sorted(list(np.arange(0.1, 5.0, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            return res
    return 1.0

def get_spatial_adj(adata, rad_cutoff=150):
    if 'spatial' not in adata.obsm:
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])[:n]
        adata.obsm['spatial'] = dummy_coords
    coords = adata.obsm['spatial']
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=rad_cutoff)
    if len(pairs) == 0:
        _, indices = tree.query(coords, k=5)
        pairs = set()
        for i in range(coords.shape[0]):
            for j in indices[i, 1:]:
                pairs.add((i, j))
    n = adata.n_obs
    adj = np.zeros((n, n))
    for i, j in pairs:
        adj[i, j] = 1
        adj[j, i] = 1
    adj = adj + np.eye(n)
    return torch.FloatTensor(adj).to(device)

# -------------------------- Core SVG Calculation Function --------------------------
def calc_cluster_SVG(adata, cluster_key="stRGAT_leiden", perm_times=1000, seed=0):
    np.random.seed(seed)
    expr_mat = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = expr_mat.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)
    global_gene_mean = expr_mat.mean(axis=0, keepdims=True)

    # Observed SVG score
    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        mask = cell_labels == c
        cluster_mean_matrix[idx] = expr_mat[mask].mean(axis=0)
    obs_score = np.sum((cluster_mean_matrix - global_gene_mean) ** 2, axis=0) / n_cluster

    # Permutation null distribution
    perm_score_set = np.zeros((perm_times, n_genes))
    for p in range(perm_times):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_cluster_mean = np.zeros((n_cluster, n_genes))
        for idx, c in enumerate(unique_clusters):
            mask = cell_labels == c
            perm_cluster_mean[idx] = perm_expr[mask].mean(axis=0)
        perm_score_set[p] = np.sum((perm_cluster_mean - global_gene_mean) ** 2, axis=0) / n_cluster

    # P-value and FDR-BH correction
    raw_p = (np.sum(perm_score_set >= obs_score, axis=0) + 1) / (perm_times + 1)
    sig_flag, fdr_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    svg_df = pd.DataFrame({
        "gene_name": adata.var.index.values,
        "stRGAT_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    return svg_df.sort_values("stRGAT_SVG_score", ascending=False).reset_index(drop=True)

# -------------------------- stRGAT Training & Clustering Pipeline --------------------------
def train_stRGAT_cluster(adata_raw, n_cluster, random_seed=0, rad_cutoff=150, epochs=1000):
    adata = adata_raw.copy()
    # Standard preprocessing
    try:
        if adata.X.max() < 20:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            adata = adata[:, adata.var.highly_variable]
        else:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            adata = adata[:, adata.var.highly_variable]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
    except:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    n_obs, n_vars = adata.shape

    # Build feature tensor
    if sp.issparse(adata.X):
        features = torch.tensor(adata.X.todense(), dtype=torch.float32).to(device)
    else:
        features = torch.tensor(adata.X, dtype=torch.float32).to(device)
    adj = get_spatial_adj(adata, rad_cutoff=rad_cutoff)

    # Model initialization
    nhid, nclass, dropout, alpha, nheads = 128, 64, 0.2, 0.2, 4
    model = stRGAT(nfeat=n_vars, nhid=nhid, nclass=nclass, dropout=dropout, alpha=alpha, nheads=nheads).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        emb = model(features, adj)
        recon_x = torch.matmul(emb, emb.T)
        recon_loss = F.mse_loss(recon_x, features @ features.T)
        reg_loss = torch.mean(emb ** 2)
        loss = recon_loss + 0.01 * reg_loss
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Extract embedding
    model.eval()
    with torch.no_grad():
        embeddings = model(features, adj).cpu().numpy()
    adata.obsm['stRGAT'] = embeddings
    adata_raw.obsm['stRGAT_embedding'] = embeddings

    # Leiden clustering
    sc.pp.neighbors(adata, use_rep='stRGAT', n_neighbors=10)
    res_opt = res_search_fixed_clus(adata, n_cluster)
    sc.tl.leiden(adata, key_added="stRGAT_leiden", resolution=res_opt, random_state=random_seed)
    adata_raw.obs["stRGAT_leiden"] = adata.obs["stRGAT_leiden"].values
    return adata_raw

# -------------------------- OOM Safe Downsampling Wrapper --------------------------
def run_stRGAT_OOM_cluster(adata_h5, n_cluster, random_seed=0, rad_cutoff=50):
    adata = adata_h5.copy()
    n_total = adata.n_obs
    target_min, target_max = 1200, 1500
    if n_total > target_max:
        target = (target_min + target_max) // 2
        adata = adata[adata.obs.sample(frac=target / n_total, random_state=42).index].copy()
        print(f"Downsampled dataset to {adata.n_obs} cells to avoid OOM")
    if 'spatial' in adata.obsm and adata.obsm['spatial'].shape[1] != 2:
        adata.obsm['spatial'] = adata.obsm['spatial'][:, :2]
    global device
    if adata.n_obs > 6000:
        device = torch.device('cpu')
        print("Switch to CPU for large dataset computation")
    return train_stRGAT_cluster(adata, n_cluster, random_seed, rad_cutoff, epochs=1000)

# -------------------------- Global Path & Dataset Configuration --------------------------
INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\stRGAT"
mk_dir(OUTPUT_DIR)

# Standardized dataset filename list
DATASET_LIST = [
    "Xenium_BC_NCBI784_standardized.h5ad",
    "Xenium_Lung_NCBI856_standardized.h5ad",
    "10X_BC_TENX13_standardized.h5ad",
    "10X_Lung_TENX62_standardized.h5ad",
    "ST_BC_SPA130_standardized.h5ad",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
    "STARmap_BOOST_HMI_standardized.h5ad"
]

# Predefined cluster numbers
CLUSTER_SETTING = {
    "Xenium_BC_NCBI784": 8,
    "Xenium_Lung_NCBI856": 6,
    "10X_BC_TENX13": 5,
    "10X_Lung_TENX62": 6,
    "ST_BC_SPA130": 4,
    "Slide-seq2_Kedney_HBM965.PSNC.855": 7,
    "STARmap_BOOST_HMI": 3
}

# Platform category mapping
PLATFORM_MAP = {
    "Xenium_BC_NCBI784_standardized.h5ad": "Xenium",
    "Xenium_Lung_NCBI856_standardized.h5ad": "Xenium",
    "10X_BC_TENX13_standardized.h5ad": "10X Visium",
    "10X_Lung_TENX62_standardized.h5ad": "10X Visium",
    "ST_BC_SPA130_standardized.h5ad": "ST",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad": "Slide-seq",
    "STARmap_BOOST_HMI_standardized.h5ad": "STARmap"
}

# -------------------------- Main Execution --------------------------
if __name__ == '__main__':
    global_summary_list = []
    print("=" * 90)
    print("stRGAT Clustering + Cluster-based SVG Detection Pipeline Start")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 90)

    for file_name in DATASET_LIST:
        file_path = os.path.join(INPUT_DIR, file_name)
        dataset_name = file_name.replace("_standardized.h5ad", "")
        dataset_save_folder = mk_dir(os.path.join(OUTPUT_DIR, dataset_name))
        print(f"\n==== Processing Dataset: {dataset_name} ====")

        # File missing check
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skip this dataset")
            continue

        adata = sc.read_h5ad(file_path)
        adata.var_names_make_unique()
        target_cluster = CLUSTER_SETTING[dataset_name]
        print(f"Target cluster number: {target_cluster}")

        # Start timing & memory recording
        task_start = time.time()
        tracemalloc.start()
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

        # Step1 stRGAT clustering
        if "10X" in dataset_name:
            clustered_adata = train_stRGAT_cluster(adata, target_cluster, rad_cutoff=150)
        else:
            clustered_adata = run_stRGAT_OOM_cluster(adata, target_cluster, rad_cutoff=50)

        # Step2 SVG identification
        svg_result = calc_cluster_SVG(clustered_adata, cluster_key="stRGAT_leiden", perm_times=1000, seed=0)

        # Calculate resource consumption
        total_time = time.time() - task_start
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        mem_growth = mem_end - mem_start
        _, peak_byte = tracemalloc.get_traced_memory()
        peak_gb = peak_byte / 1024 ** 3
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Export files
        clustered_adata.write_h5ad(os.path.join(dataset_save_folder, f"stRGAT_{dataset_name}_clustered.h5ad"))
        svg_result.to_csv(os.path.join(dataset_save_folder, f"{dataset_name}_stRGAT_SVG_full_list.csv"), index=False)
        sig_svg_df = svg_result[svg_result["is_SVG_FDR0.05"]]
        sig_svg_df.to_csv(os.path.join(dataset_save_folder, f"{dataset_name}_Significant_SVG_FDR0.05.csv"), index=False)
        svg_result.head(50).to_csv(os.path.join(dataset_save_folder, f"{dataset_name}_Top50_SVG.csv"), index=False)

        # Single dataset summary
        sig_count = sig_svg_df.shape[0]
        record = {
            "platform": PLATFORM_MAP[file_name],
            "dataset": dataset_name,
            "target_cluster": target_cluster,
            "total_gene_count": adata.n_vars,
            "significant_SVG_FDR0.05": sig_count,
            "runtime_seconds": round(total_time, 2),
            "ram_growth_GB": round(mem_growth, 3),
            "peak_memory_GB": round(peak_gb, 3)
        }
        global_summary_list.append(record)
        pd.DataFrame([record]).to_csv(os.path.join(dataset_save_folder, f"{dataset_name}_single_summary.csv"), index=False)
        print(f"✅ {dataset_name} finished | Significant SVG: {sig_count} | Runtime: {total_time:.2f}s")

    # Export global summary table
    pd.DataFrame(global_summary_list).to_csv(os.path.join(OUTPUT_DIR, "stRGAT_All_Datasets_Global_Summary.csv"), index=False)
    print("\n" + "=" * 90)
    print("All datasets finished, global summary file saved successfully!")
    print("=" * 90)