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
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

# -------------------------- Global Environment Setting --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# -------------------------- Reserved Original DSTN Model (No Modification) --------------------------
class DSTN(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, nhead=4, hidden_dim=128, num_layers=2):
        super(DSTN, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2,
            activation='relu', batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        h = self.proj(x)
        h = self.transformer_encoder(h)
        h = h.squeeze(0)
        z = self.fc(h)
        x_recon = self.decoder(z)
        return z, x_recon

# -------------------------- Reserved Fixed Resolution Leiden Search (No Modification) --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# -------------------------- Core SVG Calculation Function (Consistent Statistical Standard) --------------------------
def calc_cluster_SVG(adata, cluster_key="DSTN_leiden", perm_times=1000, seed=0):
    np.random.seed(seed)
    expr_mat = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = expr_mat.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)
    global_gene_mean = expr_mat.mean(axis=0, keepdims=True)

    # Observed SVG score
    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        cell_mask = cell_labels == c
        cluster_mean_matrix[idx] = expr_mat[cell_mask].mean(axis=0)
    obs_score = np.sum((cluster_mean_matrix - global_gene_mean) ** 2, axis=0) / n_cluster

    # Permutation null distribution
    perm_score_set = np.zeros((perm_times, n_genes))
    for p in range(perm_times):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_cluster_mean = np.zeros((n_cluster, n_genes))
        for idx, c in enumerate(unique_clusters):
            cell_mask = cell_labels == c
            perm_cluster_mean[idx] = perm_expr[cell_mask].mean(axis=0)
        perm_score_set[p] = np.sum((perm_cluster_mean - global_gene_mean) ** 2, axis=0) / n_cluster

    # P-value & FDR-BH correction
    raw_p = (np.sum(perm_score_set >= obs_score, axis=0) + 1) / (perm_times + 1)
    sig_flag, fdr_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    svg_result_df = pd.DataFrame({
        "gene_name": adata.var.index.values,
        "DSTN_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    return svg_result_df.sort_values("DSTN_SVG_score", ascending=False).reset_index(drop=True)

# -------------------------- Simplified DSTN Training & Clustering Pipeline --------------------------
def run_DSTN_cluster(adata_raw, n_cluster, epochs=200):
    adata = adata_raw.copy()
    # Standard preprocessing
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    max_pc = min(adata.n_obs, adata.n_vars)
    n_pca = min(200, max_pc - 1)
    if n_pca < 1:
        adata.obsm['X_pca'] = adata.X.copy()
    else:
        sc.pp.pca(adata, n_comps=n_pca, svd_solver='arpack')

    # DSTN model training
    X_tensor = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)
    model = DSTN(input_dim=n_pca).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        emb, recon = model(X_tensor)
        loss = loss_func(recon, X_tensor)
        loss.backward()
        optimizer.step()

    # Extract embedding & Leiden clustering
    model.eval()
    with torch.no_grad():
        embed_np = model(X_tensor)[0].cpu().numpy()
    embed_adata = anndata.AnnData(embed_np)
    sc.pp.neighbors(embed_adata, n_neighbors=20)
    res_opt = res_search_fixed_clus(embed_adata, n_cluster)
    sc.tl.leiden(embed_adata, resolution=res_opt, key_added="DSTN_leiden")

    # Write cluster label back to raw adata
    adata_raw.obs["DSTN_leiden"] = embed_adata.obs["DSTN_leiden"].values
    adata_raw.obsm["DSTN_embedding"] = embed_np
    return adata_raw

# -------------------------- Global Path Configuration --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\DSTN"

# Dataset list matching your local file names
DATA_FILE_LIST = [
    "Xenium_BC_NCBI784_standardized.h5ad",
    "Xenium_Lung_NCBI856_standardized.h5ad",
    "10X_BC_TENX13_standardized.h5ad",
    "10X_Lung_TENX62_standardized.h5ad",
    "ST_BC_SPA130_standardized.h5ad",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
    "STARmap_BOOST_HMI_standardized.h5ad"
]

# Predefined cluster number consistent with your previous setting
CLUSTER_NUM_DICT = {
    "Xenium_BC_NCBI784": 8,
    "Xenium_Lung_NCBI856": 6,
    "10X_BC_TENX13": 5,
    "10X_Lung_TENX62": 6,
    "ST_BC_SPA130": 4,
    "Slide-seq2_Kedney_HBM965.PSNC.855": 7,
    "STARmap_BOOST_HMI": 3
}

# Platform category mapping
PLATFORM_LABEL = {
    "Xenium_BC_NCBI784_standardized.h5ad": "Xenium",
    "Xenium_Lung_NCBI856_standardized.h5ad": "Xenium",
    "10X_BC_TENX13_standardized.h5ad": "10X Visium",
    "10X_Lung_TENX62_standardized.h5ad": "10X Visium",
    "ST_BC_SPA130_standardized.h5ad": "ST",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad": "Slide-seq",
    "STARmap_BOOST_HMI_standardized.h5ad": "STARmap"
}

# -------------------------- Main Execution Pipeline --------------------------
if __name__ == '__main__':
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    global_summary = []
    print("=" * 85)
    print("DSTN Clustering + Cluster-based SVG Identification Start")
    print(f"Dataset Folder: {DATA_INPUT_DIR}")
    print(f"Output Folder: {OUTPUT_ROOT_DIR}")
    print("=" * 85)

    for file_name in DATA_FILE_LIST:
        file_full_path = os.path.join(DATA_INPUT_DIR, file_name)
        dataset_name = file_name.replace("_standardized.h5ad", "")
        dataset_save_dir = os.path.join(OUTPUT_ROOT_DIR, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        print(f"\n==== Processing Dataset: {dataset_name} ====")

        # File existence check
        if not os.path.exists(file_full_path):
            print(f"Warning: File missing {file_full_path}, skip this dataset")
            continue

        # Load standardized h5ad
        adata = sc.read_h5ad(file_full_path)
        adata.var_names_make_unique()
        target_cluster = CLUSTER_NUM_DICT[dataset_name]
        print(f"Target cluster count: {target_cluster}")

        # Timer & memory record
        task_start = time.time()
        tracemalloc.start()
        mem_start_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

        # Step 1: DSTN clustering
        clustered_adata = run_DSTN_cluster(adata, n_cluster=target_cluster, epochs=200)
        # Step 2: SVG identification
        svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="DSTN_leiden", perm_times=1000, seed=0)

        # Calculate runtime & memory
        total_time = time.time() - task_start
        mem_end_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
        mem_increase = mem_end_gb - mem_start_gb
        _, peak_byte = tracemalloc.get_traced_memory()
        peak_gb = peak_byte / 1024 ** 3
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Export output files
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"DSTN_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_DSTN_SVG_full_list.csv"), index=False)
        svg_sig = svg_full_df[svg_full_df["is_SVG_FDR0.05"]]
        svg_sig.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Significant_SVG_FDR0.05.csv"), index=False)
        svg_full_df.head(50).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Top50_SVG.csv"), index=False)

        # Single dataset summary
        sig_svg_count = int(svg_sig.shape[0])
        single_record = {
            "platform": PLATFORM_LABEL[file_name],
            "dataset": dataset_name,
            "cluster_number": target_cluster,
            "total_gene_num": adata.n_vars,
            "significant_SVG_FDR0.05": sig_svg_count,
            "runtime_second": round(total_time, 2),
            "ram_increase_GB": round(mem_increase, 3),
            "peak_memory_GB": round(peak_gb, 3)
        }
        global_summary.append(single_record)
        pd.DataFrame([single_record]).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_summary.csv"), index=False)
        print(f"✅ {dataset_name} Finished | Significant SVG Count: {sig_svg_count} | Runtime: {total_time:.2f}s")

    # Export global summary table
    pd.DataFrame(global_summary).to_csv(
        os.path.join(OUTPUT_ROOT_DIR, "DSTN_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "=" * 85)
    print("All datasets completed successfully, global summary saved!")
    print("=" * 85)