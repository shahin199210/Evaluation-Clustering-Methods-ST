import sys
import os
import warnings
import gc
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

np.random.seed(0)
import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# DeepSpaCE Model
# --------------------------
class DeepSpaCE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 64, 32]):
        super(DeepSpaCE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[2], hidden_dims[3])
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[3], hidden_dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[2], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# --------------------------
# Fixed-resolution Leiden search
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# Cluster-based SVG detection (permutation test)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="DeepSpaCE_leiden", perm_times=1000, seed=0):
    np.random.seed(seed)
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = adata.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)

    if issparse(adata.X):
        expr_mat_sp = adata.X
        global_gene_mean = np.asarray(expr_mat_sp.mean(axis=0)).ravel()
    else:
        expr_mat_sp = adata.X
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
        "DeepSpaCE_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set
    gc.collect()
    return svg_result_df.sort_values("DeepSpaCE_SVG_score", ascending=False).reset_index(drop=True)

# --------------------------
# DeepSpaCE clustering + SVG pipeline
# --------------------------
def run_DeepSpaCE_svg(adata_raw, n_cluster, epochs=200):
    adata = adata_raw.copy()
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

    X_tensor = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    model = DeepSpaCE(input_dim=n_pca).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, X_recon = model(X_tensor)
        loss = criterion(X_recon, X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X_tensor).cpu().numpy()
    del model, X_tensor
    torch.cuda.empty_cache()

    embed_adata = anndata.AnnData(embedding)
    sc.pp.neighbors(embed_adata, n_neighbors=20)
    res_opt = res_search_fixed_clus(embed_adata, n_cluster)
    sc.tl.leiden(embed_adata, resolution=res_opt, key_added="DeepSpaCE_leiden")

    adata_raw.obs["DeepSpaCE_leiden"] = embed_adata.obs["DeepSpaCE_leiden"].values
    adata_raw.obsm["DeepSpaCE_embedding"] = embedding
    del adata, embed_adata
    gc.collect()
    return adata_raw

# --------------------------
# Global Configuration
# --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\DeepSpaCE"

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
if __name__ == '__main__':
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    global_summary = []

    print("=" * 80)
    print("DeepSpaCE Clustering + Cluster-based SVG Identification")
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

        clustered_adata = run_DeepSpaCE_svg(adata, n_cluster=target_cluster, epochs=200)
        svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="DeepSpaCE_leiden",
                                       perm_times=1000, seed=0)

        # Save outputs
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"DeepSpaCE_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_DeepSpaCE_SVG_full_list.csv"), index=False)

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
        os.path.join(OUTPUT_ROOT_DIR, "DeepSpaCE_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "=" * 80)
    print("ALL 7 DATASETS COMPLETED")
    print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
    print("=" * 80)