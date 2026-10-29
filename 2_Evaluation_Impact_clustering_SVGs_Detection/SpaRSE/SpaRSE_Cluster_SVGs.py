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
print('===== Using device:', device)

# --------------------------
# SpaRSE Model (UNCHANGED)
# --------------------------
class SpaRSE(torch.nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, hidden_dim=128):
        super(SpaRSE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# --------------------------
# Fixed-resolution Leiden search (UNCHANGED)
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# Cluster-based SVG detection (permutation test, same as DeepSpaCE)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="SpaRSE_leiden", perm_times=1000, seed=0):
    np.random.seed(seed)
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = adata.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)
    # Convert to dense once for fast permutations
    expr_mat = adata.X.toarray() if issparse(adata.X) else np.array(adata.X, dtype=np.float64)
    global_gene_mean = expr_mat.mean(axis=0)
    # Observed score
    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        cluster_mean_matrix[idx] = expr_mat[cell_labels == c].mean(axis=0)
    obs_score = np.sum((cluster_mean_matrix - global_gene_mean) ** 2, axis=0) / n_cluster
    # Permutation null
    perm_score_set = np.zeros((perm_times, n_genes), dtype=np.float32)
    for p in range(perm_times):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_cluster_mean = np.zeros((n_cluster, n_genes), dtype=np.float32)
        for idx, c in enumerate(unique_clusters):
            perm_cluster_mean[idx] = perm_expr[cell_labels == c].mean(axis=0)
        perm_score_set[p] = np.sum((perm_cluster_mean - global_gene_mean) ** 2, axis=0) / n_cluster
    raw_p = (np.sum(perm_score_set >= obs_score, axis=0) + 1) / (perm_times + 1)
    sig_flag, fdr_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
    svg_result_df = pd.DataFrame({
        "gene_name": adata.var.index.values,
        "SpaRSE_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set, expr_mat
    gc.collect()
    return svg_result_df.sort_values("SpaRSE_SVG_score", ascending=False).reset_index(drop=True)

# --------------------------
# SpaRSE embedding + clustering + SVG pipeline
# --------------------------
def run_SpaRSE_svg(adata_raw, n_cluster, epochs=200):
    adata = adata_raw.copy()
    # Preprocessing (matches original SpaRSE)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    if 'log1p' not in adata.uns:
        sc.pp.log1p(adata)
    sc.pp.scale(adata)
    # Dynamic PCA components (avoid exceeding sample/gene count)
    max_pca = min(200, adata.n_obs - 1, adata.n_vars - 1)
    if max_pca < 1:
        adata.obsm['X_pca'] = adata.X.copy()
        max_pca = adata.n_vars
    else:
        sc.pp.pca(adata, n_comps=max_pca, svd_solver='arpack')
    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)
    # Model & Training
    model = SpaRSE(input_dim=max_pca).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X).cpu().numpy()
    del model, X
    torch.cuda.empty_cache()
    # Clustering on embedding
    embed_adata = anndata.AnnData(embedding)
    sc.pp.neighbors(embed_adata, n_neighbors=20)
    res_opt = res_search_fixed_clus(embed_adata, n_cluster)
    sc.tl.leiden(embed_adata, resolution=res_opt, key_added="SpaRSE_leiden")
    adata_raw.obs["SpaRSE_leiden"] = embed_adata.obs["SpaRSE_leiden"].values
    adata_raw.obsm["SpaRSE_embedding"] = embedding
    del adata, embed_adata
    gc.collect()
    return adata_raw

# --------------------------
# Global Configuration (7 real datasets, actual filenames)
# --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\SpaRSE"

DATA_FILE_LIST = [
    "10X_Visium_Lung.h5ad",
    "DLPFC_151507.h5ad",
    "Slide-seq2.h5ad",
    "SNARE-seq2.h5ad",
    "ST.h5ad",
    "STARmap.h5ad",
    "Xenium_BC.h5ad"
]

CLUSTER_NUM_DICT = {
    "10X_Visium_Lung": 6,
    "DLPFC_151507": 7,
    "Slide-seq2": 7,
    "SNARE-seq2": 5,
    "ST": 4,
    "STARmap": 3,
    "Xenium_BC": 8
}

PLATFORM_LABEL = {
    "10X_Visium_Lung.h5ad": "10X Visium",
    "DLPFC_151507.h5ad": "DLPFC",
    "Slide-seq2.h5ad": "Slide-seq2",
    "SNARE-seq2.h5ad": "SNARE-seq2",
    "ST.h5ad": "ST",
    "STARmap.h5ad": "STARmap",
    "Xenium_BC.h5ad": "Xenium"
}

# --------------------------
# Main Execution
# --------------------------
if __name__ == '__main__':
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    global_summary = []
    print("=" * 80)
    print("SpaRSE Clustering + Cluster-based SVG Identification")
    print(f"Input : {DATA_INPUT_DIR}")
    print(f"Output: {OUTPUT_ROOT_DIR}")
    print("=" * 80)
    for file_name in DATA_FILE_LIST:
        file_full_path = os.path.join(DATA_INPUT_DIR, file_name)
        dataset_name = file_name.replace(".h5ad", "")
        dataset_save_dir = os.path.join(OUTPUT_ROOT_DIR, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        print(f"\n==== Processing: {dataset_name} ====")
        if not os.path.exists(file_full_path):
            print(f"  WARNING: File missing -> {file_full_path}, skipped")
            continue
        try:
            adata = sc.read_h5ad(file_full_path)
            adata.var_names_make_unique()
            target_cluster = CLUSTER_NUM_DICT[dataset_name]
            print(f"  Target clusters: {target_cluster}")
            clustered_adata = run_SpaRSE_svg(adata, n_cluster=target_cluster, epochs=200)
            svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="SpaRSE_leiden",
                                           perm_times=1000, seed=0)
            # Save outputs
            clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"SpaRSE_{dataset_name}_clustered.h5ad"))
            svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_SpaRSE_SVG_full_list.csv"), index=False)
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
        except Exception as e:
            print(f"  FAILED {dataset_name}: {str(e)}")
    if len(global_summary) > 0:
        pd.DataFrame(global_summary).to_csv(
            os.path.join(OUTPUT_ROOT_DIR, "SpaRSE_All_Datasets_Global_Summary.csv"),
            index=False
        )
        print("\n" + "=" * 80)
        print("ALL DATASETS COMPLETED")
        print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
        print("=" * 80)
    else:
        print("\nNo datasets were successfully processed. Check file paths.")