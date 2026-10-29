import sys
import os
import warnings
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

# --------------------------
# Basic Setup
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# Original STAlign Model (UNCHANGED ARCHITECTURE)
# --------------------------
class STAlign(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32):
        super(STAlign, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# --------------------------
# Fixed Core Function: STAlign Embedding + Clustering + Cluster-based SVG Detection
# --------------------------
def run_STAlign_SVG(adata_h5, save_path, dataset_name, epochs=200):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()

    # Preprocessing
    sc.pp.filter_genes(adata, min_cells=5)
    # INPUT DATA IS ALREADY LOG NORMALIZED — skip redundant transform
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    # Safe PCA dimension handling
    max_possible = min(adata.n_obs, adata.n_vars)
    n_comps = min(200, max_possible - 1)
    if n_comps < 1:
        adata.obsm['X_pca'] = adata.X.copy()
    else:
        sc.pp.pca(adata, n_comps=n_comps, svd_solver='arpack')

    # Prepare tensor for model
    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    # Train STAlign Autoencoder
    model = STAlign(input_dim=n_comps).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"==== Training STAlign for {dataset_name} ====")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Extract latent embedding
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X).cpu().numpy()

    # Leiden clustering on STAlign latent embedding
    adata_emb = anndata.AnnData(embedding)
    sc.pp.neighbors(adata_emb, n_neighbors=20, random_state=0)
    sc.tl.leiden(adata_emb, key_added="STAlign_cluster", resolution=0.8, random_state=0)
    cluster_labels = adata_emb.obs["STAlign_cluster"].values

    # --------------------------
    # FIX: Use adata.X AFTER gene filtering (critical fix for dimension mismatch)
    # --------------------------
    gene_names = adata.var_names.tolist()
    expr_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    p_values = []

    # ANOVA test per gene across clusters
    for gene_idx in range(expr_matrix.shape[1]):
        gene_expr = expr_matrix[:, gene_idx]
        cluster_groups = [gene_expr[cluster_labels == clu] for clu in np.unique(cluster_labels)]
        stat, p = f_oneway(*cluster_groups)
        p_values.append(p)

    # FDR correction (Benjamini–Hochberg)
    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    # Build result dataframe
    svg_result_df = pd.DataFrame({
        "gene": gene_names,
        "p_value": p_values,
        "p_adj_FDR": p_adj,
        "is_SVG_FDR0.05": reject
    })

    # Save SVG result CSV
    output_csv = os.path.join(save_path, f"{dataset_name}_STAlign_Cluster_SVG_Result.csv")
    svg_result_df.to_csv(output_csv, index=False)

    sig_svg_count = np.sum(reject)
    print(f"✅ {dataset_name} finished | Total Significant SVGs (FDR<0.05): {sig_svg_count}\n")
    return {"dataset": dataset_name, "significant_SVG_count": sig_svg_count}

# --------------------------
# Main Batch Runner
# --------------------------
if __name__ == '__main__':
    # Input / Output Paths
    INPUT_DIR  = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
    OUTPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\STAlign"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all standardized h5ad files
    h5ad_file_list = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith("_standardized.h5ad")
    ]
    print(f"Detected {len(h5ad_file_list)} standardized datasets to process\n")

    batch_summary = []
    for filename in h5ad_file_list:
        dataset_name = filename.replace("_standardized.h5ad", "")
        full_h5ad_path = os.path.join(INPUT_DIR, filename)
        dataset_output_folder = os.path.join(OUTPUT_DIR, dataset_name)

        print(f"==================================================")
        print(f"            PROCESS DATASET: {dataset_name}")
        print(f"==================================================")

        try:
            adata = anndata.read_h5ad(full_h5ad_path)
            res = run_STAlign_SVG(adata, dataset_output_folder, dataset_name, epochs=200)
            batch_summary.append(res)
        except Exception as err:
            print(f"❌ FAILED: {dataset_name}, Error message: {str(err)}\n")
            batch_summary.append({"dataset": dataset_name, "significant_SVG_count": "Failed"})

    # Save global batch summary table
    summary_df = pd.DataFrame(batch_summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "STAlign_Batch_SVG_Summary.csv"), index=False)
    print("\n==== ALL DATASETS PROCESSING FINISHED ====")
    print(f"All SVG results saved to: {OUTPUT_DIR}")
    print(summary_df)