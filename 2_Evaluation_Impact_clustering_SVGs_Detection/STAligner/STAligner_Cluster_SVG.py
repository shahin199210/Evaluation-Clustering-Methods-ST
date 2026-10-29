# ========== 1. Global Thread & Memory Lock ==========
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTHONUTF8"] = "1"

# ========== 2. Unified Global Imports ==========
import sys
import time
import psutil
import tracemalloc
import warnings
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, silhouette_score
from scipy.stats import f_oneway

# Fix IO encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ========== 3. Fixed Global Path & Hyperparameters (Only STAligner folder reserved) ==========
STD_DATA_DIR = Path(r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized")
OUTPUT_DIR = Path(r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_method")
STALIGNER_OUT_DIR = OUTPUT_DIR / "STAligner"
SPARKX_INPUT_DIR = OUTPUT_DIR / "SparkX_Input_Files"

# Create only needed directories
for folder in [OUTPUT_DIR, STALIGNER_OUT_DIR, SPARKX_INPUT_DIR]:
    folder.mkdir(exist_ok=True)

# Shared Hyperparameters
FDR_ALPHA = 0.05
WORKER_NUM = 1
STALIGNER_EPOCHS = 200
INCREMENT_RES = 0.02
DEFAULT_CLUSTER_NUM = 10

# ========== 4. Spark-X Input Export Function (Reserved as required) ==========
def export_sparkx_input(adata, dataset_name):
    expr_mat = adata.X
    if hasattr(expr_mat, "toarray"):
        expr_df = pd.DataFrame(expr_mat.T.toarray(), index=adata.var_names, columns=adata.obs_names)
    else:
        expr_df = pd.DataFrame(expr_mat.T, index=adata.var_names, columns=adata.obs_names)
    coord_df = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
    expr_df.to_csv(SPARKX_INPUT_DIR / f"{dataset_name}_expr.csv", index=False)
    coord_df.to_csv(SPARKX_INPUT_DIR / f"{dataset_name}_coord.csv", index=False)

# ========== 5. Complete STAligner Model & Auxiliary Functions (Original code reserved) ==========
class STAligner(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, hidden_dim=128):
        super(STAligner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

def res_search_fixed_clus(adata, fixed_clus_count, increment=INCREMENT_RES):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

def eval_model(pred, gt):
    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, nmi, ami

# ========== 6. STAligner Core Train Function ==========
def run_staligner_train(adata_raw, n_clusters):
    adata = adata_raw.copy()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    max_possible = min(adata.n_obs, adata.n_vars)
    n_comps = min(200, max_possible - 1)
    if n_comps < 1:
        adata.obsm['X_pca'] = adata.X.copy()
    else:
        sc.pp.pca(adata, n_comps=n_comps, svd_solver='arpack')

    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)
    model = STAligner(input_dim=n_comps).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    for epoch in range(STALIGNER_EPOCHS):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X).cpu().numpy()

    adata_emb = anndata.AnnData(embedding)
    adata_emb.obsm['spatial'] = adata_raw.obsm['spatial']
    sc.pp.neighbors(adata_emb, n_neighbors=20, random_state=0)
    sc.tl.umap(adata_emb, random_state=0)
    res = res_search_fixed_clus(adata_emb, n_clusters)
    sc.tl.leiden(adata_emb, key_added="STAligner_leiden", resolution=res, random_state=0)

    cluster_label = adata_emb.obs["STAligner_leiden"].astype(str)
    return cluster_label, adata

# ========== 7. Core STAligner SVG Definition: ANOVA-based cross-cluster differential genes ==========
def calculate_staligner_cluster_svg(adata_raw, cluster_label):
    """
    Compute ANOVA p-values for each gene across STAligner clusters.
    adata_raw: AnnData object (cells × genes)
    cluster_label: 1D array-like of length n_cells (cluster assignments)
    Returns DataFrame with gene-wise statistics.
    """
    # Ensure cluster_label is a numpy array
    if isinstance(cluster_label, pd.Series):
        cluster_label = cluster_label.values
    cluster_label = np.asarray(cluster_label)

    # Expression matrix: cells × genes
    X = adata_raw.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_cells, n_genes = X.shape

    # Validate lengths
    if len(cluster_label) != n_cells:
        raise ValueError(
            f"cluster_label length ({len(cluster_label)}) does not match "
            f"number of cells in adata ({n_cells})"
        )

    gene_names = adata_raw.var_names
    unique_clusters = np.unique(cluster_label)
    pval_list = []

    # Loop over genes (columns)
    for gene_idx in range(n_genes):
        gene_expr = X[:, gene_idx]  # 1D array of length n_cells
        # Split by cluster
        group_expr = [gene_expr[cluster_label == c] for c in unique_clusters]
        # Perform one-way ANOVA
        _, p_val = f_oneway(*group_expr)
        pval_list.append(p_val)

    # Multiple testing correction
    raw_p = np.array(pval_list)
    reject, padj_bh, _, _ = multipletests(raw_p, alpha=FDR_ALPHA, method="fdr_bh")

    # Build result DataFrame
    res_df = pd.DataFrame(index=gene_names)
    res_df["gene_ids"] = res_df.index
    res_df["Cluster_ANOVA_pval"] = raw_p
    res_df["FDR_BH_padj"] = padj_bh
    res_df["is_SVG_FDR0.05"] = reject
    res_df["feature_type"] = "GeneExpr"
    res_df["genome"] = "GRCh38"
    res_df["svg_method"] = "STAligner_Cluster_ANOVA"

    # Mitochondrial flag (if present)
    if "mito" in adata_raw.var.columns:
        mito_series = adata_raw.var["mito"].reindex(res_df.index, fill_value=False)
    else:
        mito_series = pd.Series(False, index=res_df.index)
    res_df["mito"] = mito_series

    return res_df

def staligner_svg_pipeline(adata, dataset_name, platform_tag, n_clusters=10):
    cluster_label, adata_proc = run_staligner_train(adata, n_clusters)
    # Extra validation: cluster_label length must equal number of cells
    if len(cluster_label) != adata.n_obs:
        raise ValueError(f"Cluster label length mismatch: {len(cluster_label)} vs {adata.n_obs}")
    svg_df = calculate_staligner_cluster_svg(adata, cluster_label)
    svg_df["gene"] = svg_df["gene_ids"]
    svg_df["dataset"] = dataset_name
    svg_df["platform"] = platform_tag
    svg_df["n_cluster"] = n_clusters
    output_columns = [
        "gene_ids", "gene", "Cluster_ANOVA_pval", "FDR_BH_padj",
        "is_SVG_FDR0.05", "feature_type", "genome", "mito", "dataset", "platform", "svg_method", "n_cluster"
    ]
    return svg_df[output_columns]

# ========== 8. Main Unified Execution Pipeline (Only STAligner task reserved) ==========
if __name__ == "__main__":
    h5ad_file_list = sorted(STD_DATA_DIR.glob("*_standardized.h5ad"))
    if len(h5ad_file_list) == 0:
        print(f"No standardized h5ad found in {STD_DATA_DIR}")
        exit()

    total_file_num = len(h5ad_file_list)
    success_count_staligner = 0
    failed_records = []

    for index, file_path in enumerate(h5ad_file_list):
        file_full_name = file_path.name
        dataset_id = file_path.stem.replace("_standardized", "")
        print(f"\n====== [{index+1}/{total_file_num}] Processing Dataset: {dataset_id} ======")

        try:
            adata = sc.read_h5ad(file_path)
            # Platform identification
            if "Xenium" in dataset_id:
                platform = "Xenium"
            elif "Visium" in dataset_id or "10X" in dataset_id:
                platform = "Visium"
            elif "STARmap" in dataset_id:
                platform = "STARmap"
            elif "CosMx" in dataset_id:
                platform = "CosMx"
            else:
                platform = "Unknown"

            # Step1: Export SparkX input files
            export_sparkx_input(adata, dataset_id)

            # Step2: Only run STAligner clustering + SVG identification
            st_result = staligner_svg_pipeline(adata, dataset_id, platform, n_clusters=DEFAULT_CLUSTER_NUM)
            st_save = STALIGNER_OUT_DIR / f"{dataset_id}_STAligner_ClusterSVG.csv"
            st_result.to_csv(st_save, index=False)
            st_svg_count = st_result["is_SVG_FDR0.05"].sum()
            success_count_staligner += 1
            print(f"[STAligner] Saved file: {st_save.name} | Total genes:{len(st_result)}, Significant SVG count:{st_svg_count}")

        except Exception as err:
            err_msg = str(err)
            print(f"Failed {file_full_name}: {err_msg}")
            failed_records.append((file_full_name, err_msg))
        finally:
            if "adata" in locals():
                del adata
            gc.collect()

    # Final summary
    print("\n==================== Task Summary ====================")
    print(f"Total datasets: {total_file_num}")
    print(f"STAligner SVG identification Success: {success_count_staligner}")
    print(f"Failed jobs: {len(failed_records)}")
    if failed_records:
        print("\nFailed list details:")
        for name, e in failed_records:
            print(f"- {name}: {e}")