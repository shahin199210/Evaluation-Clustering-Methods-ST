import os
import time
import psutil
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import torch
# Import local DSTN
from DSTN.model import DSTN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score
from scipy.sparse import csr_matrix
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger('scanpy').setLevel(logging.ERROR)
logging.getLogger('squidpy').setLevel(logging.ERROR)

# --------------------------
# Your BHI Metric
# --------------------------
def batch_homogeneity(connectivities, y_pred):
    y_pred = np.array(y_pred)
    n_spots = len(y_pred)
    bhi_score = 0.0
    for cluster_id in np.unique(y_pred):
        cluster_mask = y_pred == cluster_id
        cluster_size = np.sum(cluster_mask)
        if cluster_size <= 1: continue
        total_edges = connectivities[cluster_mask].sum()
        within_edges = connectivities[cluster_mask][:, cluster_mask].sum()
        if total_edges > 0:
            bhi_score += (within_edges / total_edges) * (cluster_size / n_spots)
    return bhi_score

# --------------------------
# Ground Truth
# --------------------------
def generate_paper_ground_truth(adata):
    y_coords = adata.obsm["spatial"][:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    bounds = [y_min + (y_max-y_min)*x for x in [0.08, 0.22, 0.36, 0.5, 0.68, 0.85, 0.95]]
    gt = []
    for y in y_coords:
        if y < bounds[0]: gt.append(0)
        elif y < bounds[1]: gt.append(1)
        elif y < bounds[2]: gt.append(2)
        elif y < bounds[3]: gt.append(3)
        elif y < bounds[4]: gt.append(4)
        elif y < bounds[5]: gt.append(5)
        elif y < bounds[6]: gt.append(6)
        else: gt.append(7)
    adata.obs["ground_truth_domain"] = pd.Categorical(gt)
    return adata

# --------------------------
# RUN DSTN (Official Model)
# --------------------------
def run_dstn(adata, n_clusters=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare PCA + Spatial coords
    X = torch.tensor(adata.obsm["X_pca"], dtype=torch.float32)
    coords = torch.tensor(adata.obsm["spatial"], dtype=torch.float32)
    
    # Initialize DSTN
    model = DSTN(input_dim=50, hidden_dim=128, output_dim=n_clusters, n_layers=2)
    model = model.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        emb, pred = model(X, coords)
    
    # Save clusters
    adata.obs["dstn_domain"] = pred.cpu().numpy().astype(str)
    return adata

# --------------------------
# Datasets & Main Loop
# --------------------------
dlpfc_datasets = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]
base_data_dir = r"D:\My_Project_ST\DataSets\DLPFC"
metrics_results = []

for dataset_id in dlpfc_datasets:
    print(f"\nProcessing {dataset_id}...")
    data_dir = os.path.join(base_data_dir, dataset_id)
    
    start_time = time.time()
    start_mem = get_peak_memory_gb() # You'll need to re-add this helper function
    
    # Load data
    adata = sc.read_visium(path=data_dir, count_file="filtered_feature_bc_matrix.h5", library_id=dataset_id)
    adata.var_names_make_unique()
    
    # Ground truth & preprocessing
    adata = generate_paper_ground_truth(adata)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata, n_comps=50)
    sq.gr.spatial_neighbors(adata, n_neighs=6)
    sc.pp.neighbors(adata, n_pcs=50)

    # RUN DSTN
    adata = run_dstn(adata, n_clusters=7)
    
    # Time & memory
    end_time = time.time()
    runtime = end_time - start_time
    peak_mem = get_peak_memory_gb() # Re-add this helper function
    
    # Metrics
    y_true = adata.obs["ground_truth_domain"].values
    y_pred = adata.obs["dstn_domain"].values
    mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
    X_pca = adata.obsm["X_pca"][mask]
    
    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    ami = adjusted_mutual_info_score(y_true[mask], y_pred[mask])
    sc_score = silhouette_score(X_pca, y_pred[mask])
    bhi = batch_homogeneity(adata.obsp["connectivities"], y_pred[mask])
    
    # Save
    metrics_results.append({
        "dataset": dataset_id, "ari": round(ari,3), "nmi": round(nmi,3),
        "ami": round(ami,3), "sc": round(sc_score,3), "bhi": round(bhi,3),
        "time(s)": round(runtime,2), "Mem(GB)": round(peak_mem,4),
        "Mem_peak(GB)": round(peak_mem,4), "round": 1
    })

# Save CSV
metrics_df = pd.DataFrame(metrics_results).set_index("dataset")
print("\n✅ DSTN Results:")
print(metrics_df)
metrics_df.to_csv("dstn_dlpfc_results.csv")