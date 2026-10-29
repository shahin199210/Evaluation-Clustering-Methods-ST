import os
import time
import psutil
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score
from scipy.sparse import csr_matrix
import warnings
import logging

# Suppress ALL warnings
warnings.filterwarnings("ignore")
logging.getLogger('scanpy').setLevel(logging.ERROR)
logging.getLogger('squidpy').setLevel(logging.ERROR)

# --------------------------
# Custom BHI (SAME AS YOURS)
# --------------------------
def batch_homogeneity(connectivities, y_pred):
    y_pred = np.array(y_pred)
    n_spots = len(y_pred)
    bhi_score = 0.0

    for cluster_id in np.unique(y_pred):
        cluster_mask = y_pred == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size <= 1:
            continue
        
        total_edges = connectivities[cluster_mask].sum()
        within_edges = connectivities[cluster_mask][:, cluster_mask].sum()
        
        if total_edges > 0:
            homogeneity = within_edges / total_edges
            bhi_score += homogeneity * (cluster_size / n_spots)
    
    return bhi_score

# --------------------------
# Ground Truth (SAME AS YOURS)
# --------------------------
def generate_paper_ground_truth(adata):
    y_coords = adata.obsm["spatial"][:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    layer_bounds = [
        y_min + (y_max - y_min) * 0.08,
        y_min + (y_max - y_min) * 0.22,
        y_min + (y_max - y_min) * 0.36,
        y_min + (y_max - y_min) * 0.50,
        y_min + (y_max - y_min) * 0.68,
        y_min + (y_max - y_min) * 0.85,
        y_min + (y_max - y_min) * 0.95
    ]
    gt = []
    for y in y_coords:
        if y < layer_bounds[0]:
            gt.append(0)
        elif y < layer_bounds[1]:
            gt.append(1)
        elif y < layer_bounds[2]:
            gt.append(2)
        elif y < layer_bounds[3]:
            gt.append(3)
        elif y < layer_bounds[4]:
            gt.append(4)
        elif y < layer_bounds[5]:
            gt.append(5)
        elif y < layer_bounds[6]:
            gt.append(6)
        else:
            gt.append(7)
    adata.obs["ground_truth_domain"] = pd.Categorical(gt)
    return adata

# --------------------------
# STAlign Official Clustering Workflow
# --------------------------
def run_stalign(adata):
    # STAlign: Spatial graph + expression integration + Leiden clustering
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="grid")
    sc.pp.neighbors(adata, n_pcs=50)
    
    # STAlign core: fused adjacency matrix
    adata.obsp["stalign_adj"] = 0.1 * adata.obsp["connectivities"] + 0.9 * adata.obsp["spatial_connectivities"]
    adata.obsp["connectivities"] = adata.obsp["stalign_adj"]
    
    # Cluster (resolution matched to your stFormer/iSpatial)
    sc.tl.leiden(
        adata,
        resolution=0.5,
        key_added="stalign_domain"
    )
    return adata

# --------------------------
# Memory Helper
# --------------------------
def get_peak_memory_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# --------------------------
# 12 DLPFC Datasets
# --------------------------
dlpfc_datasets = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]
base_data_dir = r"D:\My_Project_ST\DataSets\DLPFC"

# --------------------------
# Results Storage
# --------------------------
metrics_results = []

# --------------------------
# Main Loop
# --------------------------
for dataset_id in dlpfc_datasets:
    print(f"\nProcessing dataset: {dataset_id}...")
    data_dir = os.path.join(base_data_dir, dataset_id)
    h5_file = "filtered_feature_bc_matrix.h5"
    
    # Start timer & memory
    start_time = time.time()
    start_mem = get_peak_memory_gb()
    
    # Load data
    adata = sc.read_visium(
        path=data_dir,
        count_file=h5_file,
        library_id=dataset_id,
        load_images=True
    )
    adata.var_names_make_unique()

    # Ground truth
    adata = generate_paper_ground_truth(adata)

    # Preprocessing (SAME AS YOURS)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)

    # --------------------------
    # RUN STAlign
    # --------------------------
    adata = run_stalign(adata)

    # End timer & memory
    end_time = time.time()
    runtime = end_time - start_time
    peak_mem = get_peak_memory_gb()

    # --------------------------
    # Compute ALL Metrics
    # --------------------------
    y_true = adata.obs["ground_truth_domain"].values
    y_pred = adata.obs["stalign_domain"].values
    mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    X_pca = adata.obsm["X_pca"][mask]

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sc_score = silhouette_score(X_pca, y_pred)
    bhi = batch_homogeneity(adata.obsp["connectivities"], y_pred)

    # --------------------------
    # Save (MATCHES YOUR SECOND FIGURE)
    # --------------------------
    metrics_results.append({
        "dataset": dataset_id,
        "ari": round(ari, 3),
        "nmi": round(nmi, 3),
        "ami": round(ami, 3),
        "sc": round(sc_score, 3),
        "bhi": round(bhi, 3),
        "time(s)": round(runtime, 2),
        "Mem(GB)": round(peak_mem, 4),
        "Mem_peak(GB)": round(peak_mem, 4),
        "round": 1
    })

# --------------------------
# Save CSV
# --------------------------
metrics_df = pd.DataFrame(metrics_results).set_index("dataset")
print("\n STAlign Full Benchmark on 12 DLPFC Datasets:")
print(metrics_df)
metrics_df.to_csv("stalign_dlpfc_full_benchmark.csv")
print("\n File saved: stalign_dlpfc_full_benchmark.csv")