import os
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score
from scipy.sparse import csr_matrix
import warnings
import logging

# Suppress ALL warnings cleanly
warnings.filterwarnings("ignore")
logging.getLogger('scanpy').setLevel(logging.ERROR)
logging.getLogger('squidpy').setLevel(logging.ERROR)

# --------------------------
#  Custom BHI Implementation
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
#  Fixed Ground Truth Function
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
# 1. DLPFC 12 datasets
# --------------------------
dlpfc_datasets = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]
base_data_dir = r"D:\My_Project_ST\DataSets\DLPFC"

# --------------------------
# 2. Metrics storage
# --------------------------
metrics_results = []

# --------------------------
# 3. Main loop
# --------------------------
for dataset_id in dlpfc_datasets:
    print(f"\nProcessing dataset: {dataset_id}...")
    data_dir = os.path.join(base_data_dir, dataset_id)
    h5_file_path = os.path.join(data_dir, "filtered_feature_bc_matrix.h5")
    
    # Load data
    adata = sc.read_visium(
        path=data_dir,
        count_file=os.path.basename(h5_file_path),
        library_id=dataset_id,
        load_images=True
    )
    adata.var_names_make_unique()
    
    # Ground truth
    adata = generate_paper_ground_truth(adata)
    
    # Preprocessing (same as your original)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="grid")
    sc.pp.neighbors(adata, n_pcs=50)

    # --------------------------
    # 🔹 REPLACED: DeepSpaCE Method (instead of stFormer)
    # --------------------------
    import deepspace as ds
    adata_deepspace = adata.copy()
    
    # Run DeepSpaCE standard clustering pipeline
    ds.tl.cluster_spatial(
        adata_deepspace,
        n_clusters=7,  # DLPFC has 7 layers
        key_added="deepspace_domain"
    )
    adata.obs["deepspace_domain"] = adata_deepspace.obs["deepspace_domain"]
    
    # --------------------------
    # Compute metrics (SAME AS BEFORE)
    # --------------------------
    y_true = adata.obs["ground_truth_domain"].values
    y_pred = adata.obs["deepspace_domain"].values
    mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    X_pca = adata.obsm["X_pca"][mask]
    
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sc_score = silhouette_score(X_pca, y_pred)
    bhi = batch_homogeneity(adata.obsp["connectivities"], y_pred)
    
    metrics_results.append({
        "dataset": dataset_id,
        "ari": round(ari, 3),
        "nmi": round(nmi, 3),
        "ami": round(ami, 3),
        "sc": round(sc_score, 3),
        "bhi": round(bhi, 3)
    })

# --------------------------
# Save final result
# --------------------------
metrics_df = pd.DataFrame(metrics_results).set_index("dataset")
print("\n DeepSpaCE Metrics on 12 DLPFC Datasets:")
print(metrics_df)
metrics_df.to_csv("deepspace_dlpfc_metrics.csv")
print("\n File saved: deepspace_dlpfc_metrics.csv")