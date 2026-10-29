import os
import pickle
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def preprocess_visium(dataset_path, dataset_name, save_dir):
    """
    Preprocess raw 10X Visium/DLPFC data into CCST-compatible format
    Args:
        dataset_path: Path to the raw data folder (e.g., D:\My_Project_ST\DataSets\DLPFC\151507)
        dataset_name: Name of the dataset (e.g., "151507")
        save_dir: Path to save preprocessed files (e.g., D:\My_Project_ST\DataSets\CCST_Preprocessed\DLPFC\151507)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load raw Visium data
    adata = sc.read_10x_mtx(
        os.path.join(dataset_path, "filtered_feature_bc_matrix"),
        gex_only=True
    )
    spatial_path = os.path.join(dataset_path, "spatial", "tissue_positions_list.csv")
    spatial = pd.read_csv(spatial_path, header=None, index_col=0)
    spatial.columns = ['in_tissue', 'row', 'col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    spatial = spatial.loc[adata.obs_names]
    adata.obsm['spatial'] = spatial[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values

    # Normalize data
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    # Compute spatial adjacency matrix (k-nearest neighbors)
    from sklearn.neighbors import NearestNeighbors
    coords = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    adj = np.zeros((adata.n_obs, adata.n_obs))
    for i in range(adata.n_obs):
        adj[i, indices[i]] = 1
    adj = csr_matrix(adj)

    # Save files in CCST format
    with open(os.path.join(save_dir, f"{dataset_name}Adjacent"), 'wb') as f:
        pickle.dump(adj, f)
    with open(os.path.join(save_dir, f"{dataset_name}Feature"), 'wb') as f:
        pickle.dump(adata.X, f)
    # If labels exist (DLPFC), save them
    if os.path.exists(os.path.join(dataset_path, "metadata.tsv")):
        metadata = pd.read_csv(os.path.join(dataset_path, "metadata.tsv"), sep='\t', index_col=0)
        labels = metadata['layer'].values
        with open(os.path.join(save_dir, f"{dataset_name}Label"), 'wb') as f:
            pickle.dump(labels, f)
    print(f"Preprocessed {dataset_name} saved to {save_dir}")

if __name__ == "__main__":
    # Process all DLPFC datasets
    base_data_path = r"D:\My_Project_ST\DataSets\DLPFC"
    save_base_dir = r"D:\My_Project_ST\DataSets\CCST_Preprocessed\DLPFC"
    datasets = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']

    for dataset in datasets:
        dataset_path = os.path.join(base_data_path, dataset)
        save_dir = os.path.join(save_base_dir, dataset)
        preprocess_visium(dataset_path, dataset, save_dir)