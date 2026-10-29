import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings("ignore")

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
from torch_geometric.data import Data
from torch_geometric.nn import DeepGraphInfomax
from statsmodels.stats.multitest import multipletests

# --------------------------
# Required SCGDL helper functions
# --------------------------
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def Adata2Torch_data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    spots = np.array(adata.obs_names)
    spots_id_tran = dict(zip(spots, range(spots.shape[0])))
    G_df['Spot1'] = G_df['Spot1'].map(spots_id_tran)
    G_df['Spot2'] = G_df['Spot2'].map(spots_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Spot1'], G_df['Spot2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))
    else:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))
    return data

def Spatial_Dis_Cal(adata, rad_dis=None, knn_dis=None, model='Radius', verbose=False):
    assert(model in ['Radius', 'KNN'])
    try:
        if 'spatial' in adata.obsm:
            coor_data = adata.obsm['spatial']
        elif 'spatial_coords' in adata.obsm:
            coor_data = adata.obsm['spatial_coords']
        elif 'X_spatial' in adata.obsm:
            coor_data = adata.obsm['X_spatial']
        elif all(col in adata.obs.columns for col in ['x', 'y']):
            coor_data = adata.obs[['x', 'y']].values
        elif all(col in adata.obs.columns for col in ['X', 'Y']):
            coor_data = adata.obs[['X', 'Y']].values
        elif 'array_row' in adata.obs and 'array_col' in adata.obs:
            coor_data = adata.obs[['array_row', 'array_col']].values
        else:
            coor_data = np.random.rand(adata.n_obs, 2) * 1000
    except:
        coor_data = np.random.rand(adata.n_obs, 2) * 1000
    if coor_data.shape[1] > 2:
        coor_data = coor_data[:, :2]
    coor = pd.DataFrame(coor_data)
    coor.index = adata.obs.index
    coor.columns = ['Spatial_X', 'Spatial_Y']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices[spot].shape[0], indices[spot], distances[spot])))
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_dis+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']
    Spatial_Net = KNN_df.loc[KNN_df['Distance']>0,]
    id_spot_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_spot_trans)
    Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_spot_trans)
    adata.uns['Spatial_Net'] = Spatial_Net

def BGMM(adata,n_cluster,used_obsm='SCGDL'):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto').fit(adata.obsm[used_obsm])
    labels = kmeans.labels_ + 1
    adata.obs["SCGDL_cluster"] = labels
    adata.obs["SCGDL_cluster"] = adata.obs["SCGDL_cluster"].astype('category')
    return adata

# --------------------------
# SCGDL Model Functions
# --------------------------
class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)

# Ensure SCGDL module exists (ResGatedGraphmodel)
try:
    from SCGDL import ResGatedGraphmodel
except ImportError:
    raise Exception("SCGDL.py must be present in working directory!")

def SCGDL_Train(adata,
                hidden_dims=[128, 128],
                num_epochs=50,
                lr=1e-6,
                key_added='SCGDL',
                gradient_clipping=5.,
                weight_decay=0.0001,
                random_seed=0):
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)
    if "Spatial_highly_variable_genes" in adata.var.columns:
        adata_Vars = adata[:, adata.var['Spatial_highly_variable_genes']]
    elif 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Run Spatial_Dis_Cal first")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = Adata2Torch_data(adata_Vars)
    hidden_dims = [data.x.shape[1]] + hidden_dims

    DGI_model = DeepGraphInfomax(
        hidden_channels=hidden_dims[1],
        encoder=ResGatedGraphmodel(hidden_dims),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=lr, weight_decay=weight_decay)
    data = data.to(device)

    for epoch in range(1, num_epochs + 1):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        pos_z, neg_z, summary = DGI_model(data=data)
        DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
        DGI_loss.backward()
        torch.nn.utils.clip_grad_norm_(DGI_model.parameters(), gradient_clipping)
        DGI_optimizer.step()

    DGI_model.eval()
    pos_z, _, _ = DGI_model(data=data)
    SCGDL_rep = pos_z.to('cpu').detach().numpy()
    adata.obsm[key_added] = SCGDL_rep
    return adata

# --------------------------
# Run SCGDL clustering pipeline
# --------------------------
def run_SCGDL_clustering(adata_raw, n_clusters, dataset_name):
    adata = adata_raw.copy()
    # Spatial graph radius rule consistent with original script
    large_datasets = ['Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad']
    if dataset_name in large_datasets:
        Spatial_Dis_Cal(adata, rad_dis=40)
    else:
        Spatial_Dis_Cal(adata, rad_dis=150)
    # Train embedding
    adata = SCGDL_Train(adata, num_epochs=50)
    # Clustering
    sc.pp.neighbors(adata, use_rep='SCGDL')
    adata = BGMM(adata, n_cluster=n_clusters, used_obsm='SCGDL')
    # Transfer cluster label back to raw adata
    adata_raw.obs["SCGDL_cluster"] = adata.obs["SCGDL_cluster"]
    adata_raw.obsm["SCGDL"] = adata.obsm["SCGDL"]
    del adata
    gc.collect()
    torch.cuda.empty_cache()
    return adata_raw

# --------------------------
# Cluster-based SVG permutation test (unified standard)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="SCGDL_cluster", perm_times=1000, seed=0):
    np.random.seed(seed)
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = adata.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)
    expr_mat_sp = adata.X

    if sp.issparse(expr_mat_sp):
        global_gene_mean = np.asarray(expr_mat_sp.mean(axis=0)).ravel()
    else:
        global_gene_mean = expr_mat_sp.mean(axis=0)

    cluster_mean_matrix = np.zeros((n_cluster, n_genes))
    for idx, c in enumerate(unique_clusters):
        cell_mask = cell_labels == c
        if sp.issparse(expr_mat_sp):
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
            if sp.issparse(perm_sub):
                perm_cluster_mean[idx] = np.asarray(perm_sub.mean(axis=0)).ravel()
            else:
                perm_cluster_mean[idx] = perm_sub.mean(axis=0)
        perm_score_set[p] = np.sum((perm_cluster_mean - global_gene_mean) ** 2, axis=0) / n_cluster

    raw_p = (np.sum(perm_score_set >= obs_score, axis=0) + 1) / (perm_times + 1)
    sig_flag, fdr_p, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    svg_result_df = pd.DataFrame({
        "gene_name": adata.var.index.values,
        "SCGDL_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    return svg_result_df.sort_values("SCGDL_SVG_score", ascending=False).reset_index(drop=True)

# ======================
# Global Configuration (7 standardized datasets)
# ======================
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\SCGDL"

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

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    mk_dir(OUTPUT_ROOT_DIR)
    global_summary = []
    print("=" * 80)
    print("SCGDL Clustering + Cluster‑based SVG Identification")
    print(f"Input Folder: {DATA_INPUT_DIR}")
    print(f"Output Folder: {OUTPUT_ROOT_DIR}")
    print("=" * 80)

    for file_name in DATA_FILE_LIST:
        file_full_path = os.path.join(DATA_INPUT_DIR, file_name)
        dataset_name = file_name.replace("_standardized.h5ad", "")
        dataset_save_dir = os.path.join(OUTPUT_ROOT_DIR, dataset_name)
        mk_dir(dataset_save_dir)

        print(f"\n==== Processing: {dataset_name} ====")
        if not os.path.exists(file_full_path):
            print(f"WARNING: Missing file {file_full_path}, skip")
            continue

        adata = sc.read_h5ad(file_full_path)
        adata.var_names_make_unique()
        target_cluster = CLUSTER_NUM_DICT[dataset_name]
        print(f"Target cluster number: {target_cluster}")

        # SCGDL clustering
        clustered_adata = run_SCGDL_clustering(adata, n_clusters=target_cluster, dataset_name=file_name)
        # Compute SVG
        svg_df = calc_cluster_SVG(clustered_adata, cluster_key="SCGDL_cluster", perm_times=1000, seed=0)

        # Save outputs
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"SCGDL_{dataset_name}_clustered.h5ad"))
        svg_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_SCGDL_SVG_full_list.csv"), index=False)

        svg_sig = svg_df[svg_df["is_SVG_FDR0.05"]]
        svg_sig.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Significant_SVG_FDR0.05.csv"), index=False)
        svg_df.head(50).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_Top50_SVG.csv"), index=False)

        sig_count = svg_sig.shape[0]
        record = {
            "platform": PLATFORM_LABEL[file_name],
            "dataset": dataset_name,
            "cluster_number": target_cluster,
            "total_gene_num": adata.n_vars,
            "significant_SVG_FDR0.05": sig_count
        }
        global_summary.append(record)
        pd.DataFrame([record]).to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_summary.csv"), index=False)
        print(f"DONE | Significant SVGs (FDR<0.05): {sig_count}")

        del adata, clustered_adata, svg_df, svg_sig
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(global_summary).to_csv(
        os.path.join(OUTPUT_ROOT_DIR, "SCGDL_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "=" * 80)
    print("ALL SEVEN DATASETS FINISHED")
    print("Global summary saved.")
    print("=" * 80)