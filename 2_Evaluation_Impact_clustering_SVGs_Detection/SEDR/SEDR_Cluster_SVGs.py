import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import gc
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import warnings
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

# SEDR source imports
sys.path.insert(0, r"D:\Benchmark_ST_analysis-main\SEDR")
sys.path.insert(0, r"D:\Benchmark_ST_analysis-main")
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess
from src.SEDR_train import SEDR_Train

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# Spatial coordinate helper (keep for SEDR graph building)
# --------------------------
def get_spatial_coords(adata, dataset_name):
    if dataset_name.startswith("STARmap"):
        if 'spatial' in adata.obsm:
            try:
                coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
                if coords.ndim == 2 and coords.shape[0] == adata.shape[0] and coords.shape[1]==2:
                    return coords
            except (ValueError, TypeError):
                pass
        for key in ['spatial_coords','pos','coord']:
            if key in adata.obsm:
                try:
                    coords = np.asarray(adata.obsm[key], dtype=np.float32)
                    if coords.ndim ==2 and coords.shape[0]==adata.shape[0] and coords.shape[1]==2:
                        return coords
                except:
                    continue
        for col_pair in [['x','y'],['X','Y'],['x_coord','y_coord'],['coord_x','coord_y']]:
            if all(col in adata.obs.columns for col in col_pair):
                return np.asarray(adata.obs[col_pair].values, dtype=np.float32)
        n_cells = adata.shape[0]
        grid_size=int(np.ceil(np.sqrt(n_cells)))
        x,y=np.meshgrid(np.arange(grid_size),np.arange(grid_size))
        coords=np.stack([x.flatten(),y.flatten()],axis=1)[:n_cells].astype(np.float32)
        return coords

    if 'spatial' in adata.uns:
        if isinstance(adata.uns['spatial'], dict):
            for k in ['coords','pos','tissue_positions']:
                if k in adata.uns['spatial']:
                    v=adata.uns['spatial'][k]
                    if isinstance(v,np.ndarray) and v.ndim==2 and v.shape[0]==adata.shape[0]:
                        return np.asarray(v,dtype=np.float32)
        else:
            coords=np.asarray(adata.uns['spatial'],dtype=np.float32)
            if coords.ndim==2 and coords.shape[0]==adata.shape[0]:
                return coords
    for col_pair in [['x','y'],['X','Y'],['x_coord','y_coord'],['coord_x','coord_y'],['spatial_x','spatial_y'],['cell_x','cell_y']]:
        if all(col in adata.obs.columns for col in col_pair):
            return np.asarray(adata.obs[col_pair].values,dtype=np.float32)
    n_cells=adata.shape[0]
    grid_size=int(np.ceil(np.sqrt(n_cells)))
    x,y=np.meshgrid(np.arange(grid_size),np.arange(grid_size))
    coords=np.stack([x.flatten(),y.flatten()],axis=1)[:n_cells].astype(np.float32)
    return coords

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    res_list=list(np.arange(0.2,2.5,increment))
    res_list.reverse()
    for res in res_list:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique())==fixed_clus_count:
            break
    return res

# --------------------------
# SEDR embedding + clustering, NO evaluation / NO plotting
# --------------------------
def run_SEDR_clustering(adata_raw, n_cluster, epochs=200):
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--k',type=int,default=10)
    parser.add_argument('--knn_distanceType',type=str,default='euclidean')
    parser.add_argument('--epochs',type=int,default=epochs)
    parser.add_argument('--cell_feat_dim',type=int,default=200)
    parser.add_argument('--feat_hidden1',type=int,default=100)
    parser.add_argument('--feat_hidden2',type=int,default=20)
    parser.add_argument('--gcn_hidden1',type=int,default=32)
    parser.add_argument('--gcn_hidden2',type=int,default=8)
    parser.add_argument('--p_drop',type=float,default=0.2)
    parser.add_argument('--using_dec',type=bool,default=True)
    parser.add_argument('--using_mask',type=bool,default=False)
    parser.add_argument('--feat_w',type=float,default=10)
    parser.add_argument('--gcn_w',type=float,default=0.1)
    parser.add_argument('--dec_kl_w',type=float,default=10)
    parser.add_argument('--gcn_lr',type=float,default=0.01)
    parser.add_argument('--gcn_decay',type=float,default=0.01)
    parser.add_argument('--dec_cluster_n',type=int,default=10)
    parser.add_argument('--dec_interval',type=int,default=20)
    parser.add_argument('--dec_tol',type=float,default=0.00)
    parser.add_argument('--eval_resolution',type=int,default=1)
    parser.add_argument('--eval_graph_n',type=int,default=20)
    params=parser.parse_args([])
    params.device=device

    adata=adata_raw.copy()
    adata_X=adata_preprocess(adata, min_cells=5, pca_n_comps=params.cell_feat_dim)
    spatial_coords=get_spatial_coords(adata_raw, dataset_name="tmp")
    params.cell_num=adata.shape[0]
    graph_dict=graph_construction(spatial_coords, adata.shape[0], params)

    sedr_net=SEDR_Train(adata_X, graph_dict, params)
    if params.using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat,_,_,_=sedr_net.process()

    adata_sedr=anndata.AnnData(sedr_feat)
    adata_sedr.obsm["spatial"]=spatial_coords
    sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
    eval_res=res_search_fixed_clus(adata_sedr, n_cluster)
    sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_res)

    # attach cluster label and embedding back to original adata
    adata_raw.obs["SEDR_leiden"]=adata_sedr.obs["SEDR_leiden"].values
    adata_raw.obsm["SEDR_embedding"]=sedr_feat

    del adata, adata_sedr, sedr_net
    gc.collect()
    torch.cuda.empty_cache()
    return adata_raw

# --------------------------
# Shared permutation‑test SVG function (same as DeepSpaCE / iSpatial / SpaNCMG)
# --------------------------
def calc_cluster_SVG(adata, cluster_key="SEDR_leiden", perm_times=1000, seed=0):
    np.random.seed(seed)
    cell_labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = adata.shape
    unique_clusters = np.unique(cell_labels)
    n_cluster = len(unique_clusters)

    expr_mat_sp = adata.X
    if issparse(expr_mat_sp):
        global_gene_mean = np.asarray(expr_mat_sp.mean(axis=0)).ravel()
    else:
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
        "SEDR_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set
    gc.collect()
    return svg_result_df.sort_values("SEDR_SVG_score", ascending=False).reset_index(drop=True)

# --------------------------
# Global config 7 datasets
# --------------------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\SEDR"

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
# Main execution
# --------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    global_summary = []

    print("="*80)
    print("SEDR Clustering + Cluster‑based SVG Identification")
    print(f"Input : {DATA_INPUT_DIR}")
    print(f"Output: {OUTPUT_ROOT_DIR}")
    print("="*80)

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

        # run SEDR clustering
        clustered_adata = run_SEDR_clustering(adata, n_cluster=target_cluster, epochs=200)
        # compute SVGs
        svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="SEDR_leiden", perm_times=1000, seed=0)

        # save outputs
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"SEDR_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_SEDR_SVG_full_list.csv"), index=False)

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
        os.path.join(OUTPUT_ROOT_DIR, "SEDR_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "="*80)
    print("ALL 7 DATASETS COMPLETED")
    print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
    print("="*80)