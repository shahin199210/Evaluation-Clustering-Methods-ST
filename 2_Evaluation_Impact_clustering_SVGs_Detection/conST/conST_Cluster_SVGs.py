import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gc
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import warnings
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ===================== conST required imports & helper functions =====================
sys.path.append(r"D:\My_Project_ST\1_Evaluation_SRT")
from src.graph_func import graph_construction
from src.utils_func import adata_preprocess, res_search_fixed_clus
from src.training import conST_training
import argparse

def seed_torch(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ensure_spatial_coords(adata):
    if 'spatial' in adata.obsm:
        del adata.obsm['spatial']
    n_cells = adata.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_cells)))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    dummy_spatial = np.vstack([x.ravel(), y.ravel()]).T[:n_cells]
    adata.obsm['spatial'] = dummy_spatial.astype(np.float32)
    return adata

def refine_function(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        num_nbs = 6
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values(ascending=False)
        nbs = dis_tmp[0:num_nbs+1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs/2) and (np.max(v_c) > num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

# ---------------- conST clustering pipeline (Removed evaluation, ground‑truth, plotting, metrics) ----------------
def run_conST_clustering(adata_raw, n_clusters, epochs=200):
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--knn_distanceType', type=str, default='euclidean')
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--cell_feat_dim', type=int, default=300)
    parser.add_argument('--feat_hidden1', type=int, default=100)
    parser.add_argument('--feat_hidden2', type=int, default=20)
    parser.add_argument('--gcn_hidden1', type=int, default=32)
    parser.add_argument('--gcn_hidden2', type=int, default=8)
    parser.add_argument('--p_drop', type=float, default=0.2)
    parser.add_argument('--use_img', type=bool, default=False)
    parser.add_argument('--img_w', type=float, default=0.1)
    parser.add_argument('--use_pretrained', type=bool, default=False)
    parser.add_argument('--using_mask', type=bool, default=False)
    parser.add_argument('--feat_w', type=float, default=10)
    parser.add_argument('--gcn_w', type=float, default=0.1)
    parser.add_argument('--dec_kl_w', type=float, default=10)
    parser.add_argument('--gcn_lr', type=float, default=0.01)
    parser.add_argument('--gcn_decay', type=float, default=0.01)
    parser.add_argument('--dec_cluster_n', type=int, default=10)
    parser.add_argument('--dec_interval', type=int, default=20)
    parser.add_argument('--dec_tol', type=float, default=0.00)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=100)
    parser.add_argument('--cont_l2l', type=float, default=0.3)
    parser.add_argument('--cont_l2c', type=float, default=0.1)
    parser.add_argument('--cont_l2g', type=float, default=0.1)
    parser.add_argument('--edge_drop_p1', type=float, default=0.1)
    parser.add_argument('--edge_drop_p2', type=float, default=0.1)
    parser.add_argument('--node_drop_p1', type=float, default=0.2)
    parser.add_argument('--node_drop_p2', type=float, default=0.3)
    parser.add_argument('--eval_resolution', type=int, default=1)
    parser.add_argument('--eval_graph_n', type=int, default=20)

    params = parser.parse_args([])
    params.device = torch.device('cpu')

    adata = adata_raw.copy()
    adata = ensure_spatial_coords(adata)
    adata_X = adata_preprocess(adata, min_cells=5, pca_n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], params)
    params.cell_num = adata.shape[0]

    conST_net = conST_training(adata_X, graph_dict, params, n_clusters)
    conST_net.pretraining()
    conST_net.major_training()
    conST_embedding = conST_net.get_embedding()

    embed_adata = anndata.AnnData(conST_embedding)
    sc.pp.neighbors(embed_adata, n_neighbors=params.eval_graph_n)
    eval_resolution = res_search_fixed_clus(embed_adata, n_clusters)
    sc.tl.leiden(embed_adata, key_added="conST_leiden", resolution=eval_resolution)

    # run cluster refinement (square/hexagon fallback hexagon)
    index = [str(i) for i in range(embed_adata.n_obs)]
    dis = graph_dict['adj_norm'].to_dense().numpy() + np.eye(graph_dict['adj_norm'].shape[0])
    refined_labels = refine_function(sample_id=index, pred=embed_adata.obs["conST_leiden"].tolist(), dis=dis, shape="hexagon")

    # attach back to original adata
    adata_raw.obs["conST_leiden"] = embed_adata.obs["conST_leiden"].values
    adata_raw.obs["conST_refine"] = refined_labels
    adata_raw.obsm["conST_embedding"] = conST_embedding

    del adata, embed_adata, conST_net
    gc.collect()
    torch.cuda.empty_cache()
    return adata_raw

# ---------------- Shared permutation‑test SVG function (consistent with all other methods) ----------------
def calc_cluster_SVG(adata, cluster_key="conST_refine", perm_times=1000, seed=0):
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
        "conST_SVG_score": obs_score,
        "raw_p_value": raw_p,
        "fdr_BH_p_adjust": fdr_p,
        "is_SVG_FDR0.05": sig_flag
    })
    del cluster_mean_matrix, perm_score_set
    gc.collect()
    return svg_result_df.sort_values("conST_SVG_score", ascending=False).reset_index(drop=True)

# ---------------- Global config (7 standardized datasets) ----------------
DATA_INPUT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT_DIR = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\conST"

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

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    global_summary = []
    seed_torch(0)

    print("="*80)
    print("conST Clustering + Cluster‑based SVG Identification")
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

        # run conST clustering + refinement
        clustered_adata = run_conST_clustering(adata, n_clusters=target_cluster, epochs=200)
        # compute SVGs using refined cluster labels "conST_refine"
        svg_full_df = calc_cluster_SVG(clustered_adata, cluster_key="conST_refine", perm_times=1000, seed=0)

        # save outputs
        clustered_adata.write_h5ad(os.path.join(dataset_save_dir, f"conST_{dataset_name}_clustered.h5ad"))
        svg_full_df.to_csv(os.path.join(dataset_save_dir, f"{dataset_name}_conST_SVG_full_list.csv"), index=False)

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
        os.path.join(OUTPUT_ROOT_DIR, "conST_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("\n" + "="*80)
    print("ALL 7 DATASETS COMPLETED")
    print(f"Global summary saved to: {OUTPUT_ROOT_DIR}")
    print("="*80)