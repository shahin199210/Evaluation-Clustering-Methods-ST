# -*- coding:utf-8 -*-
import sys
import io
# Global UTF-8 encoding lock
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import time
import psutil
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scanpy as sc
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import KMeans

# -------------------------- Minimal Core Model & Loss Functions (reserved only for STMGCN training) --------------------------
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def kl_loss(q_log, p):
    return torch.sum(p * (p.log() - q_log))

class STMGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nclass):
        super(STMGCN, self).__init__()
        from torch.nn import Linear
        self.gcn1 = Linear(nfeat, nhid1)
        self.gcn2 = Linear(nhid1, nhid1)
        self.cluster_layer = torch.nn.Parameter(torch.Tensor(nclass, nhid1))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def mgcn(self, x, adj1, adj2):
        x = torch.relu(self.gcn1(torch.matmul(adj1, x)))
        x = torch.relu(self.gcn2(torch.matmul(adj2, x)))
        return x

    def forward(self, x, adj1, adj2):
        emb = self.mgcn(x, adj1, adj2)
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return emb, q

# -------------------------- Basic Utility --------------------------
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# -------------------------- SVG Calculation Function (Consistent with your Leiden SVG logic) --------------------------
def calculate_cluster_based_SVG_fast(adata, cluster_key="pred", n_permutation=1000, seed=0):
    np.random.seed(seed)
    expr_mat = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
    labels = adata.obs[cluster_key].astype(np.int64).values
    n_cells, n_genes = expr_mat.shape
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    global_mean = expr_mat.mean(axis=0, keepdims=True)

    # Observed SVG Score
    cluster_means = np.zeros((n_clusters, n_genes))
    for idx, c in enumerate(clusters):
        mask = labels == c
        cluster_means[idx] = expr_mat[mask].mean(axis=0)
    svg_score = np.sum((cluster_means - global_mean) ** 2, axis=0) / n_clusters

    # Permutation null distribution
    perm_scores = np.zeros((n_permutation, n_genes))
    for p in range(n_permutation):
        shuffle_idx = np.random.permutation(n_cells)
        perm_expr = expr_mat[shuffle_idx, :]
        perm_cluster_means = np.zeros((n_clusters, n_genes))
        for idx, c in enumerate(clusters):
            mask = labels == c
            perm_cluster_means[idx] = perm_expr[mask].mean(axis=0)
        perm_scores[p] = np.sum((perm_cluster_means - global_mean) ** 2, axis=0) / n_clusters

    raw_pval = (np.sum(perm_scores >= svg_score, axis=0) + 1) / (n_permutation + 1)
    reject, fdr_padj, _, _ = multipletests(raw_pval, alpha=0.05, method='fdr_bh')

    df = pd.DataFrame({
        "gene_ids": adata.var.index.values,
        "STMGCN_SVG_score": svg_score,
        "raw_pvalue": raw_pval,
        "fdr_bh_adjust_p": fdr_padj,
        "is_SVG_FDR0.05": reject
    })
    return df.sort_values("STMGCN_SVG_score", ascending=False).reset_index(drop=True)

# -------------------------- Modified STMGCN Train Function (Only clustering, remove all evaluation metrics) --------------------------
def train_STMGCN_cluster(adata_raw, n_cluster, random_seed=50):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        torch.cuda.manual_seed_all(random_seed)

    # Preprocess feature & graph construction (consistent with original logic)
    adata = adata_raw.copy()
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    adj_np = adata.obsp["connectivities"].toarray()
    adj1 = torch.FloatTensor(adj_np)
    adj2 = torch.FloatTensor(adj_np)
    feat_np = adata.X.toarray() if issparse(adata.X) else adata.X
    features = torch.FloatTensor(feat_np)

    model = STMGCN(nfeat=features.shape[1], nhid1=32, nclass=n_cluster)
    if cuda_flag:
        model.cuda()
        features = features.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    emb = model.mgcn(features, adj1, adj2)

    # Kmeans init clustering center
    kmeans = KMeans(n_clusters=n_cluster, n_init=20, random_state=random_seed)
    y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())
    emb_pd = pd.DataFrame(emb.detach().cpu().numpy())
    group_pd = pd.Series(y_pred, name="Group")
    merge_df = pd.concat([emb_pd, group_pd], axis=1)
    cluster_centers = np.asarray(merge_df.groupby("Group").mean())

    y_pred_last = y_pred
    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(cluster_centers))

    # Training loop
    model.train()
    max_epochs = 2000
    update_interval = 3
    tol = 1e-4
    for epoch in range(max_epochs):
        if epoch % update_interval == 0:
            _, tem_q = model(features, adj1, adj2)
            tem_q = tem_q.detach()
            p = target_distribution(tem_q)
            y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last) / y_pred.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and delta_label < tol:
                print(f"Delta label {delta_label:.6f} < tol {tol}, early stop")
                break
        optimizer.zero_grad()
        x_out, q_out = model(features, adj1, adj2)
        loss = kl_loss(q_out.log(), p)
        loss.backward()
        optimizer.step()

    # Write clustering result back to AnnData
    adata_raw.obs["pred"] = y_pred_last.astype(str)
    adata_raw.obsm["STMGCN_embedding"] = x_out.detach().cpu().numpy()
    return adata_raw

# -------------------------- Global Path Configuration --------------------------
BASE_DATA_PATH = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
OUTPUT_ROOT = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\STMGCN"

# Dataset list matching your local file names
DATA_FILE_LIST = [
    "Xenium_BC_NCBI784_standardized.h5ad",
    "Xenium_Lung_NCBI856_standardized.h5ad",
    "10X_BC_TENX13_standardized.h5ad",
    "10X_Lung_TENX62_standardized.h5ad",
    "ST_BC_SPA130_standardized.h5ad",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad",
    "STARmap_BOOST_HMI_standardized.h5ad"
    
]

PLATFORM_MAP = {
    "Xenium_BC_NCBI784_standardized.h5ad": "Xenium",
    "Xenium_Lung_NCBI856_standardized.h5ad": "Xenium",
    "10X_BC_TENX13_standardized.h5ad": "10X Visium",
    "10X_Lung_TENX62_standardized.h5ad": "10X Visium",
    "ST_BC_SPA130_standardized.h5ad": "ST",
    "Slide-seq2_Kedney_HBM965.PSNC.855_standardized.h5ad": "Slide-seq",
    "STARmap_BOOST_HMI_standardized.h5ad": "STARmap"
    
}

DATASET_CLUSTER_NUM = {
    "Xenium_BC_NCBI784": 8,
    "Xenium_Lung_NCBI856": 6,
    "10X_BC_TENX13": 5,
    "10X_Lung_TENX62": 6,
    "ST_BC_SPA130": 4,
    "Slide-seq2_Kedney_HBM965.PSNC.855": 7,
    "STARmap_BOOST_HMI": 3
    
}

# -------------------------- Main Execution Pipeline --------------------------
if __name__ == "__main__":
    mk_dir(OUTPUT_ROOT)
    global_summary_list = []
    print(f"\n{'='*80}")
    print(f"STMGCN CLUSTERING + SVG IDENTIFICATION START")
    print(f"Dataset Source Folder: {BASE_DATA_PATH}")
    print(f"Output Save Folder: {OUTPUT_ROOT}")
    print(f"{'='*80}")

    for file_name in DATA_FILE_LIST:
        full_path = os.path.join(BASE_DATA_PATH, file_name)
        ds_name = file_name.replace("_standardized.h5ad", "")
        save_dir = mk_dir(os.path.join(OUTPUT_ROOT, ds_name))
        print(f"\n==== Current Dataset Processing: {ds_name} ====")

        # File missing safety check
        if not os.path.exists(full_path):
            print(f"⚠️ Warning: File missing -> {full_path}, skip current dataset")
            continue

        # Load standardized h5ad
        adata = sc.read_h5ad(full_path)
        adata.var_names_make_unique()
        target_cluster = DATASET_CLUSTER_NUM[ds_name]
        print(f"Target cluster number: {target_cluster}")

        # Runtime & Memory timer start
        start_time = time.time()
        tracemalloc.start()
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024**3

        # Step1: STMGCN clustering
        adata_clustered = train_STMGCN_cluster(adata, n_cluster=target_cluster)
        # Step2: SVG identification
        svg_df = calculate_cluster_based_SVG_fast(adata_clustered, cluster_key="pred", n_permutation=1000, seed=0)

        # Runtime & Memory statistics
        total_runtime = time.time() - start_time
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        mem_increase = mem_end - mem_start
        _, peak_mem_byte = tracemalloc.get_traced_memory()
        peak_memory_GB = peak_mem_byte / 1024**3
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Export all output files
        adata_clustered.write_h5ad(os.path.join(save_dir, f"STMGCN_{ds_name}_clustered.h5ad"))
        svg_df.to_csv(os.path.join(save_dir, f"{ds_name}_STMGCN_Based_SVG_full_table.csv"), index=False)
        svg_df[svg_df["is_SVG_FDR0.05"]].to_csv(os.path.join(save_dir, f"{ds_name}_Significant_SVG_FDR0.05.csv"), index=False)
        svg_df.head(50).to_csv(os.path.join(save_dir, f"{ds_name}_Top50_SVG.csv"), index=False)

        # Single dataset summary record
        sig_svg_count = int(svg_df["is_SVG_FDR0.05"].sum())
        single_summary = {
            "platform_category": PLATFORM_MAP[file_name],
            "dataset_name": ds_name,
            "target_cluster_num": target_cluster,
            "total_genes": adata_clustered.n_vars,
            "significant_SVG_FDR0.05": sig_svg_count,
            "runtime_seconds": round(total_runtime, 2),
            "ram_increase_GB": round(mem_increase, 3),
            "peak_memory_GB": round(peak_memory_GB, 3)
        }
        global_summary_list.append(single_summary)
        pd.DataFrame([single_summary]).to_csv(os.path.join(save_dir, f"{ds_name}_single_summary.csv"), index=False)

        print(f"✅ {ds_name} finished | Significant SVG count: {sig_svg_count}")
        print(f"Runtime: {total_runtime:.2f}s | Peak Memory: {peak_memory_GB:.3f}GB\n")

    # Export global all-dataset summary
    pd.DataFrame(global_summary_list).to_csv(
        os.path.join(OUTPUT_ROOT, "STMGCN_All_Datasets_Global_Summary.csv"),
        index=False
    )
    print("="*80)
    print("All available datasets completed successfully, global summary saved!")
    print("="*80)