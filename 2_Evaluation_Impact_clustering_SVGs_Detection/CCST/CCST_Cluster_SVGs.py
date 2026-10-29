# -*- coding:utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import torch
import warnings
import numpy as np
import scanpy as sc
import pandas as pd

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# -------------------------- CCST Model Definition --------------------------
class CCST(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 64], num_clusters=10, dropout=0.2):
        super(CCST, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU()
        )
        self.cluster_layer = torch.nn.Linear(hidden_dims[1], num_clusters)
        self.decoder = torch.nn.Linear(hidden_dims[1], input_dim)

    def forward(self, x):
        feat = self.encoder(x)
        recon = self.decoder(feat)
        cluster_prob = torch.softmax(self.cluster_layer(feat), dim=1)
        return feat, recon, cluster_prob

def train_ccst(adata, epochs=200, lr=0.001, hidden_dim=64, num_clusters=10, device='cuda:0'):
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    if "log1p" not in adata.uns:
        sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=200)

    pca_data = adata.obsm['X_pca'].copy()
    x = torch.tensor(pca_data, dtype=torch.float32).to(device)

    model = CCST(input_dim=200, hidden_dims=[256, hidden_dim], num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        feat, recon, prob = model(x)
        recon_loss = mse_loss(recon, x)
        cluster_loss = -torch.mean(torch.sum(prob * torch.log(prob + 1e-10), dim=1))
        loss = recon_loss + 0.1 * cluster_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings, _, _ = model(x)
    adata.obsm['CCST_emb'] = embeddings.cpu().numpy()
    return adata

# -------------------------- Fixed resolution search for target cluster count --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(adata.obs['leiden'].unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

# -------------------------- CCST SVG Pipeline (MODIFIED: one‑row‑per‑gene output) --------------------------
def run_ccst_svg(dataset_name, n_cluster):
    h5ad_path = os.path.join(BASE_DATA_PATH, f"{dataset_name}_standardized.h5ad")
    print(f"\n==================================================")
    print(f"Processing dataset: {dataset_name}")
    print(f"File: {h5ad_path}")
    print(f"==================================================")

    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    total_original_gene = adata.n_vars
    print(f"Original total genes in dataset: {total_original_gene}")

    adata_copy = adata.copy()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train CCST embedding
    adata_ccst = train_ccst(adata_copy, epochs=200, num_clusters=n_cluster, device=device)

    # Leiden clustering on CCST embedding
    sc.pp.neighbors(adata_ccst, use_rep='CCST_emb', n_neighbors=20)
    eval_resolution = res_search_fixed_clus(adata_ccst, n_cluster)
    sc.tl.leiden(adata_ccst, key_added="CCST_leiden", resolution=eval_resolution)

    # Identify SVGs: Wilcoxon test between CCST spatial clusters
    sc.tl.rank_genes_groups(adata_ccst, groupby="CCST_leiden", method="wilcoxon")

    # Collect all per‑cluster marker results
    cluster_list = sorted(adata_ccst.obs["CCST_leiden"].unique())
    svg_records = []
    for clus in cluster_list:
        df = sc.get.rank_genes_groups_df(adata_ccst, group=str(clus))
        df["cluster"] = clus
        svg_records.append(df)
    svg_result = pd.concat(svg_records, ignore_index=True)

    # ---------- MODIFICATION: aggregate to ONE ROW PER GENE ----------
    agg_df = svg_result.groupby("names").agg(
        min_pvals_adj=("pvals_adj", "min"),
        min_pvals=("pvals", "min"),
        max_logFC=("logfoldchanges", "max"),
    ).reset_index()
    agg_df.rename(columns={"names":"gene_name"}, inplace=True)

    # SVG flag threshold adj p <0.05
    agg_df["is_SVG"] = agg_df["min_pvals_adj"] < 0.05

    # Make sure output table contains ALL original genes (fill missing genes with NA)
    all_gene_df = pd.DataFrame({"gene_name": adata.var_names.tolist()})
    final_gene_table = pd.merge(all_gene_df, agg_df, on="gene_name", how="left")

    sig_unique_count = int(final_gene_table["is_SVG"].sum())

    # Save outputs
    save_folder = os.path.join(OUTPUT_ROOT, dataset_name)
    mk_dir(save_folder)

    # Main output: one‑row‑per‑all‑genes table (matches original gene number e.g.539 rows)
    main_csv_path = os.path.join(save_folder, f"{dataset_name}_CCST_SVG_AllGenes.csv")
    final_gene_table.to_csv(main_csv_path, index=False)

    # Optional: keep original full per‑cluster table for reference
    full_cluster_csv = os.path.join(save_folder, f"{dataset_name}_CCST_SVG_FullPerCluster.csv")
    svg_result.to_csv(full_cluster_csv, index=False)

    # Unique SVG subset
    svg_only_df = final_gene_table[final_gene_table["is_SVG"]].copy()
    svg_unique_csv = os.path.join(save_folder, f"{dataset_name}_CCST_SVG_UniqueList.csv")
    svg_only_df.to_csv(svg_unique_csv, index=False)

    print(f"✅ All‑gene table saved (total rows = original gene count {final_gene_table.shape[0]}): {main_csv_path}")
    print(f"✅ Unique significant SVG count (adj.p<0.05): {sig_unique_count}")

    return {"dataset": dataset_name, "total_genes": total_original_gene, "sig_SVG_unique_count": sig_unique_count}

# -------------------------- Global Configuration --------------------------
if __name__ == '__main__':
    # Input & Output Path
    BASE_DATA_PATH = r"D:\My_Project_ST\benchmarking of ST platforms\Datasets_Standardized"
    OUTPUT_ROOT    = r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\CCST"
    mk_dir(OUTPUT_ROOT)

    target_dataset_info = [
        {"name": "10X_BC_TENX13", "n_cluster": 7},
        {"name": "10X_Lung_TENX62", "n_cluster": 6},
        {"name": "Slide-seq2_Kedney_HBM965.PSNC.855", "n_cluster": 8},
        {"name": "ST_BC_SPA130", "n_cluster": 5},
        {"name": "STARmap_BOOST_HMI", "n_cluster": 6},
        {"name": "Xenium_BC_NCBI784", "n_cluster": 7},
        {"name": "Xenium_Lung_NCBI856", "n_cluster": 6}
    ]

    warnings.filterwarnings('ignore')
    np.random.seed(0)
    torch.manual_seed(0)
    summary_list = []

    for info in target_dataset_info:
        ds_name = info["name"]
        n_clus = info["n_cluster"]
        try:
            res = run_ccst_svg(ds_name, n_clus)
            summary_list.append(res)
        except Exception as e:
            print(f"\n❌ FAILED {ds_name}: {str(e)}")
            summary_list.append({"dataset": ds_name, "total_genes":"NA", "sig_SVG_unique_count": "Failed"})

    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(OUTPUT_ROOT, "CCST_Batch_SVG_Summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n==================== ALL DATASETS FINISHED ====================")
    print(f"Global summary saved to: {summary_csv}")
    print(summary_df)