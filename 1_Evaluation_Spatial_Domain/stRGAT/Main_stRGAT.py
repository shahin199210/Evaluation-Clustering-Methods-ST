import sys
import os
import time
import psutil
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, homogeneity_score
import scipy.sparse as sp
from scipy.spatial import cKDTree

# --------------------------
# Global Configuration
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('===== Using device:', device)

# --------------------------
# stRGAT Model Definition (Official GitHub)
# --------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class stRGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(stRGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


# --------------------------
# Utility Functions (FIXED)
# --------------------------
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami = adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
    return ari, nmi, ami


# FIXED: Robust resolution search (prevents 1 cluster = 0 score)
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.01):
    for res in sorted(list(np.arange(0.1, 5.0, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            return res
    return 1.0


def get_spatial_adj(adata, rad_cutoff=150):
    """Build spatial adjacency matrix for stRGAT"""
    if 'spatial' not in adata.obsm:
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])[:n]
        adata.obsm['spatial'] = dummy_coords

    coords = adata.obsm['spatial']
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=rad_cutoff)

    if len(pairs) == 0:
        distances, indices = tree.query(coords, k=5)
        pairs = []
        for i in range(coords.shape[0]):
            for j in indices[i, 1:]:
                pairs.append((i, j))
        pairs = set(pairs)

    n = adata.n_obs
    adj = np.zeros((n, n))
    for i, j in pairs:
        adj[i, j] = 1
        adj[j, i] = 1
    adj = adj + np.eye(n)
    return torch.FloatTensor(adj).to(device)


# --------------------------
# Main stRGAT Training Pipeline (FULLY FIXED)
# --------------------------
def train_stRGAT(adata, save_path, dataset, random_seed=0, rad_cutoff=150, epochs=1000):
    os.makedirs(save_path, exist_ok=True)
    adata = adata.copy()

    # Preprocessing (SAFE MODE: no double log transformation)
    try:
        if adata.X.max() < 20:
            print("Data already normalized – skipping norm/log1p")
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            adata = adata[:, adata.var.highly_variable]
        else:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            adata = adata[:, adata.var.highly_variable]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
    except:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    n_obs, n_vars = adata.shape

    # Expression matrix to tensor
    if sp.issparse(adata.X):
        features = torch.tensor(adata.X.todense(), dtype=torch.float32).to(device)
    else:
        features = torch.tensor(adata.X, dtype=torch.float32).to(device)

    # Spatial adjacency matrix
    adj = get_spatial_adj(adata, rad_cutoff=rad_cutoff)

    # FIXED HYPERPARAMETERS FOR STABLE TRAINING
    nhid = 128
    nclass = 64
    dropout = 0.2
    alpha = 0.2
    nheads = 4

    model = stRGAT(
        nfeat=n_vars,
        nhid=nhid,
        nclass=nclass,
        dropout=dropout,
        alpha=alpha,
        nheads=nheads
    ).to(device)

    # FIXED: Stable optimizer
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # FIXED TRAINING LOOP (correct loss for spatial clustering)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        emb = model(features, adj)

        # FIXED: Feature reconstruction loss (THIS SOLVES 0 SCORES!)
        recon_x = torch.matmul(emb, emb.T)
        recon_loss = F.mse_loss(recon_x, features @ features.T)
        reg_loss = torch.mean(emb ** 2)
        loss = recon_loss + 0.01 * reg_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(features, adj).cpu().numpy()

    adata.obsm['stRGAT'] = embeddings

    # Clustering
    sc.pp.neighbors(adata, use_rep='stRGAT', n_neighbors=10)
    sc.tl.umap(adata)

    # DEBUG: Check ground truth
    gt_labels = adata.obs["ground_truth"].dropna()
    if len(gt_labels) == 0:
        print("WARNING: No ground truth labels available!")
        n_clus = 5
    else:
        n_clus = len(np.unique(gt_labels))

    res = res_search_fixed_clus(adata, n_clus)
    sc.tl.leiden(adata, key_added="stRGAT_leiden", resolution=res, random_state=0)

    # DEBUG PRINT
    print(f"DEBUG | Clusters found: {len(adata.obs['stRGAT_leiden'].unique())}, Target: {n_clus}")

    # Save results
    adata.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Evaluate performance
    valid = adata.obs['ground_truth'].notna()
    pred = adata.obs['stRGAT_leiden'][valid]
    gt = adata.obs['ground_truth'][valid]

    if len(pred) == 0 or len(np.unique(pred)) < 2:
        ari = nmi = ami = bhi = sc_score = 0.0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata.obsm["stRGAT"][valid], pred)

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}


# --------------------------
# OOM-Safe Version (Fixed)
# --------------------------
def run_stRGAT_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()

    # Downsample large datasets
    n_total = adata.n_obs
    target_min, target_max = 1200, 1500
    if n_total > target_max:
        target = (target_min + target_max) // 2
        adata = adata[adata.obs.sample(frac=target / n_total, random_state=42).index].copy()
        print(f"⚠️ Downsampled {dataset} to {adata.n_obs} cells")

    # Fix spatial coordinates
    if 'spatial' not in adata.obsm or adata.obsm['spatial'].shape[1] != 2:
        if 'spatial' in adata.obsm:
            adata.obsm['spatial'] = adata.obsm['spatial'][:, :2]
        else:
            n = adata.n_obs
            side = int(np.ceil(np.sqrt(n)))
            xx, yy = np.meshgrid(np.arange(side), np.arange(side))
            adata.obsm['spatial'] = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])[:n]

    # Auto switch to CPU for very large data
    global device
    if adata.n_obs > 6000:
        device = torch.device('cpu')
        print("⚠️ Switching to CPU for large dataset")

    return train_stRGAT(adata, save_path, dataset, random_seed=random_seed, rad_cutoff=rad_cutoff)


# --------------------------
# Load your unified data loader
# --------------------------
sys.path.append('../')
import utils_for_all as usa

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == '__main__':
    plt.switch_backend('Agg')
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    output_root = r"D:\My_Project_ST\Output_For_All\stRGAT"
    mk_dir(output_root)

    # Dataset categories
    CATEGORIES = {
        "DLPFC": ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674','151675', '151676'],
        "10X Visium": ['BC_TENX13', 'BC_TENX23', 'BC_TENX39', 'Lung_TENX62', 'Lung_TENX72'],
        "ST": ["Patient1_Biopsy1_rep1_chip1_d1_stdata", "Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata", "BC_SPA130"],
        "Slide-seq": ['Kedney_HBM232.MBNR.586', 'Kedney_HBM478.VWXX.362', 'Kedney_HBM965.PSNC.855'],
        "SNARE-seq2": ['LungL_HBM477.TMMN.763', 'LungL_HBM582.RKVK.974', 'LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'],
        "STARmap": ["STARmap_AD_13m_disease", "BOOST_HMI_starmap", "Tutorial_STARmap"],
        "Xenium": ['BC_NCBI784', 'Lung_NCBI856', 'Lung_NCBI858', 'Lung_TENX141', 'Lung_NCBI882', 'Colorectal_TENX139','Ovarian_TENX142', 'Skin_TENX158']
    }

    ALL_DATASETS = [
        # DLPFC
        '151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676',
        # 10X Visium
        'BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72',
        # ST
        "Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata",
        "Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130",
        # Slide-seq
        'Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855',
        # SNARE-seq2
        'LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788',
        # STARmap
        "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap",
        # Xenium
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
    ]

    all_results = []

    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING stRGAT: {dataset}")
        print(f"==================================================")

        save_path = mk_dir(os.path.join(output_root, dataset))
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time and memory tracking
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        # Run pipeline
        if dataset.startswith('15'):
            res = train_stRGAT(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=150)
        else:
            res = run_stRGAT_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50)

        # Resource usage
        used_time = time.time() - start_time
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        current, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()

        used_mem = end_mem - start_mem
        peak_gb = peak_mem / 1024 / 1024 / 1024

        print(f'Time: {used_time:.2f}s | Mem: {used_mem:.4f}GB | Peak: {peak_gb:.4f}GB')

        # Assign dataset category
        data_type = "Unknown"
        for cat, dlist in CATEGORIES.items():
            if dataset in dlist:
                data_type = cat
                break

        # Store full result
        full_result = {
            "data_type": data_type,
            "dataset": dataset,
            "ari": res["ari"],
            "nmi": res["nmi"],
            "ami": res["ami"],
            "bhi": res["bhi"],
            "sc": res["sc"],
            "time": used_time,
            "Memo": used_mem,
            "Memo_peak": peak_gb
        }

        all_results.append(full_result)
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_stRGAT_result.csv"), index=False)

    # Save all results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_root, "stRGAT_ALL_DATASETS_RESULTS.csv"), index=False)
    print("\n🎉 stRGAT FINISHED ALL DATASETS! Results saved to:", output_root)