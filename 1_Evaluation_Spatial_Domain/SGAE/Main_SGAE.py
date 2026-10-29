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
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, homogeneity_score

# --------------------------
# Basic Setup (Consistent with DeepGFT)
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('===== Using device:', device)

# --------------------------
# SGAE Core Model (from STOmics/SGAE)
# --------------------------
class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAE(nn.Module):
    def __init__(self, nfeat, nhid, nemb, dropout):
        super(GAE, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nemb)
        self.dropout = dropout

    def encode(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def forward(self, x, adj):
        emb = self.encode(x, adj)
        adj_recon = self.reconstruct(emb)
        return emb, adj_recon

    def reconstruct(self, emb):
        return torch.sigmoid(torch.mm(emb, emb.t()))

class SpatialGraphAutoEncoder(nn.Module):
    """SGAE with spatial graph constraint (core from STOmics/SGAE)"""
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=32, dropout=0.1):
        super(SpatialGraphAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        # Graph Convolution for spatial constraint
        self.gcn = GAE(embedding_dim, hidden_dim//2, embedding_dim, dropout)
        
    def forward(self, x, adj):
        # Feature encoding
        feat_emb = self.encoder(x)
        # Spatial graph encoding
        spatial_emb, adj_recon = self.gcn(feat_emb, adj)
        # Feature reconstruction
        x_recon = self.decoder(spatial_emb)
        return spatial_emb, x_recon, adj_recon

# --------------------------
# Utility Functions (Consistent with DeepGFT)
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

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

def normalize_adjacency(adj):
    """Normalize adjacency matrix (required for GCN)"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # Convert to torch sparse tensor
    adj_normalized = torch.sparse_coo_tensor(
        np.vstack((adj_normalized.row, adj_normalized.col)),
        adj_normalized.data,
        adj_normalized.shape,
        dtype=torch.float32
    ).to(device)
    return adj_normalized

def get_spatial_adjacency(adata, rad_cutoff=150):
    """Create spatial adjacency matrix (compatible with SGAE)"""
    if 'spatial' not in adata.obsm:
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])[:n]
        adata.obsm['spatial'] = dummy_coords
    
    coords = adata.obsm['spatial']
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=rad_cutoff)
    
    # Fallback for zero pairs
    if len(pairs) == 0:
        distances, indices = tree.query(coords, k=5)
        pairs = []
        for i in range(coords.shape[0]):
            for j in indices[i, 1:]:
                pairs.append((i, j))
        pairs = set(pairs)
    
    # Build adjacency matrix
    adj = sp.lil_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
    for i, j in pairs:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj = adj.tocoo()
    return normalize_adjacency(adj)

# --------------------------
# SGAE Training Pipeline
# --------------------------
def train_SGAE(adata, save_path, dataset, random_seed=0, rad_cutoff=150, epochs=200, lr=0.001):
    os.makedirs(save_path, exist_ok=True)
    adata = adata.copy()

    # Preprocessing (match DeepGFT)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Prepare data
    n_obs, n_vars = adata.shape
    features = torch.tensor(
        adata.X.todense() if sp.issparse(adata.X) else adata.X,
        dtype=torch.float32
    ).to(device)
    adj = get_spatial_adjacency(adata, rad_cutoff=rad_cutoff)

    # Initialize model
    model = SpatialGraphAutoEncoder(
        input_dim=n_vars,
        hidden_dim=128,
        embedding_dim=32,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_recon = nn.MSELoss()  # Feature reconstruction loss
    criterion_adj = nn.BCELoss()    # Adjacency reconstruction loss

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        emb, x_recon, adj_recon = model(features, adj)
        
        # Loss calculation
        loss_recon = criterion_recon(x_recon, features)
        loss_adj = criterion_adj(adj_recon, adj.to_dense())
        loss = loss_recon + 0.1 * loss_adj  # Balance losses
        
        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {loss.item():.4f} | Recon Loss: {loss_recon.item():.4f} | Adj Loss: {loss_adj.item():.4f}")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        emb, _, _ = model(features, adj)
    adata.obsm['SGAE'] = emb.cpu().numpy()

    # Clustering (Leiden, same as DeepGFT)
    sc.pp.neighbors(adata, use_rep='SGAE')
    sc.tl.umap(adata)
    n_clus = len(np.unique(adata.obs["ground_truth"].dropna()))
    res = res_search_fixed_clus(adata, n_clus)
    sc.tl.leiden(adata, key_added="SGAE_leiden", resolution=res, random_state=0)

    # Save results
    adata.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Evaluation
    valid = adata.obs['ground_truth'].notna()
    pred = adata.obs['SGAE_leiden'][valid]
    gt = adata.obs['ground_truth'][valid]
    ari, nmi, ami = eval_model(pred, gt)
    bhi = homogeneity_score(gt, pred) if len(pred) > 0 else 0
    sc_score = silhouette_score(adata.obsm["SGAE"][valid], pred) if len(np.unique(pred))>1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# --------------------------
# OOM-Safe SGAE Runner (Consistent with DeepGFT)
# --------------------------
def run_SGAE_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()

    # Downsample if needed (OOM fix)
    n_total = adata.n_obs
    target_min, target_max = 1000, 1200
    if n_total > target_max:
        target = (target_min + target_max) // 2
        adata = adata[adata.obs.sample(frac=target/n_total, random_state=42).index].copy()
        print(f"⚠️ Downsampled {dataset} to {adata.n_obs} cells")

    # Fix spatial coordinates
    if 'spatial' not in adata.obsm or adata.obsm['spatial'].shape[1] != 2:
        if 'spatial' in adata.obsm:
            adata.obsm['spatial'] = adata.obsm['spatial'][:, :2]
        else:
            n = adata.n_obs
            side = int(np.ceil(np.sqrt(n)))
            xx, yy = np.meshgrid(np.arange(side), np.arange(side))
            adata.obsm['spatial'] = np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])[:n]

    # Use CPU for large datasets
    global device
    if adata.n_obs > 5000:
        device = torch.device('cpu')
        print("⚠️ Switching to CPU for large dataset")

    return train_SGAE(adata, save_path, dataset, random_seed=random_seed, rad_cutoff=rad_cutoff)

# --------------------------
# Load Data Loader (Same as DeepGFT)
# --------------------------
sys.path.append('../')
import utils_for_all as usa

# --------------------------
# MAIN RUN (ALL DATASETS)
# --------------------------
if __name__ == '__main__':
    plt.switch_backend('Agg')
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    output_root = r"D:\My_Project_ST\Output_For_All\SGAE"
    mk_dir(output_root)

    # Dataset categories (identical to DeepGFT)
    CATEGORIES = {
        "DLPFC": ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'],
        "10X Visium": ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'],
        "ST": ["Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"],
        "Slide-seq": ['Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'],
        "SNARE-seq2": ['LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'],
        "STARmap": ["STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"],
        "Xenium": ['BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158']
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
        print(f"            RUNNING SGAE: {dataset}")
        print(f"==================================================")
        
        save_path = mk_dir(os.path.join(output_root, dataset))
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time & Memory tracking (same as DeepGFT)
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024

        # Choose pipeline (same rad_cutoff logic as DeepGFT)
        if dataset.startswith('15'):
            res = train_SGAE(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=150)
        else:
            res = run_SGAE_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50)

        # Resource usage
        used_time = time.time() - start_time
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024
        current, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()

        used_mem = end_mem - start_mem
        peak_gb = peak_mem / 1024/1024/1024

        print(f'Time: {used_time:.2f}s | Mem: {used_mem:.4f}GB | Peak: {peak_gb:.4f}GB')

        # Add category info
        data_type = "Unknown"
        for cat, dlist in CATEGORIES.items():
            if dataset in dlist:
                data_type = cat
                break

        # Full result entry (same format as DeepGFT)
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
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_SGAE_result.csv"), index=False)

    # Save all results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_root, "SGAE_ALL_DATASETS_RESULTS.csv"), index=False)
    print("\n🎉 SGAE FINISHED ALL DATASETS! Results saved to:", output_root)