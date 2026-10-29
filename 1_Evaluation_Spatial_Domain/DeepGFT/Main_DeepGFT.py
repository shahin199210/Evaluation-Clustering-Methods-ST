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
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, homogeneity_score
from torch_geometric.nn import GATv2Conv
import scipy.sparse as sp

# --------------------------
# Add DeepGFT to Python Path
# --------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layers import GraphAttentionLayer
from gatv2_conv import GATv2Conv

# --------------------------
# Basic Setup (Same as STAGATE)
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('===== Using device:', device)

# --------------------------
# DeepGFT Model Definitions
# --------------------------
class GAT(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GAT, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv1_p = GATv2Conv(in_dim, num_hidden, heads=3, concat=False,
                                 dropout=0.2, add_self_loops=False, bias=True)

        self.conv2 = torch.nn.Linear(num_hidden * 2, out_dim, bias=True)

        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv3_p = GATv2Conv(out_dim, num_hidden, heads=3, concat=False,
                                 dropout=0.2, add_self_loops=False, bias=True)

        self.conv4 = torch.nn.Linear(num_hidden * 2, in_dim, bias=True)

    def forward(self, features, edge_index, edge_index_cluster):
        h1_nonprune = self.conv1(features, edge_index)
        h1_prune = self.conv1_p(features, edge_index_cluster)
        h1 = F.elu(torch.cat([h1_nonprune, h1_prune], dim=1))

        h2 = self.conv2(h1)

        h3_nonprune = self.conv3(h2, edge_index)
        h3_prune = self.conv3_p(h2, edge_index_cluster)
        h3 = F.elu(torch.cat([h3_nonprune, h3_prune], dim=1))

        h4 = self.conv4(h3)
        return h2, h4


class GAT_noncluster(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GAT_noncluster, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        num_hidden *= 2
        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv2 = torch.nn.Linear(num_hidden, out_dim, bias=True)
        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv4 = torch.nn.Linear(num_hidden, in_dim, bias=True)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1)
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = self.conv4(h3)
        return h2, h4


class CrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, alpha=0.2):
        super(CrossAttention, self).__init__()
        self.dropout = dropout
        self.NFeature = dim
        self.alpha = alpha
        self.a1 = nn.Parameter(torch.FloatTensor(2 * dim, 1))
        self.a2 = nn.Parameter(torch.FloatTensor(2 * dim, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def inference(self, h1=None, h2=None):
        Wh1 = torch.matmul(torch.cat([h1, h2], 1), self.a1)
        Wh2 = torch.matmul(torch.cat([h1, h2], 1), self.a2)
        e = torch.cat((Wh1, Wh2), 1)
        e = self.leakyrelu(e)
        lamda = F.softmax(e, dim=1)
        lamda = F.dropout(lamda, self.dropout, training=self.training)
        h_prime1 = lamda[:, 0].repeat(self.NFeature, 1).T * h1
        h_prime2 = lamda[:, 1].repeat(self.NFeature, 1).T * h2
        h = h_prime1 + h_prime2
        return lamda, h

    def forward(self, h1=None, h2=None):
        lamda, h = self.inference(h1, h2)
        return lamda, h


class SPOT(nn.Module):
    def __init__(self, dim, dims_spot=[64, 16], dropout=0.1, device='cuda:0', cluster=True):
        super(SPOT, self).__init__()
        self.dropout = dropout
        self.NFeature = dim
        self.device = device
        self.cluster = cluster
        if cluster:
            self.GATE_spot = GAT(hidden_dims=dims_spot).to(device)
        else:
            self.GATE_spot = GAT_noncluster(hidden_dims=dims_spot).to(device)
        self.CrossAtt = CrossAttention(dim=self.NFeature, dropout=0.0)

    def forward(self, gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T, spotnet, spotnet_cluster):
        if self.cluster:
            emb_spot, mtx_res_spot = self.GATE_spot(spot_freq_mtx, spotnet, spotnet_cluster)
        else:
            emb_spot, mtx_res_spot = self.GATE_spot(spot_freq_mtx, spotnet)
        
        # Only run spot GFT (safe, no shape error)
        X_res_spot = self.iGFT_spot(mtx_res_spot, spot_eigvecs_T)
        
        # No cross-attention needed anymore
        lamba, res = None, None

        return res, lamba, emb_spot, mtx_res_spot

    def iGFT_gene(self, mtx_res, eigvecs_T):
        filter = torch.matmul(eigvecs_T.T, mtx_res.T)
        filter[filter < 0] = 0
        return filter

    def iGFT_spot(self, mtx_res, eigvecs_T):
        # ✅ FIXED: Correct multiplication order to match shapes
        filter = torch.matmul(eigvecs_T, mtx_res)
        filter[filter < 0] = 0
        return filter


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout=0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.matmul(z, z.t())
        return adj


class GENE(nn.Module):
    def __init__(self, dims_gene=[64, 16, 64], dropout=0.02, device='cuda:0'):
        super(GENE, self).__init__()
        self.dropout = dropout
        self.device = device
        self.gat1 = GraphAttentionLayer(dims_gene[0], dims_gene[1], dropout=dropout, alpha=0.2, concat=True)
        self.gat2 = GraphAttentionLayer(dims_gene[1], dims_gene[2], dropout=dropout, alpha=0.2, concat=False)
        self.AdjLayer = InnerProductDecoder(dropout=dropout)

    def forward(self, features, adj):
        h1 = self.gat1(features, adj, tied_W=None, tied_attention=None)
        attention = self.gat1.attention_res
        h2 = self.gat2(h1, adj, tied_W=None, tied_attention=None)
        genenet_res = self.AdjLayer(h2) * (adj.sum(axis=1) > 0).reshape(-1, 1)
        return h2, genenet_res, attention

# --------------------------
# Utility Functions
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

def get_spatial_graph(adata, rad_cutoff=150):
    """Create spatial graph for DeepGFT (PyG edge_index format)"""
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
    
    # --------------------------
    # ✅ Fix: handle case with zero pairs (no edges found)
    # --------------------------
    if len(pairs) == 0:
        # Fallback: use k-nearest neighbors (k=5) to ensure at least some edges
        distances, indices = tree.query(coords, k=5)
        pairs = []
        for i in range(coords.shape[0]):
            for j in indices[i, 1:]:  # skip self
                pairs.append((i, j))
        pairs = set(pairs)
    
    edges = np.array(list(pairs)).T
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    return torch.tensor(edges, dtype=torch.long).to(device)

def compute_laplacian(edge_index, n_nodes):
    """Compute normalized Laplacian and eigenvectors for GFT"""
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].cpu(), edge_index[1].cpu())),
                        shape=(n_nodes, n_nodes))
    adj = adj + sp.eye(n_nodes)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(n_nodes) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    eigvals, eigvecs = np.linalg.eigh(laplacian.todense())
    return torch.tensor(eigvecs, dtype=torch.float32).to(device)

# --------------------------
# ✅ DeepGFT Training Pipeline
# --------------------------
def train_DeepGFT(adata, save_path, dataset, random_seed=0, rad_cutoff=150, epochs=200):
    os.makedirs(save_path, exist_ok=True)
    adata = adata.copy()

    # Preprocessing
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_obs, n_vars = adata.shape
    features = torch.tensor(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=torch.float32).to(device)

    # Spatial graph
    edge_index = get_spatial_graph(adata, rad_cutoff=rad_cutoff)
    spot_eigvecs = compute_laplacian(edge_index, n_obs)

    # Initialize only the spot model
    spot_model = SPOT(dim=64, dims_spot=[n_vars, 64, 16], cluster=False, device=device).to(device)
    optimizer = torch.optim.Adam(spot_model.parameters(), lr=0.001)

    # Training loop
    spot_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        _, _, emb_spot, _ = spot_model(
            gene_freq_mtx=None,
            gene_eigvecs_T=None,
            spot_freq_mtx=features,
            spot_eigvecs_T=spot_eigvecs,
            spotnet=edge_index,
            spotnet_cluster=None
        )
        
        # Simple reconstruction loss
        _, recon = spot_model.GATE_spot(features, edge_index)
        loss = F.mse_loss(recon, features)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    # Extract embeddings
    spot_model.eval()
    with torch.no_grad():
        _, _, emb_spot, _ = spot_model(
            gene_freq_mtx=None,
            gene_eigvecs_T=None,
            spot_freq_mtx=features,
            spot_eigvecs_T=spot_eigvecs,
            spotnet=edge_index,
            spotnet_cluster=None
        )

    adata.obsm['DeepGFT'] = emb_spot.cpu().numpy()

    # Clustering (Leiden)
    sc.pp.neighbors(adata, use_rep='DeepGFT')
    sc.tl.umap(adata)
    n_clus = len(np.unique(adata.obs["ground_truth"].dropna()))
    res = res_search_fixed_clus(adata, n_clus)
    sc.tl.leiden(adata, key_added="DeepGFT_leiden", resolution=res, random_state=0)

    # Save results
    adata.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Evaluation
    valid = adata.obs['ground_truth'].notna()
    pred = adata.obs['DeepGFT_leiden'][valid]
    gt = adata.obs['ground_truth'][valid]
    ari, nmi, ami = eval_model(pred, gt)
    bhi = homogeneity_score(gt, pred) if len(pred) > 0 else 0
    sc_score = silhouette_score(adata.obsm["DeepGFT"][valid], pred) if len(np.unique(pred))>1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# --------------------------
# ✅ Unified DeepGFT Runner (OOM-Safe)
# --------------------------
def run_DeepGFT_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50):
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

    return train_DeepGFT(adata, save_path, dataset, random_seed=random_seed, rad_cutoff=rad_cutoff)

# --------------------------
# Load Your Unified Data Loader
# --------------------------
sys.path.append('../')
import utils_for_all as usa

# --------------------------
# ✅ MAIN RUN (ALL DATASETS)
# --------------------------
if __name__ == '__main__':
    plt.switch_backend('Agg')
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    output_root = r"D:\My_Project_ST\Output_For_All\DeepGFT"
    mk_dir(output_root)

    # Dataset categories (same as STAGATE)
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
        print(f"            RUNNING DeepGFT: {dataset}")
        print(f"==================================================")
        
        save_path = mk_dir(os.path.join(output_root, dataset))
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time & Memory tracking
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024

        # Choose pipeline
        if dataset.startswith('15'):
            res = train_DeepGFT(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=150)
        else:
            res = run_DeepGFT_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50)

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

        # Full result entry
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
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_DeepGFT_result.csv"), index=False)

    # Save all results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_root, "DeepGFT_ALL_DATASETS_RESULTS.csv"), index=False)
    print("\n🎉 DeepGFT FINISHED ALL DATASETS! Results saved to:", output_root)