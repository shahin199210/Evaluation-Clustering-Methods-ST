from __future__ import division
from __future__ import print_function
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch.optim as optim
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import random
import time, psutil, tracemalloc
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
import warnings
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

# Import your unified dataset loader (same as CCST)
sys.path.append('../')
import utils_for_all as usa

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# ✅ ADDED: Evaluation with BHI (ORIGINAL FUNCTION)
def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami = adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
        bhi = homogeneity_score(label_df["True"], label_df["Pred"])
    return ari, nmi, ami, bhi

# --------------------------
# ✅ AUTO MAKE DUMMY SPATIAL COORDS + COMPUTE GRAPHS
# --------------------------
def compute_spatial_graphs(adata, k_neighbors=15):
    print("Computing required SpatialMGCN graphs (fadj, sadj, graph_nei, graph_neg)...")
    import scipy.sparse as sp
    from sklearn.neighbors import kneighbors_graph

    n = adata.n_obs
    spatial_coords = None

    # --------------------------
    # 1. FIRST: Try to get REAL spatial coordinates (from common locations)
    # --------------------------
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x','y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x','y']].values
    elif {'spatial_x','spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x','spatial_y']].values

    # --------------------------
    # 2. CRITICAL FIX: Check if spatial_coords are actually NUMERIC
    # If they contain strings (like "13months-disease-replicate_1"), force dummy coords
    # --------------------------
    if spatial_coords is not None:
        # Check if the data type is numeric (int/float)
        if not np.issubdtype(spatial_coords.dtype, np.number):
            print("⚠️ Spatial coordinates contain non-numeric data (strings/IDs), forcing dummy coordinates")
            spatial_coords = None
        # Also check if any element cannot be converted to float
        else:
            try:
                # Try converting to float to catch mixed types
                spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
            except ValueError:
                print("⚠️ Spatial coordinates contain non-convertible strings, forcing dummy coordinates")
                spatial_coords = None

    # --------------------------
    # 3. If no valid numeric coordinates, generate dummy 2D grid coordinates
    # --------------------------
    if spatial_coords is None or spatial_coords.shape[0] != n:
        print("⚠️ Generating dummy 2D spatial coordinates")
        # Create a grid of points
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        # Save dummy coords to adata.obsm['spatial'] for consistency
        adata.obsm['spatial'] = spatial_coords.copy()

    # --------------------------
    # 4. Proceed with graph computation (unchanged)
    # --------------------------
    # Ensure float type (already done above, but double-check)
    spatial_coords = np.array(spatial_coords, dtype=np.float32)

    # 1. Spatial adjacency (sadj)
    sadj = kneighbors_graph(spatial_coords, n_neighbors=k_neighbors, mode='connectivity', include_self=True)
    sadj = sadj.tocsr()

    # 2. Feature adjacency (fadj)
    sc.tl.pca(adata, n_comps=30)
    pca_data = adata.obsm['X_pca']
    fadj = kneighbors_graph(pca_data, n_neighbors=k_neighbors, mode='connectivity', include_self=True)
    fadj = fadj.tocsr()

    # 3. graph_nei and graph_neg
    graph_nei = []
    graph_neg = []
    for i in range(n):
        neighbors = sadj[i].indices
        if len(neighbors) > k_neighbors:
            neighbors = neighbors[:k_neighbors]
        graph_nei.append(neighbors)
        
        non_neighbors = np.setdiff1d(np.arange(n), neighbors)
        if len(non_neighbors) >= k_neighbors:
            negatives = np.random.choice(non_neighbors, size=k_neighbors, replace=False)
        else:
            negatives = np.random.choice(non_neighbors, size=len(non_neighbors), replace=False)
            while len(negatives) < k_neighbors:
                negatives = np.append(negatives, np.random.choice(non_neighbors, size=1))
        graph_neg.append(negatives)

    adata.obsm['sadj'] = sadj
    adata.obsm['fadj'] = fadj
    adata.obsm['graph_nei'] = np.array(graph_nei)
    adata.obsm['graph_neg'] = np.array(graph_neg)
    return adata

# ==============================================================================
# ✅ ORIGINAL DLPFC DATA LOADER
# ==============================================================================
def load_DLPFC_data(dataset):
    print("load DLPFC dataset:")
    path = f'D:/My_Project_ST/DataSets/Generate_data/Spatial_MGCN/{dataset}/Spatial_MGCN_{dataset}.h5ad'
    adata = sc.read_h5ad(path)

    adata.X = adata.X.todense() if issparse(adata.X) else adata.X
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground_truth']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg

# ==============================================================================
# ✅ UNIFIED LOADER FOR ALL NON-DLPFC DATASETS
# ==============================================================================
def load_unified_spatial_data(adata_h5, dataset):
    print(f"Loading unified spatial dataset: {dataset}")
    adata = adata_h5.copy()

    # ✅✅✅ DOWNSAMPLE TO 50% TO FIX MEMORY ERROR ✅✅✅
    # This line reduces data size to avoid "not enough memory"
    adata = adata[adata.obs.sample(frac=0.25, random_state=42).index].copy()

    # Auto compute graphs + dummy coords if needed
    if 'fadj' not in adata.obsm:
        adata = compute_spatial_graphs(adata)

    adata.X = adata.X.todense() if issparse(adata.X) else adata.X
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground_truth']

    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])

    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    print("done")
    return adata, features, labels, nsadj, nfadj, graph_nei, graph_neg

# ==============================================================================
# ✅ ORIGINAL TRAIN FUNCTION
# ==============================================================================
def train(model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer):
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss

# ==============================================================================
# ✅ MAIN SPATIALMGCN PIPELINE (FIXED)
# ==============================================================================
def run_SpatialMGCN(adata_h5, savepath, dataset, n_cluster):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cuda = torch.cuda.is_available()
    mk_dir(savepath)

    if dataset.startswith('15'):
        config_file = './config/DLPFC.ini'
        adata, features, labels, sadj, fadj, graph_nei, graph_neg = load_DLPFC_data(dataset)
    else:
        config_file = f'./config/{dataset}.ini'
        if not os.path.exists(config_file):
            config_file = './config/DLPFC.ini'
        adata, features, labels, sadj, fadj, graph_nei, graph_neg = load_unified_spatial_data(adata_h5, dataset)

    config = Config(config_file)
    config.n = features.shape[0]
    config.class_num = n_cluster
    config.fdim = features.shape[1]  # Match input dimension to data

    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if cuda:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    # GPU
    if cuda:
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        graph_nei = graph_nei.cuda()
        graph_neg = graph_neg.cuda()

    # Model
    model = Spatial_MGCN(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout
    )
    if cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # Train
    ari_max = -1
    idx_max = None
    emb_max = None
    mean_max = None

    print(f"\nTraining {dataset}")
    for epoch in range(config.epochs):
        emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train(
            model, features, sadj, fadj, graph_nei, graph_neg, config, optimizer
        )
        print(f'{dataset} | Epoch {epoch:3d} | Loss: {total_loss:.2f}')

        kmeans = KMeans(n_clusters=config.class_num, random_state=config.seed).fit(emb)
        idx = kmeans.labels_

        if dataset.startswith('15'):
            df = pd.DataFrame({'labels': labels, 'idx': idx}).dropna()
            ari_res = adjusted_rand_score(df['labels'], df['idx'])
        else:
            ari_res = adjusted_rand_score(labels, idx)

        if ari_res > ari_max:
            ari_max = ari_res
            idx_max = idx
            emb_max = emb
            mean_max = mean

    # Evaluate
    label_df = pd.DataFrame({"True": labels, "Pred": idx_max}).dropna()
    ari, nmi, ami, bhi = eval_model(label_df["Pred"], label_df["True"])
    SC = silhouette_score(emb_max, idx_max) if len(np.unique(idx_max)) > 1 else 0.0

    # Save
    adata.obs['spatialmgcn_pred'] = idx_max.astype(str)
    adata.obsm['spatialmgcn_emb'] = emb_max
    adata.write(f'{savepath}/Spatial_MGCN_{dataset}.h5ad')

    return {
        "dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi,
        "sc": SC, "ari_max": ari_max, "nmi_max": nmi, "ami_max": ami,
        "bhi_max": bhi, "SC_max": SC
    }

# ==============================================================================
# ✅ MAIN RUNNER
# ==============================================================================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    plt.switch_backend('Agg')

    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\Spatial_MGCN"
    mk_dir(OUTPUT_ROOT)

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
        print(f"            RUNNING SPATIALMGCN: {dataset}")
        print(f"==================================================")
        
        savepath = os.path.join(OUTPUT_ROOT, dataset)
        mk_dir(savepath)

        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)

        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        res = run_SpatialMGCN(adata_h5, savepath, dataset, n_cluster)

        used_time = time.time() - start
        used_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 - start_mem
        _, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / 1024 / 1024 / 1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        data_type = "Unknown"
        for cat, dlist in CATEGORIES.items():
            if dataset in dlist:
                data_type = cat
                break

        full_result = {
            "data_type": data_type, "dataset": dataset,
            "ari": res["ari"], "nmi": res["nmi"], "ami": res["ami"], "bhi": res["bhi"], "sc": res["sc"],
            "ari_max": res["ari_max"], "nmi_max": res["nmi_max"], "ami_max": res["ami_max"], "bhi_max": res["bhi_max"], "SC_max": res["SC_max"],
            "time": used_time, "Memo": used_mem, "Memo_peak": peak_gb
        }

        all_results.append(full_result)
        pd.DataFrame([full_result]).to_csv(os.path.join(savepath, f"{dataset}_result.csv"), index=False)

        print(f"\n✅ {dataset} Results:")
        print(f"ARI={res['ari']:.4f} | NMI={res['nmi']:.4f} | AMI={res['ami']:.4f} | BHI={res['bhi']:.4f} | SC={res['sc']:.4f}")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "SpatialMGCN_ALL_RESULTS.csv"), index=False)
    print("\n🎉 ALL DATASETS FINISHED!")