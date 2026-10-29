import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import psutil
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

# Encoding & warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Random seeds
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device:', device)

# --------------------------
# 1. SpaNCMG Model (UNCHANGED)
# --------------------------
class SpaNCMG_model(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# --------------------------
# 2. Spatial Graph Construction (UNCHANGED)
# --------------------------
def build_spatial_graph(spatial_coords, k=10):
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    adj = np.zeros((spatial_coords.shape[0], spatial_coords.shape[0]))
    for i in range(spatial_coords.shape[0]):
        adj[i, indices[i, 1:]] = 1
    return torch.tensor(adj, dtype=torch.float32)

# --------------------------
# 3. Resolution Search (UNCHANGED)
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    res_list = list(np.arange(0.2, 2.5, increment))
    res_list.reverse()
    for res in res_list:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# 4. ORIGINAL SpaNCMG Train (FOR DLPFC DATA ONLY - UNCHANGED)
# --------------------------
def SpaNCMG_train(adata_h5, n_clusters, device, epochs=200, lr=1e-3, k_spatial=10):
    adata = adata_h5.copy()
    # Preprocess input
    X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Build spatial graph
    spatial_coords = adata.obsm['spatial']
    adj = build_spatial_graph(spatial_coords, k=k_spatial).to(device)

    # Model
    model = SpaNCMG_model(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_recon = nn.MSELoss()
    criterion_spatial = nn.MSELoss()

    # Train with reconstruction + spatial loss
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_recon, z = model(X)
        loss_recon = criterion_recon(x_recon, X)
        z_neighbors = torch.matmul(adj, z) / k_spatial
        loss_spatial = criterion_spatial(z, z_neighbors)
        loss = loss_recon + 0.5 * loss_spatial
        loss.backward()
        optimizer.step()

    # Get embedding
    model.eval()
    with torch.no_grad():
        _, z = model(X)
    embedding = z.cpu().numpy()

    # Normalize embedding
    embedding_scaled = StandardScaler().fit_transform(embedding)

    # Leiden clustering
    adata_emb = sc.AnnData(embedding_scaled)
    adata_emb.obsm['spatial'] = spatial_coords
    sc.pp.neighbors(adata_emb, n_neighbors=15)
    best_res = res_search_fixed_clus(adata_emb, n_clusters)
    sc.tl.leiden(adata_emb, resolution=best_res, key_added="SpaNCMG_leiden")
    clusters = adata_emb.obs['SpaNCMG_leiden'].values

    # Save to adata
    adata.obsm["SpaNCMG_embedding"] = embedding
    adata.obs["SpaNCMG_cluster"] = clusters.astype(str)
    return adata

# --------------------------
# 5. NEW: SpaNCMG Train FOR ALL NON-DLPFC DATASETS (with auto spatial coords + downsampling)
# --------------------------
def SpaNCMG_train_unified(adata_h5, n_clusters, device, epochs=200, lr=1e-3, k_spatial=10):
    adata = adata_h5.copy()
    
    # ✅ DOWNSAMPLE to 25% (SAME AS Spatial_MGCN) to avoid memory error
    adata = adata[adata.obs.sample(frac=0.25, random_state=42).index].copy()
    n = adata.n_obs

    # --------------------------
    # AUTO HANDLE SPATIAL COORDINATES (for non-DLPFC data)
    # --------------------------
    spatial_coords = None
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x','y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x','y']].values
    elif {'spatial_x','spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x','spatial_y']].values

    # Validate numeric coords + generate dummy if missing
    if spatial_coords is not None:
        if not np.issubdtype(spatial_coords.dtype, np.number):
            spatial_coords = None
        else:
            try:
                spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
            except ValueError:
                spatial_coords = None

    # Generate dummy 2D grid coordinates if no valid spatial info
    if spatial_coords is None or spatial_coords.shape[0] != n:
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        adata.obsm['spatial'] = spatial_coords.copy()

    # --------------------------
    # Original SpaNCMG training pipeline
    # --------------------------
    X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Build spatial graph
    adj = build_spatial_graph(spatial_coords, k=k_spatial).to(device)

    # Model
    model = SpaNCMG_model(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_recon = nn.MSELoss()
    criterion_spatial = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_recon, z = model(X)
        loss_recon = criterion_recon(x_recon, X)
        z_neighbors = torch.matmul(adj, z) / k_spatial
        loss_spatial = criterion_spatial(z, z_neighbors)
        loss = loss_recon + 0.5 * loss_spatial
        loss.backward()
        optimizer.step()

    # Embedding
    model.eval()
    with torch.no_grad():
        _, z = model(X)
    embedding = z.cpu().numpy()
    embedding_scaled = StandardScaler().fit_transform(embedding)

    # Clustering
    adata_emb = sc.AnnData(embedding_scaled)
    adata_emb.obsm['spatial'] = spatial_coords
    sc.pp.neighbors(adata_emb, n_neighbors=15)
    best_res = res_search_fixed_clus(adata_emb, n_clusters)
    sc.tl.leiden(adata_emb, resolution=best_res, key_added="SpaNCMG_leiden")
    clusters = adata_emb.obs['SpaNCMG_leiden'].values

    # Save results
    adata.obsm["SpaNCMG_embedding"] = embedding
    adata.obs["SpaNCMG_cluster"] = clusters.astype(str)
    return adata

# --------------------------
# 6. Score Calculation (UNCHANGED)
# --------------------------
def calculate_scores(adata):
    valid_mask = adata.obs['ground_truth'].notna()
    pred_labels = adata.obs['SpaNCMG_cluster'].values
    gt_labels = adata.obs['ground_truth'].values

    ari, nmi, ami, bhi = 0, 0, 0, 0
    if np.any(valid_mask):
        pred_valid = pred_labels[valid_mask]
        gt_valid = gt_labels[valid_mask]
        ari = adjusted_rand_score(gt_valid, pred_valid)
        nmi = normalized_mutual_info_score(gt_valid, pred_valid)
        ami = normalized_mutual_info_score(gt_valid, pred_valid, average_method='arithmetic')
        bhi = homogeneity_score(gt_valid, pred_valid)

    # SC (all cells + predicted labels)
    sc_score = 0.0
    if len(np.unique(pred_labels)) >= 2:
        sc_score = silhouette_score(adata.obsm["SpaNCMG_embedding"], pred_labels, random_state=42)

    # SC_revise (valid cells + ground truth)
    sc_revise = 0.0
    valid_emb = adata.obsm["SpaNCMG_embedding"][valid_mask]
    valid_gt = gt_labels[valid_mask]
    if len(np.unique(valid_gt)) >= 2 and len(valid_emb) > 0:
        sc_revise = silhouette_score(valid_emb, valid_gt, random_state=42)

    return ari, nmi, ami, bhi, sc_score, sc_revise

# --------------------------
# 7. MAIN UNIFIED BENCHMARK (ALL DATASETS)
# --------------------------
sys.path.append('../')
import utils_for_all as usa

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

if __name__ == '__main__':
    # --------------------------
    # DATASET LIST (EXACT SAME AS Spatial_MGCN)
    # --------------------------
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

    # Paths (MATCH YOUR Spatial_MGCN PATHS)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\SpaNCMG"
    mk_dir(OUTPUT_ROOT)

    all_results = []

    # --------------------------
    # RUN FOR ALL DATASETS
    # --------------------------
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING SpaNCMG: {dataset}")
        print(f"==================================================")
        
        save_path = mk_dir(os.path.join(OUTPUT_ROOT, dataset))

        # Load data using YOUR EXISTING utils_for_all loader
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Memory & Time tracking
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024

        # --------------------------
        # RUN MODEL: DLPFC = original, OTHERS = unified
        # --------------------------
        if dataset.startswith('15'):
            adata_result = SpaNCMG_train(adata_h5, n_clusters=n_cluster, device=device)
        else:
            adata_result = SpaNCMG_train_unified(adata_h5, n_clusters=n_cluster, device=device)

        # Calculate metrics
        ari, nmi, ami, bhi, sc_score, sc_revise = calculate_scores(adata_result)

        # Time & Memory
        used_time = time.time() - start_time
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024
        used_mem = end_mem - start_mem
        current, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()
        peak_gb = peak_mem / 1024/1024/1024

        # Print results
        print(f"✅ {dataset} Final Results:")
        print(f"ARI={ari:.4f} | NMI={nmi:.4f} | AMI={ami:.4f} | BHI={bhi:.4f}")
        print(f"SC={sc_score:.4f} | SC_rev={sc_revise:.4f}")
        print(f'Time: {used_time:.2f}s | Mem: {used_mem:.4f}GB | Peak: {peak_gb:.4f}GB')

        # Save adata
        adata_result.write(os.path.join(save_path, f"SpaNCMG_{dataset}.h5ad"))

        # Get data type
        data_type = "Unknown"
        for cat, dlist in CATEGORIES.items():
            if dataset in dlist:
                data_type = cat
                break

        # Full result
        res = {
            "data_type": data_type, "dataset": dataset,
            "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi,
            "sc": sc_score, "sc_revise": sc_revise,
            "time": used_time, "Memo": used_mem, "Memo_peak": peak_gb
        }
        all_results.append(res)

        # Save per-dataset CSV
        pd.DataFrame([res]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    # --------------------------
    # SAVE ALL FINAL RESULTS
    # --------------------------
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "SpaNCMG_ALL_RESULTS.csv"), index=False)
    print("\n🎉 SpaNCMG: ALL DATASETS FINISHED!")