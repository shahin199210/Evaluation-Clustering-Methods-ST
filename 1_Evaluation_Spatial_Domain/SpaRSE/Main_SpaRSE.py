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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# Basic Setup
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device:', device)

# --------------------------
# SpaRSE Model (Spatial Representation Learning) - UNCHANGED
# --------------------------
class SpaRSE(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, hidden_dim=128):
        super(SpaRSE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# --------------------------
# Resolution Search (Same as SEDR) - UNCHANGED
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# Evaluation Metrics - UNCHANGED
# --------------------------
def eval_model(pred, gt):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, nmi, ami

# --------------------------
# ✅ ORIGINAL: SpaRSE for DLPFC DATA (151xxx) - UNCHANGED
# --------------------------
def run_SpaRSE_DLPFC(adata_h5, save_path, dataset, epochs=50):
    os.makedirs(save_path, exist_ok=True)

    # Preprocessing (matches SEDR logic)
    adata = adata_h5.copy()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=200)

    # Fix negative stride error
    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    # Model
    model = SpaRSE(input_dim=200).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    print("==== Training SpaRSE (DLPFC) ====")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Get embedding
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X).cpu().numpy()

    # Create result anndata
    adata_result = anndata.AnnData(embedding)
    adata_result.obsm['spatial'] = adata_h5.obsm['spatial']
    adata_result.obs['ground_truth'] = adata_h5.obs['ground_truth'].values

    # Clustering
    sc.pp.neighbors(adata_result, n_neighbors=20)
    sc.tl.umap(adata_result)
    n_clus = len(np.unique(adata_h5.obs['ground_truth'].dropna()))
    res = res_search_fixed_clus(adata_result, n_clus)
    sc.tl.leiden(adata_result, key_added="SpaRSE_leiden", resolution=res)

    # Save h5ad
    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Save plot
    try:
        sc.pl.spatial(adata_result, color="SpaRSE_leiden", img_key="hires", show=False)
        plt.savefig(os.path.join(save_path, "SpaRSE_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
        plt.close()
    except:
        print("⚠️ Spatial plot not available")

    # Evaluate
    valid = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['SpaRSE_leiden'][valid]
    gt = adata_result.obs['ground_truth'][valid]

    if len(pred) == 0:
        ari, nmi, ami, bhi, sc_score = 0,0,0,0,0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        from sklearn.metrics import homogeneity_score, silhouette_score
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata_result.X[valid], pred) if len(np.unique(pred))>1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# --------------------------
# ✅ NEW: SpaRSE for ALL NON-DLPFC DATASETS (Unified)
# --------------------------
def run_SpaRSE_All(adata_h5, save_path, dataset, epochs=200):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()

    # ✅ MEMORY FIX: Downsample 25% (same as SpatialMGCN)
    adata = adata[adata.obs.sample(frac=0.25, random_state=42).index].copy()

    # ✅ AUTO ADD DUMMY SPATIAL COORDINATES (if missing)
    if 'spatial' not in adata.obsm:
        print("⚠️ Generating dummy spatial coordinates for non-DLPFC data")
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])
        adata.obsm['spatial'] = dummy_coords[:n]

    # ✅ SAME PREPROCESSING AS DLPFC
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    
    # ✅ EXACT FIX: Dynamically set PCA components to avoid exceeding sample count
    max_pca_components = min(200, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=max_pca_components)

    # Tensor
    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    # Model & Training
    model = SpaRSE(input_dim=max_pca_components).to(device)  # Match input dim to PCA components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("==== Training SpaRSE (Unified Non-DLPFC) ====")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Embedding
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X).cpu().numpy()

    # Result
    adata_result = anndata.AnnData(embedding)
    adata_result.obsm['spatial'] = adata.obsm['spatial']
    adata_result.obs['ground_truth'] = adata.obs['ground_truth'].values

    # Clustering
    sc.pp.neighbors(adata_result, n_neighbors=20)
    sc.tl.umap(adata_result)
    n_clus = len(np.unique(adata.obs['ground_truth'].dropna()))
    res = res_search_fixed_clus(adata_result, n_clus)
    sc.tl.leiden(adata_result, key_added="SpaRSE_leiden", resolution=res)

    # Save
    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Plot
    try:
        sc.pl.spatial(adata_result, color="SpaRSE_leiden", show=False)
        plt.savefig(os.path.join(save_path, "SpaRSE_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
        plt.close()
    except:
        print("⚠️ Spatial plot not available")

    # Evaluate
    valid = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['SpaRSE_leiden'][valid]
    gt = adata_result.obs['ground_truth'][valid]

    if len(pred) == 0:
        ari, nmi, ami, bhi, sc_score = 0,0,0,0,0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        from sklearn.metrics import homogeneity_score, silhouette_score
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata_result.X[valid], pred) if len(np.unique(pred))>1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# --------------------------
# Load your unified data loader (SAME AS SpatialMGCN)
# --------------------------
sys.path.append('../')
import utils_for_all as usa

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# --------------------------
# ✅ MAIN RUN (ALL DATASETS - SAME STRUCTURE AS SpatialMGCN)
# --------------------------
if __name__ == '__main__':
    plt.switch_backend('Agg')
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    output_root = r"D:\My_Project_ST\Output_For_All\SpaRSE"
    mk_dir(output_root)

    # ✅ FULL DATASET CATEGORIES (SAME AS YOUR SpatialMGCN CODE)
    CATEGORIES = {
        "DLPFC": ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'],
        "10X Visium": ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'],
        "ST": ["Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"],
        "Slide-seq": ['Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'],
        "SNARE-seq2": ['LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'],
        "STARmap": ["STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"],
        "Xenium": ['BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158']
    }

    # ✅ ALL DATASETS LIST
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

    # --------------------------
    # Run every dataset
    # --------------------------
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING SpaRSE: {dataset}")
        print(f"==================================================")
        
        save_path = mk_dir(os.path.join(output_root, dataset))

        # ✅ LOAD DATA VIA YOUR utils_for_all.py
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time & Memory tracking
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024

        # ✅ AUTO CHOOSE FUNCTION: DLPFC vs Others
        if dataset.startswith('15'):
            res = run_SpaRSE_DLPFC(adata_h5, save_path, dataset, epochs=50)
        else:
            res = run_SpaRSE_All(adata_h5, save_path, dataset, epochs=50)

        # Calculate resources
        used_time = time.time() - start_time
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024
        current, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()

        used_mem = end_mem - start_mem
        peak_gb = peak_mem / 1024/1024/1024

        print(f'Time: {used_time:.2f}s | Mem: {used_mem:.4f}GB | Peak: {peak_gb:.4f}GB')

        # Add dataset category
        data_type = "Unknown"
        for cat, dlist in CATEGORIES.items():
            if dataset in dlist:
                data_type = cat
                break

        # Full result
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
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    # ✅ Save final full results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_root, "SpaRSE_ALL_DATASETS_RESULTS.csv"), index=False)
    print("\n🎉 SpaRSE FINISHED ALL DATASETS! Results saved to:", output_root)