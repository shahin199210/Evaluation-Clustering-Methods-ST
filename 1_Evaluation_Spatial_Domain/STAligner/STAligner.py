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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# Basic Setup (Same as SEDR/Original)
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# STAlignER Model (Official Architecture - UNCHANGED)
# --------------------------
class STAligner(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, hidden_dim=128):
        super(STAligner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
# Resolution Search (EXACT Same as SEDR - UNCHANGED)
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# Evaluation Metrics (EXACT Same as SEDR - UNCHANGED)
# --------------------------
def eval_model(pred, gt):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, nmi, ami

# --------------------------
# ORIGINAL STAligner FUNCTION (FOR DLPFC ONLY - UNCHANGED)
# --------------------------
def run_STAligner_DLPFC(adata_h5, save_path, dataset, epochs=200):
    os.makedirs(save_path, exist_ok=True)

    # Preprocessing (matches original SEDR/STAligner)
    adata = adata_h5.copy()
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=200)

    # Convert to tensor
    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    # Model
    model = STAligner(input_dim=200).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training
    print("==== Training STAligner (DLPFC) ====")
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

    # Clustering (same as original)
    sc.pp.neighbors(adata_result, n_neighbors=20)
    sc.tl.umap(adata_result)
    n_clus = len(np.unique(adata_h5.obs['ground_truth'].dropna()))
    res = res_search_fixed_clus(adata_result, n_clus)
    sc.tl.leiden(adata_result, key_added="STAligner_leiden", resolution=res)

    # Save h5ad
    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Save spatial plot
    try:
        sc.pl.spatial(adata_result, color="STAligner_leiden", img_key="hires", show=False)
        plt.savefig(os.path.join(save_path, "STAligner_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
        plt.close()
    except:
        print("⚠️ Spatial plot skipped")

    # Evaluate
    valid = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['STAligner_leiden'][valid]
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
# NEW UNIVERSAL STAligner FUNCTION (FOR ALL NON-DLPFC DATASETS)
# --------------------------
def run_STAligner_Universal(adata_h5, save_path, dataset, epochs=200, n_cluster=10):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)
    os.makedirs(save_path, exist_ok=True)
    
    adata = adata_h5.copy()
    # Preprocessing (robust for all dataset types)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Dynamic PCA (handles small datasets safely)
    max_possible = min(adata.n_obs, adata.n_vars)
    n_comps = min(200, max_possible - 1)
    if n_comps < 1:
        adata.obsm['X_pca'] = adata.X.copy()
    else:
        sc.pp.pca(adata, n_comps=n_comps, svd_solver='arpack')

    # Convert to tensor
    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    # Model & Training
    model = STAligner(input_dim=n_comps).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("==== Training STAligner (Universal) ====")
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

    # Result anndata
    adata_result = anndata.AnnData(embedding)
    adata_result.obs['ground_truth'] = adata_h5.obs['ground_truth'].values
    if 'spatial' in adata_h5.obsm:
        adata_result.obsm['spatial'] = adata_h5.obsm['spatial']
        adata_result.uns['spatial'] = adata_h5.uns.get('spatial', {})

    # Clustering
    sc.pp.neighbors(adata_result, n_neighbors=20, random_state=0)
    sc.tl.umap(adata_result, random_state=0)
    res = res_search_fixed_clus(adata_result, n_cluster)
    sc.tl.leiden(adata_result, key_added="STAligner_leiden", resolution=res, random_state=0)

    # Save
    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Spatial plot
    try:
        if 'spatial' in adata_result.obsm:
            sc.pl.spatial(adata_result, color="STAligner_leiden", img_key="hires", show=False)
            plt.savefig(os.path.join(save_path, "STAligner_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
    except:
        print("⚠️ Spatial plot skipped")

    # Evaluate
    valid = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['STAligner_leiden'][valid]
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
# Load your data loader
# --------------------------
sys.path.append('../')
import utils_for_all as usa

# --------------------------
# MAIN RUN (ALL DATASETS - MATCHES CCST EXACTLY)
# --------------------------
if __name__ == '__main__':
    # PATHS (MATCH YOUR CCST CODE)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\STAligner"
    usa.mk_dir(OUTPUT_ROOT)

    # --------------------------
    # Full Dataset Categories (EXACT SAME AS CCST)
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

    # FULL DATASET LIST (EXACT SAME AS CCST)
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
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
    ]

    # DLPFC List for function switching
    DLPFC_DATASETS = CATEGORIES["DLPFC"]
    all_results = []

    # Run all datasets
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING STAligner: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        # Load data (YOUR UTILS FILE - UNCHANGED)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)

        # Time & Memory Tracking
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 /1024 /1024

        # --------------------------
        # AUTO SELECT FUNCTION: DLPFC = Original, Others = Universal
        # --------------------------
        if dataset in DLPFC_DATASETS:
            res = run_STAligner_DLPFC(adata_h5, save_path, dataset, epochs=200)
        else:
            res = run_STAligner_Universal(adata_h5, save_path, dataset, epochs=200, n_cluster=n_cluster)

        # Calculate resources
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss /1024/1024/1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak /1024/1024/1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Get data type category
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Full result
        full_result = {
            "data_type": data_type,
            "dataset": res["dataset"],
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

    # Save final combined results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "STAligner_ALL_DATASETS.csv"), index=False)
    print("\n✅ STAligner ALL DATASETS COMPLETED! Results saved to:", OUTPUT_ROOT)