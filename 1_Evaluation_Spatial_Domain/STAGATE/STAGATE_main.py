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
import multiprocessing as mp
import matplotlib.pyplot as plt

# --------------------------
# Basic Setup (Same as SpaRSE)
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('===== Using device:', device)

# --------------------------
# Import STAGATE
# --------------------------
import STAGATE_pyG as STAGATE

# --------------------------
# Utility Functions
# --------------------------
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def eval_model(pred, labels=None):
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
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

# --------------------------
# ✅ ORIGINAL STAGATE: For DLPFC Data (151xxx samples)
# --------------------------
def run_STAGATE_DLPFC(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=150):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()
    
    # STAGATE Standard Preprocessing
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Compute Spatial Network
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    # Train STAGATE
    adata = STAGATE.train_STAGATE(adata, device=device, random_seed=random_seed)
    
    # Clustering (Leiden instead of mclust_R)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    n_clus = len(np.unique(adata.obs["ground_truth"].dropna()))

    def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            if len(adata.obs['leiden'].unique()) == fixed_clus_count:
                break
        return res

    res = res_search_fixed_clus(adata, n_clus)
    sc.tl.leiden(adata, key_added="mclust", resolution=res, random_state=0)
    
    # Save results
    adata.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))
    
    # Evaluate
    valid = adata.obs['ground_truth'].notna()
    pred = adata.obs['mclust'][valid]
    gt = adata.obs['ground_truth'][valid]
    
    if len(pred) == 0:
        ari, nmi, ami, sc_score, bhi = 0,0,0,0,0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        from sklearn.metrics import silhouette_score, homogeneity_score
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata.obsm["STAGATE"][valid], pred) if len(np.unique(pred))>1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# --------------------------
# ✅ NEW STAGATE: For ALL NON-DLPFC Datasets (Unified)
# --------------------------
def run_STAGATE_All_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50):
    os.makedirs(save_path, exist_ok=True)
    adata = adata_h5.copy()

    # --------------------------
    # 1. DYNAMIC DOWNSAMPLING (keep 1000–1200 cells)
    # --------------------------
    n_total = adata.n_obs
    target_min = 1000
    target_max = 1200

    if n_total > target_max:
        # Calculate fraction to hit ~1100 cells (midpoint of range)
        target = (target_min + target_max) // 2
        downsample_frac = target / n_total
        print(f"⚠️ OOM Fix: Downsampling {dataset} ({n_total} cells) to {target} cells ({downsample_frac:.1%})")
        adata = adata[adata.obs.sample(frac=downsample_frac, random_state=42).index].copy()
    else:
        # Keep all cells if already below 1200
        print(f"✅ No downsampling needed for {dataset} ({n_total} cells)")

    print(f"After downsampling: {adata.shape}")

    # --------------------------
    # 2. FIX SPATIAL COORDINATES FOR STAGATE
    # --------------------------
    if 'spatial' not in adata.obsm or adata.obsm['spatial'].shape[1] != 2:
        print("⚠️ Fixing spatial coordinates (must be 2D for STAGATE)")
        # If spatial exists but has >2 columns, keep only the first two
        if 'spatial' in adata.obsm:
            adata.obsm['spatial'] = adata.obsm['spatial'][:, :2]
        # If spatial is missing, generate dummy coordinates
        else:
            n = adata.n_obs
            side = int(np.ceil(np.sqrt(n)))
            xx, yy = np.meshgrid(np.arange(side), np.arange(side))
            dummy_coords = np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])
            adata.obsm['spatial'] = dummy_coords[:n]

    # --------------------------
    # 3. MEMORY-EFFICIENT PREPROCESSING
    # --------------------------
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1500)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --------------------------
    # 4. REDUCE SPATIAL NETWORK COMPLEXITY
    # --------------------------
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
    print(f"Created spatial network with {adata.uns['Spatial_Net'].shape[0]} edges")

    # --------------------------
    # 5. STAGATE TRAINING WITH MEMORY OPTIMIZATIONS
    # --------------------------
    print("Training STAGATE (OOM-safe mode)...")
    device_train = torch.device('cpu')
    adata = STAGATE.train_STAGATE(adata, device=device_train, random_seed=random_seed)

    # --------------------------
    # 6. CLUSTERING (Leiden, no R)
    # --------------------------
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    n_clus = len(np.unique(adata.obs["ground_truth"].dropna()))

    def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            if len(adata.obs['leiden'].unique()) == fixed_clus_count:
                break
        return res

    res = res_search_fixed_clus(adata, n_clus)
    sc.tl.leiden(adata, key_added="mclust", resolution=res, random_state=0)

    # --------------------------
    # 7. SAVE AND EVALUATE
    # --------------------------
    adata.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    valid = adata.obs['ground_truth'].notna()
    pred = adata.obs['mclust'][valid]
    gt = adata.obs['ground_truth'][valid]

    if len(pred) == 0:
        ari, nmi, ami, sc_score, bhi = 0,0,0,0,0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        from sklearn.metrics import silhouette_score, homogeneity_score
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata.obsm["STAGATE"][valid], pred) if len(np.unique(pred))>1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}
# --------------------------
# Load Your Unified Data Loader
# --------------------------
sys.path.append('../')
import utils_for_all as usa

# --------------------------
# ✅ MAIN RUN (ALL DATASETS - SAME STRUCTURE AS YOUR SpaRSE CODE)
# --------------------------
if __name__ == '__main__':
    plt.switch_backend('Agg')
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    output_root = r"D:\My_Project_ST\Output_For_All\STAGATE"
    mk_dir(output_root)

    # FULL DATASET CATEGORIES (EXACT SAME AS YOUR SpaRSE CODE)
    CATEGORIES = {
        "DLPFC": ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'],
        "10X Visium": ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'],
        "ST": ["Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"],
        "Slide-seq": ['Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'],
        "SNARE-seq2": ['LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'],
        "STARmap": ["STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"],
        "Xenium": ['BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158']
    }

    # ALL DATASETS LIST (EXACT SAME AS YOUR SpaRSE CODE)
    ALL_DATASETS = [
        # Xenium
        'Lung_NCBI882','BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158',
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
        "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"
        
    ]

    all_results = []

    # Run every dataset
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING STAGATE: {dataset}")
        print(f"==================================================")
        
        save_path = mk_dir(os.path.join(output_root, dataset))

        # Load data using YOUR utils_for_all.py
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time & Memory tracking
        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024

        # Auto choose DLPFC / All function
        if dataset.startswith('15'):
            res = run_STAGATE_DLPFC(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=150)
        else:
            res = run_STAGATE_All_OOM(adata_h5, save_path, dataset, random_seed=0, rad_cutoff=50)

        # Resource usage
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

    # Save final full results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(output_root, "STAGATE_ALL_DATASETS_RESULTS.csv"), index=False)
    print("\n🎉 STAGATE FINISHED ALL DATASETS! Results saved to:", output_root)