import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
sys.path.append(r"D:\My_Project_ST\1_Evaluation_SRT")
import torch
import argparse
import warnings
import numpy as np
import anndata
import scanpy as sc
import matplotlib
matplotlib.use('Agg')  # Headless plotting for batch run (critical for all datasets)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, homogeneity_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import time
import psutil
import tracemalloc
import scipy.interpolate as interp
import scipy.ndimage as ndimage

from stFormer import STFormer_train, preprocess_data

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# ----------------- Smooth Plot Function (Paper Style) -----------------
def smooth_spatial_clusters(adata, cluster_key, grid_size=250, sigma=2.0):
    x = adata.obsm['spatial'][:, 0]
    y = adata.obsm['spatial'][:, 1]
    labels = adata.obs[cluster_key].astype(int).values
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)
    ZI = interp.griddata((x, y), labels, (XI, YI), method='nearest')
    ZI_smooth = ndimage.gaussian_filter(ZI, sigma=sigma)
    return XI, YI, ZI_smooth

# ----------------- Helper Functions -----------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

def eval_model(pred, labels):
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    ami = adjusted_mutual_info_score(labels, pred)
    return ari, nmi, ami

# ----------------- UNIVERSAL STFormer Run Function (WORKS FOR ALL DATASETS) -----------------
def run_STFormer_universal(adata_h5, save_path, epochs, dataset, n_cluster):
    # --------------------------
    
    already_logged = adata_h5.X.mean() < 1.0

    # For the two problematic datasets, use a gentler preprocessing pipeline
    if dataset in ["Colorectal_TENX139", "Ovarian_TENX142"]:
        print(f"⚠️ Using gentle preprocessing for {dataset} (already log-transformed)")
        adata_preprocessed = adata_h5.copy()
        
        # Only minimal filtering (avoid filtering out all genes/cells)
        sc.pp.filter_genes(adata_preprocessed, min_cells=1)  # Keep genes present in at least 1 cell
        sc.pp.filter_cells(adata_preprocessed, min_genes=1) # Keep cells with at least 1 expressed gene
        
        # Skip log1p if already logged
        if not already_logged:
            sc.pp.log1p(adata_preprocessed)
    else:
        # Use original STFormer preprocessing for all other datasets
        adata_preprocessed = preprocess_data(adata_h5)

    # --------------------------
    # Step 2: Validate preprocessed data before training
    # --------------------------
    if adata_preprocessed.n_obs == 0 or adata_preprocessed.n_vars == 0:
        print(f"❌ Preprocessing failed for {dataset}: empty matrix after filtering!")
        print(f"   Shape after filtering: {adata_preprocessed.shape}")
        print(f"   Using original adata_h5 directly (no filtering)")
        adata_preprocessed = adata_h5.copy()

    print(f"✅ Final preprocessed shape for {dataset}: {adata_preprocessed.shape}")

    # --------------------------
    # Step 3: Train STFormer
    # --------------------------
    print(f"==== Training STFormer on {dataset} ====")
    adata_result = STFormer_train(adata_preprocessed, n_clusters=n_cluster, device=device, epochs=epochs)

    # --------------------------
    # Step 4: Align indices (keep your original fix)
    # --------------------------
    common_obs = adata_result.obs_names.intersection(adata_h5.obs_names)
    adata_result = adata_result[common_obs, :].copy()
    adata_h5_aligned = adata_h5[common_obs, :].copy()

    # Copy ground truth and spatial info safely
    adata_result.obs['ground_truth'] = adata_h5_aligned.obs['ground_truth'].values
    if 'spatial' in adata_h5_aligned.obsm:
        adata_result.obsm['spatial'] = adata_h5_aligned.obsm['spatial']
        adata_result.uns['spatial'] = adata_h5_aligned.uns.get('spatial', {})

    # --------------------------
    # Step 5: Evaluation (unchanged)
    # --------------------------
    valid_mask = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['STFormer_cluster'][valid_mask]
    gt = adata_result.obs['ground_truth'][valid_mask]

    if len(pred) == 0 or len(gt) == 0:
        print(f"⚠️ WARNING: No valid ground truth labels for {dataset}!")
        ari, nmi, ami, bhi, sc_score = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        bhi = homogeneity_score(gt, pred)
        if len(np.unique(pred)) > 1:
            sc_score = silhouette_score(adata_result.obsm['STFormer_embedding'][valid_mask], pred)
        else:
            sc_score = 0.0

    print(f"✅ {dataset} Scores: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")

    # --------------------------
    # Step 6: Save and plot (unchanged)
    # --------------------------
    adata_result.write_h5ad(os.path.join(save_path, f'{dataset}.h5ad'))

    try:
        if 'spatial' in adata_result.obsm:
            XI, YI, ZI_smooth = smooth_spatial_clusters(adata_result, cluster_key='STFormer_cluster', grid_size=250, sigma=2.0)
            fig, ax = plt.subplots(figsize=(7,7))
            im = ax.pcolormesh(XI, YI, ZI_smooth, cmap='coolwarm', shading='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Spatial Domain')
            ax.set_title(f'STFormer - {dataset}', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.savefig(os.path.join(save_path, f"STFormer_{dataset}_smooth_domains.pdf"), bbox_inches='tight', dpi=300)
            plt.close()
    except Exception as e:
        print(f"⚠️ Plot skipped for {dataset}: {str(e)}")

    res = {
        "dataset": dataset,
        "ari": ari,
        "nmi": nmi,
        "ami": ami,
        "bhi": bhi,
        "sc": sc_score
    }
    return res

# ----------------- MAIN RUNNER FOR ALL DATASETS -----------------
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    # --------------------------
    # PATHS (MATCH YOUR CCST CODE)
    # --------------------------
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\STFormer"  # Universal output folder
    usa.mk_dir(OUTPUT_ROOT)

    # --------------------------
    # Dataset Categories (EXACT SAME AS CCST)
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

    # Seed setup
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    all_results = []

    # --------------------------
    # Run STFormer on ALL datasets
    # --------------------------
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING STFormer: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        # ✅ Load ALL datasets via your utils_for_all.py (SAME AS CCST)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # --------------------------
        # Time & Memory Tracking
        # --------------------------
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        # Run UNIVERSAL STFormer
        res = run_STFormer_universal(adata_h5, save_path, epochs=200, dataset=dataset, n_cluster=n_cluster)

        # Calculate resource usage
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / 1024 / 1024 / 1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # --------------------------
        # Assign dataset category (data_type)
        # --------------------------
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Build full result (SAME COLUMN ORDER AS CCST)
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
        # Save per-dataset result
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_STFormer_result.csv"), index=False)

    # --------------------------
    # Save FINAL combined results
    # --------------------------
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "STFormer_ALL_DATASETS.csv"), index=False)
    print("\n✅ ALL DATASETS COMPLETED! Results saved to:", OUTPUT_ROOT)