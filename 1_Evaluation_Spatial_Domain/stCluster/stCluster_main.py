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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, homogeneity_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import time
import psutil
import tracemalloc

from stCluster import stCluster_train, preprocess_data

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# --------------------------
# Original Helper Functions (UNCHANGED - for DLPFC compatibility)
# --------------------------
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

# --------------------------
# ORIGINAL run_stCluster (UNCHANGED - works for ALL datasets)
# --------------------------
def run_stCluster(adata_h5, save_path, epochs, dataset, n_cluster):
    
    adata_preprocessed = adata_h5.copy()
    
    # We'll do it manually with a check:
    sc.pp.filter_genes(adata_preprocessed, min_cells=3)
    sc.pp.normalize_total(adata_preprocessed, target_sum=1e4)
    
    # --------------------------
    # ✅ KEY FIX: Skip log1p if already transformed
    # --------------------------
    # Check if data is already log-transformed (values > 0 but < ~20, no zeros)
    if (adata_preprocessed.X.min() < 0 or adata_preprocessed.X.max() > 100):
        print(f"[{dataset}] Data looks like raw counts → applying log1p")
        sc.pp.log1p(adata_preprocessed)
    else:
        print(f"[{dataset}] Data already log-transformed → skipping log1p")

    sc.pp.highly_variable_genes(adata_preprocessed, flavor='seurat_v3', n_top_genes=3000, subset=True)

    print(f"==== Training stCluster on {dataset} ====")
    adata_result = stCluster_train(adata_preprocessed, n_clusters=n_cluster, device=device, epochs=epochs)

    # Rest of your original code remains unchanged
    valid_mask = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['stCluster_cluster'][valid_mask]
    gt = adata_result.obs['ground_truth'][valid_mask]

    if len(pred) == 0 or len(gt) == 0:
        print(f"⚠️ WARNING: No valid ground truth labels for {dataset}!")
        ari, nmi, ami, bhi, sc_score = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        bhi = homogeneity_score(gt, pred)
        if len(np.unique(pred)) > 1:
            sc_score = silhouette_score(adata_result.obsm['stCluster_embedding'][valid_mask], pred)
        else:
            sc_score = 0.0

    print(f"✅ {dataset} Scores: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")

    res = {
        "dataset": dataset,
        "ari": ari,
        "nmi": nmi,
        "ami": ami,
        "bhi": bhi,
        "sc": sc_score
    }

    adata_result.write_h5ad(os.path.join(save_path, f'{dataset}.h5ad'))
    
    try:
        if 'spatial' in adata_result.obsm:
            sc.pl.spatial(adata_result, color='stCluster_cluster', show=False)
            plt.savefig(os.path.join(save_path, f"{dataset}_stCluster_clusters.pdf"), 
                        bbox_inches='tight', dpi=150)
            plt.close()
    except Exception as e:
        print(f"Plot skipped for {dataset}: {str(e)}")

    return res

# ====================== MAIN RUNNER FOR ALL DATASETS ======================
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    # PATHS (MATCH CCST STRUCTURE)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\stCluster"
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
        # SNARE-seq2
        'LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788',
        
        # DLPFC
        '151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676',
        # 10X Visium
        'BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72',
        # ST
        "Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata",
        "Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130",
        # Slide-seq
        'Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855',
        
        # STARmap
        "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap",
        # Xenium
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'       
    ]

    # Fixed seed
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    all_results = []

    # --------------------------
    # Run stCluster on ALL datasets
    # --------------------------
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING stCluster: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        # Data loading (SAME AS CCST - uses your utils_for_all.py)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time & Memory Tracking
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        # Run ORIGINAL stCluster (no changes to core function)
        res = run_stCluster(adata_h5, save_path, epochs=200, dataset=dataset, n_cluster=n_cluster)

        # Calculate resources
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / 1024**3
        tracemalloc.stop()
        tracemalloc.clear_traces()

        print(f'Time: {used_time:.4f}s | Memory: {used_mem:.4f}GB | Peak: {peak_gb:.4f}GB')

        # Auto-detect dataset category
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Build result (SAME FORMAT AS CCST)
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
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_stCluster_result.csv"), index=False)

    # Save final combined results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "stCluster_ALL_DATASETS.csv"), index=False)
    print("\n✅ stCluster ANALYSIS COMPLETED FOR ALL DATASETS!")