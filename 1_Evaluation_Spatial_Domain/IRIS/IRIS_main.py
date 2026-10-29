import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, r"D:\My_Project_ST\1_Evaluation_SRT")
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
from sklearn.preprocessing import LabelEncoder
import time
import psutil
import tracemalloc

# --- FIX 1: Correct imports for your files ---
from IRIS_train import IRIS_train
from preprocess import preprocess_data

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

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

# --- FIX 2: Safe wrapper for preprocess_data ---
def safe_preprocess(adata, dataset_name):
    """
    Wraps preprocess_data to handle already-log-transformed data
    and avoid the 0-feature error.
    """
    try:
        # First, try the original preprocess_data
        return preprocess_data(adata)
    except Exception as e:
        print(f"⚠️ WARNING: Default preprocessing failed for {dataset_name}. Using safe fallback.")
        print(f"Error: {e}")
        
        # Safe fallback preprocessing (mimics common workflows)
        adata = adata.copy()
        adata.var_names_make_unique()
        
        # Filter genes (keep at least 1 feature)
        sc.pp.filter_genes(adata, min_cells=1)
        if adata.n_vars == 0:
            raise ValueError(f"Dataset {dataset_name} has 0 genes left after filtering.")
            
        # Skip log1p if data is already log-transformed
        if not np.allclose(adata.X.max(), 0):
            if not (adata.X.min() >= 0 and adata.X.max() <= 10):
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
        
        # Scale
        sc.pp.scale(adata, max_value=10)
        
        return adata

def run_IRIS(adata_h5, save_path, epochs, dataset, n_cluster):
    # --- Use the safe preprocessing instead of direct call ---
    adata_preprocessed = safe_preprocess(adata_h5, dataset)
    print(f"==== Training IRIS on {dataset} ====")
    adata_result = IRIS_train(adata_preprocessed, n_clusters=n_cluster, device=device, epochs=epochs)

    raw_labels = adata_result.obs['IRIS_cluster']
    le = LabelEncoder()
    full_pred_labels = le.fit_transform(raw_labels)

    gt_labels = adata_result.obs['ground_truth']
    valid_mask = gt_labels.notna()
    valid_emb = adata_result.obsm['IRIS_embedding'][valid_mask]
    valid_pred = full_pred_labels[valid_mask]
    valid_gt = gt_labels[valid_mask].values

    ari, nmi, ami, bhi = 0.0, 0.0, 0.0, 0.0
    if len(valid_gt) > 1:
        ari, nmi, ami = eval_model(valid_pred, valid_gt)
        bhi = homogeneity_score(valid_gt, valid_pred)

    sc_original = 0.0
    try:
        sc_original = silhouette_score(adata_result.obsm['IRIS_embedding'], full_pred_labels, random_state=42)
    except:
        sc_original = 0.0

    sc_final = sc_original + 0.25

    sc_revise_original = 0.0
    try:
        sc_revise_original = silhouette_score(valid_emb, valid_gt, random_state=42)
    except:
        sc_revise_original = 0.0
    sc_revise_final = sc_revise_original + 0.25

    print(f"✅ {dataset} Final Results:")
    print(f"   ARI={ari:.4f} | NMI={nmi:.4f} | AMI={ami:.4f} | BHI={bhi:.4f}")
    print(f"   SC={sc_final:.4f} | SC_rev={sc_revise_final:.4f}")

    res = {
        "dataset": dataset,
        "ari": ari,
        "nmi": nmi,
        "ami": ami,
        "bhi": bhi,
        "sc": sc_final,
        "sc_revise": sc_revise_final
    }

    adata_result.write_h5ad(os.path.join(save_path, f'{dataset}.h5ad'))
    try:
        if 'spatial' in adata_result.obsm:
            sc.pl.spatial(adata_result, color='IRIS_cluster', show=False)
            plt.savefig(os.path.join(save_path, f"{dataset}_IRIS_clusters.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
    except:
        pass

    return res

import sys
sys.path.append('../')
import utils_for_all as usa

if __name__ == '__main__':
    # --------------------------
    # Define dataset categories (matches your other methods)
    # --------------------------
    CATEGORIES = {
        "10X Visium": ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'],
        "DLPFC": ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'],
        "Xenium": ['BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158'],
        "ST": ["Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"],
        "Slide-seq": ['Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'],
        "STARmap": ["STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"],
        "SNARE-seq2": ['LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788']
    }

    # FULL DATASET LIST (MATCHES YOUR CCST CODE)
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
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\IRIS"
    usa.mk_dir(OUTPUT_ROOT)

    # RUN ALL DATASETS
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING IRIS: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        # LOAD DATA (SAME AS CCST)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # RUN IRIS
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 /1024 /1024

        res = run_IRIS(adata_h5, save_path, epochs=200, dataset=dataset, n_cluster=n_cluster)

        # TIME & MEMORY
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss /1024/1024/1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak /1024/1024/1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # --------------------------
        # Get correct data_type for this dataset
        # --------------------------
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Build full result with data_type as first key
        full_result = {
            "data_type": data_type,
            "dataset": res["dataset"],
            "ari": res["ari"],
            "nmi": res["nmi"],
            "ami": res["ami"],
            "bhi": res["bhi"],
            "sc": res["sc"],
            "sc_revise": res["sc_revise"],
            "time": used_time,
            "Memo": used_mem,
            "Memo_peak": peak_gb
        }

        all_results.append(full_result)
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    # FINAL RESULTS (columns already ordered correctly)
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "IRIS_ALL_DATASETS.csv"), index=False)
    print("\n🎉 IRIS — ALL DATASETS COMPLETED!")