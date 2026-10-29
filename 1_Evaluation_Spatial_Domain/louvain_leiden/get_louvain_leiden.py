# -*- coding:utf-8 -*-
import sys
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from scipy.sparse import issparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    homogeneity_score
)
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import warnings
import time
import psutil
import tracemalloc

sys.path.append('../')
# Ensure we can import utils_for_all from the parent folder
sys.path.append(r"D:\My_Project_ST\1_Evaluation_SRT")
import utils_for_all as usa

warnings.filterwarnings('ignore')
np.random.seed(0)

# ==========================
# Utility Functions
# ==========================
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
        bhi = homogeneity_score(label_df["True"], label_df["Pred"])
    else:
        ari, nmi, ami, bhi = 0, 0, 0, 0
    return ari, nmi, ami, bhi

def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        if cluster_type == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(adata.obs['leiden'].unique())
        elif cluster_type == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(adata.obs['louvain'].unique())
        if count_unique == fixed_clus_count:
            break
    return res

# ==========================
# Unified Clustering Function for ALL datasets (DLPFC + Others)
# ==========================
def run_cluster_all(adata_h5, save_path, dataset, n_cluster, cluster_method):
    """
    Unified Louvain/Leiden pipeline for ALL datasets
    Uses same preprocessing as original code
    """
    adata = adata_h5.copy()
    ground_truth = adata.obs["ground_truth"].values
    le = LabelEncoder()
    ground_truth_le = le.fit_transform(ground_truth)

    # Preprocess
    adata.X = adata.X.toarray() if issparse(adata.X) else adata.X

    # Standard clustering pipeline
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
    eval_res = res_search_fixed_clus(cluster_method, adata, n_cluster)

    if cluster_method == 'leiden':
        sc.tl.leiden(adata, key_added="cluster", resolution=eval_res)
    else:
        sc.tl.louvain(adata, key_added="cluster", resolution=eval_res)

    cluster_labels = adata.obs["cluster"].astype(int).values

    # Calculate metrics
    ari, nmi, ami, bhi = eval_model(cluster_labels, ground_truth_le)
    SC = silhouette_score(adata.obsm['X_pca'], cluster_labels)
    SC_revise = silhouette_score(adata.obsm['X_pca'], ground_truth_le)

    # Save adata
    adata.write(f"{save_path}/{cluster_method}_{dataset}.h5ad")

    # Return results
    res = {
        "dataset": dataset,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "ami": round(ami, 4),
        "bhi": round(bhi, 4),
        "sc": round(SC, 4),
        "sc_revise": round(SC_revise, 4)
    }
    return res, adata

# ==========================
# Dataset Categories (MATCH CCST CODE)
# ==========================
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

# ==========================
# MAIN: Run Louvain + Leiden on ALL DATASETS (SEPARATE FOLDERS)
# ==========================
if __name__ == "__main__":
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    
    # Run both clustering methods (SEPARATE ROOT OUTPUT FOLDERS)
    for cluster_method in ['leiden', 'louvain']:
        print(f"\n{'='*60}")
        print(f"          RUNNING {cluster_method.upper()} ON ALL DATASETS")
        print(f"{'='*60}")

        all_results = []
        # ✅ SEPARATE OUTPUT FOLDER FOR EACH METHOD
        if cluster_method == "louvain":
            OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\Louvain"
        else:
            OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\Leiden"
        
        mk_dir(OUTPUT_ROOT)

        for dataset in ALL_DATASETS:
            print(f"\n--- Processing {dataset} ({cluster_method}) ---")
            save_path = os.path.join(OUTPUT_ROOT, dataset)
            mk_dir(save_path)

            # Time & Memory tracking
            start = time.time()
            tracemalloc.start()
            start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

            # --------------------------
            # Load dataset using usa.load_dataset() (works for ALL datasets, including DLPFC)
            # --------------------------
            adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
            res, adata = run_cluster_all(adata_h5, save_path, dataset, n_cluster, cluster_method)

            # --------------------------
            # Final stats
            # --------------------------
            used_time = time.time() - start
            used_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) - start_mem
            current, peak = tracemalloc.get_traced_memory()
            peak_gb = peak / 1024 / 1024 / 1024
            tracemalloc.stop()
            tracemalloc.clear_traces()

            # Get data category
            data_type = "Unknown"
            for cat, dset in CATEGORIES.items():
                if dataset in dset:
                    data_type = cat
                    break

            # Full result
            full_res = {
                "method": cluster_method,
                "data_type": data_type,
                "dataset": dataset,
                "ari": res["ari"],
                "nmi": res["nmi"],
                "ami": res["ami"],
                "bhi": res["bhi"],
                "sc": res["sc"],
                "sc_revise": res["sc_revise"],
                "time": round(used_time,2),
                "Memo": round(used_mem,4),
                "Memo_peak": round(peak_gb,4)
            }

            all_results.append(full_res)
            pd.DataFrame([full_res]).to_csv(os.path.join(save_path, f"{dataset}_{cluster_method}.csv"), index=False)
            print(f"✅ Done: ARI={res['ari']:.4f} NMI={res['nmi']:.4f}")

        # Save combined results for this method
        combined_df = pd.DataFrame(all_results)
        combined_csv = os.path.join(OUTPUT_ROOT, f"{cluster_method}_ALL_RESULTS.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n🎉 {cluster_method.upper()} FINISHED! Results saved to: {OUTPUT_ROOT}")

    print("\n🎉 ALL METHODS (Louvain + Leiden) COMPLETED ON ALL DATASETS!")