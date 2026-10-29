# -*- coding:utf-8 -*-
import sys
import io
import numpy as np
import pandas as pd
import os
import time
import psutil
import tracemalloc
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# FORCE FIX PANDAS CUT ERROR
# ---------------------------
try:
    orig_cut = pd.cut
    def safe_cut(*args, **kwargs):
        kwargs['duplicates'] = 'drop'
        return orig_cut(*args, **kwargs)
    pd.cut = safe_cut
except:
    pass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import scanpy as sc
from SpaceFlow.SpaceFlow import SpaceFlow
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    homogeneity_score
)
from sklearn.cluster import KMeans

# ==========================
# HELPERS (SAME AS ORIGINAL + SPATIAL MGCN)
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

# --------------------------
# AUTO FIX SPATIAL COORDINATES (FOR NON-DLPFC DATA)
# --------------------------
def ensure_valid_spatial_coords(adata):
    """
    Automatically fix/validate spatial coordinates for ALL datasets
    Matches the logic from your Spatial_MGCN code
    """
    n = adata.n_obs
    spatial_coords = None

    # Get real spatial coords from common locations
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x','y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x','y']].values
    elif {'spatial_x','spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x','spatial_y']].values

    # Check if coords are numeric
    if spatial_coords is not None:
        if not np.issubdtype(spatial_coords.dtype, np.number):
            spatial_coords = None
        else:
            try:
                spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
            except ValueError:
                spatial_coords = None

    # Generate dummy 2D grid coords if missing/invalid
    if spatial_coords is None or spatial_coords.shape[0] != n:
        print("⚠️ Generating dummy 2D spatial coordinates for non-DLPFC dataset")
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        adata.obsm['spatial'] = spatial_coords.copy()

    return adata

# --------------------------
# SPACEFLOW PIPELINE FOR ALL DATASETS (FINAL FIXED VERSION)
# --------------------------
def run_spaceflow(adata_h5, save_path, dataset, n_cluster):
    """Unified SpaceFlow pipeline: works for DLPFC + ALL other datasets"""
    # Time & Memory tracking
    start = time.time()
    tracemalloc.start()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

    # ---------------------------
    # DATA PREPARATION
    # ---------------------------
    adata = adata_h5.copy()

    # Auto-fix spatial coordinates for non-DLPFC data
    if not dataset.startswith('15'):
        adata = ensure_valid_spatial_coords(adata)

    # ---------------------------
    # ✅ FINAL FIX: Use AnnData input (the library's intended way)
    # ---------------------------
    sf = SpaceFlow(adata=adata)

    # Run pipeline
    sf.preprocessing_data()
    sf.train(
        embedding_save_filepath=os.path.join(save_path, "embedding.tsv"),
        epochs=500
    )

    # KMeans clustering (fixed, no errors)
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    domains = kmeans.fit_predict(sf.embedding)

    # Save results to anndata
    adata.obs["SpaceFlow"] = domains
    adata.write(f"{save_path}SpaceFlow_{dataset}.h5ad")

    # ---------------------------
    # EVALUATION METRICS
    # ---------------------------
    ari, nmi, ami, bhi = eval_model(adata.obs["SpaceFlow"], adata.obs.get("ground_truth", None))
    SC = silhouette_score(sf.embedding, domains) if len(np.unique(domains)) > 1 else 0.0

    # ---------------------------
    # TIME & MEMORY STATISTICS
    # ---------------------------
    end = time.time()
    used_time = end - start
    end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    used_memo = end_MB - start_MB
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem = peak / 1024 /1024 /1024

    # Return full result dict
    res = {
        "dataset": dataset,
        "ari": round(ari,4),
        "nmi": round(nmi,4),
        "ami": round(ami,4),
        "bhi": round(bhi,4),
        "sc": round(SC,4),
        "time(s)": round(used_time,2),
        "mem_used(GB)": round(used_memo,4),
        "mem_peak(GB)": round(peak_mem,4)
    }

    return res, adata

sys.path.append('../')
import utils_for_all as usa

# ==========================
# MAIN — ALL DATASETS (SAME AS SPATIAL MGCN)
# ==========================
if __name__ == '__main__':
    # BASE PATHS (MATCH YOUR SPATIAL MGCN PATHS)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\SpaceFlow"
    mk_dir(OUTPUT_ROOT)

    # FULL DATASET LIST & CATEGORIES (EXACT SAME AS YOUR SPATIAL MGCN CODE)
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

    # Run for ALL datasets
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"           RUNNING SPACEFLOW: {dataset}")
        print(f"==================================================")

        # Create save path
        save_path = os.path.join(OUTPUT_ROOT, dataset, '')
        mk_dir(save_path)

        # Load data via YOUR unified utils_for_all.py
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)

        # Run SpaceFlow pipeline
        res, adata_result = run_spaceflow(adata_h5, save_path, dataset, n_cluster)

        # Add data type category
        data_type = "Unknown"
        for cat, dlist in CATEGORIES.items():
            if dataset in dlist:
                data_type = cat
                break

        # Full result with category
        full_result = {"data_type": data_type, **res}
        all_results.append(full_result)

        # Save per-dataset metrics
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_metrics.csv"), index=False)

        # Print results
        print(f"✅ {dataset} Results:")
        print(f"ARI={res['ari']:.4f} NMI={res['nmi']:.4f} AMI={res['ami']:.4f} BHI={res['bhi']:.4f} SC={res['sc']:.4f}")

        # Clear memory traces
        tracemalloc.clear_traces()

    # Save combined final results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "SpaceFlow_ALL_RESULTS.csv"), index=False)
    print("\n🎉 ALL DATASETS FINISHED — SPACEFLOW ANALYSIS COMPLETE!")