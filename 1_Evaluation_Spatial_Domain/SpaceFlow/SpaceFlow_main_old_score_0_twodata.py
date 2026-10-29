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
from SpaceFlow import SpaceFlow
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    homogeneity_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================
# HELPERS (MATCH CCST STRUCTURE)
# ==========================
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def eval_model(pred, labels=None):
    if labels is not None:
        # Keep ONLY rows where labels are NOT NaN (ignore NaN in pred)
        valid_mask = ~pd.isna(labels)
        df = pd.DataFrame({
            "True": labels[valid_mask],
            "Pred": pred[valid_mask]
        })
        
        # Check if enough valid data
        if len(df) < 2 or len(df['True'].unique()) < 2 or len(df['Pred'].unique()) < 2:
            print("⚠️ Not enough valid labels/predictions for evaluation, returning 0")
            return 0, 0, 0, 0
        
        ari = adjusted_rand_score(df["True"], df["Pred"])
        nmi = normalized_mutual_info_score(df["True"], df["Pred"])
        ami = adjusted_mutual_info_score(df["True"], df["Pred"])
        bhi = homogeneity_score(df["True"], df["Pred"])
    else:
        ari, nmi, ami, bhi = 0, 0, 0, 0
    return ari, nmi, ami, bhi

# Leiden clustering for consistent evaluation (MATCH CCST)
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    # Broader range (0.1 to 3.0) to find resolution for high-cluster datasets
    resolutions = sorted(list(np.arange(0.1, 3.0, increment)), reverse=True)
    for res in resolutions:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        current_clusters = len(adata.obs['leiden'].unique())
        if current_clusters == fixed_clus_count:
            return res
        # Early exit if we go too low (prevents unnecessary loops)
        if current_clusters < fixed_clus_count and res < 0.2:
            break
    print(f"⚠️ WARNING: Could not find exact resolution for {fixed_clus_count} clusters. Using default 1.0")
    return 1.0

# --------------------------
#  FINAL FIXED Spatial Coordinate Helper Function (from SEDR)
# --------------------------
def get_spatial_coords(adata, dataset_name):
    """
    Final version that handles all cases, including STARmap with non-standard indices.
    """
    # --- Problem 1: Handle STARmap first ---
    if dataset_name.startswith("STARmap"):
        print(f"Handling STARmap-specific coordinate extraction for {dataset_name}")
        
        # Check if there is a valid 'spatial' obsm entry (that contains actual coordinates)
        if 'spatial' in adata.obsm:
            # Try to convert it to float32 to see if it's numeric
            try:
                coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
                if coords.ndim == 2 and coords.shape[0] == adata.shape[0] and coords.shape[1] == 2:
                    print(f"✅ Using existing numeric coordinates from adata.obsm['spatial']")
                    return coords
            except (ValueError, TypeError):
                print(f"⚠️ adata.obsm['spatial'] contains non-numeric data, skipping it")

        # Try to find coordinates in other standard locations
        for key in ['spatial_coords', 'pos', 'coord']:
            if key in adata.obsm:
                try:
                    coords = np.asarray(adata.obsm[key], dtype=np.float32)
                    if coords.ndim == 2 and coords.shape[0] == adata.shape[0] and coords.shape[1] == 2:
                        print(f"✅ Found coordinates in adata.obsm['{key}']")
                        return coords
                except:
                    continue

        # Try to find coordinates in .obs (common STARmap formats)
        for col_pair in [['x', 'y'], ['X', 'Y'], ['x_coord', 'y_coord'], ['coord_x', 'coord_y']]:
            if all(col in adata.obs.columns for col in col_pair):
                coords = adata.obs[col_pair].values
                coords = np.asarray(coords, dtype=np.float32)
                print(f"✅ Found coordinates in .obs columns {col_pair}")
                return coords

        # Fallback: If no coordinates found, use cell indices to create dummy coordinates
        print(f"⚠️ WARNING: No valid coordinates found for {dataset_name}. Using cell index as dummy coordinates (only for model training)")
        # Create a grid of coordinates based on the cell index
        n_cells = adata.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_cells)))
        x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        coords = np.stack([x.flatten(), y.flatten()], axis=1)[:n_cells].astype(np.float32)
        print(f"✅ Created dummy coordinates with shape {coords.shape}")
        return coords

    # --- Problem 2: Handle datasets with NO standard coordinates (SNARE-seq2/Slide-seq) ---
    print(f"Handling non-STARmap dataset: {dataset_name}")
    print("Available obsm keys:", list(adata.obsm.keys()))

    # Fallback 1: If coordinates are stored in .uns['spatial']
    if 'spatial' in adata.uns:
        if isinstance(adata.uns['spatial'], dict):
            # Check common keys in the spatial dict
            for key in ['coords', 'pos', 'tissue_positions']:
                if key in adata.uns['spatial']:
                    coords = adata.uns['spatial'][key]
                    if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[0] == adata.shape[0]:
                        coords = np.asarray(coords, dtype=np.float32)
                        print(f"✅ Found coordinates in adata.uns['spatial']['{key}']")
                        return coords
            # Last resort: check all values in the spatial dict
            for key, val in adata.uns['spatial'].items():
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == adata.shape[0]:
                    coords = np.asarray(val, dtype=np.float32)
                    print(f"✅ Found coordinates in adata.uns['spatial']['{key}']")
                    return coords
        else:
            coords = np.asarray(adata.uns['spatial'], dtype=np.float32)
            if coords.ndim == 2 and coords.shape[0] == adata.shape[0]:
                print(f"✅ Found coordinates in adata.uns['spatial']")
                return coords

    # Fallback 2: If coordinates are stored in .obs with non-standard column names
    for col_pair in [
        ['x', 'y'], ['X', 'Y'], ['x_coord', 'y_coord'], ['coord_x', 'coord_y'],
        ['spatial_x', 'spatial_y'], ['cell_x', 'cell_y']
    ]:
        if all(col in adata.obs.columns for col in col_pair):
            coords = adata.obs[col_pair].values
            coords = np.asarray(coords, dtype=np.float32)
            print(f"✅ Found coordinates in .obs columns {col_pair}")
            return coords

    # Fallback 3: Emergency fallback - check common obsm keys
    for key in ['spatial', 'spatial_coords', 'pos', 'coord']:
        if key in adata.obsm:
            try:
                coords = np.asarray(adata.obsm[key], dtype=np.float32)
                if coords.ndim == 2 and coords.shape[0] == adata.shape[0] and coords.shape[1] == 2:
                    print(f"✅ Found coordinates in adata.obsm['{key}']")
                    return coords
            except:
                continue

    # Fallback 4: Emergency fallback - create dummy coordinates (only for training)
    print(f"⚠️ WARNING: No coordinates found for {dataset_name}. Creating dummy coordinates (only for model training)")
    n_cells = adata.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_cells)))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    coords = np.stack([x.flatten(), y.flatten()], axis=1)[:n_cells].astype(np.float32)
    print(f"✅ Created dummy coordinates with shape {coords.shape}")
    return coords

sys.path.append('../')
import utils_for_all as usa

# ==========================
# SPACEFLOW PIPELINE (ALL DATASETS)
# ==========================
def run_SpaceFlow(adata_h5, save_path, dataset, n_cluster=10, epochs=200):
    """
    Unified SpaceFlow pipeline for ALL datasets (matches CCST structure)
    """
    # Convert sparse matrix to dense safely
    X_dense = adata_h5.X.toarray() if hasattr(adata_h5.X, 'toarray') else adata_h5.X

    # ---------------------------
    # ✅ FIX: Use SEDR's robust coordinate extraction
    # ---------------------------
    spatial = get_spatial_coords(adata_h5, dataset)

    # Initialize and train SpaceFlow
    sf = SpaceFlow.SpaceFlow(
        count_matrix=X_dense,
        spatial_locs=spatial
    )
    sf.preprocessing_data()
    sf.train(
        embedding_save_filepath=os.path.join(save_path, "SpaceFlow_embedding.tsv"),
        epochs=epochs
    )

    # Add embedding to anndata
    adata_h5.obsm['SpaceFlow_emb'] = sf.embedding

    # ---------------------------
    # Clustering (match CCST: Leiden with fixed cluster count)
    # ---------------------------
    sc.pp.neighbors(adata_h5, use_rep='SpaceFlow_emb', n_neighbors=20, random_state=0)
    sc.tl.umap(adata_h5, random_state=0)
    eval_resolution = res_search_fixed_clus(adata_h5, n_cluster)
    sc.tl.leiden(adata_h5, key_added="SpaceFlow_leiden", resolution=eval_resolution, random_state=0)

    # Save final anndata
    adata_h5.write(os.path.join(save_path, f"SpaceFlow_{dataset}.h5ad"))

    # ---------------------------
    # Plot spatial domains (if spatial data exists)
    # ---------------------------
    try:
        if 'spatial' in adata_h5.obsm:
            sc.pl.spatial(
                adata_h5,
                img_key="hires",
                color=['SpaceFlow_leiden'],
                title=f"SpaceFlow - {dataset}",
                show=False
            )
            plt.savefig(
                os.path.join(save_path, "SpaceFlow_spatial_plot.pdf"),
                bbox_inches='tight',
                dpi=150
            )
            plt.close()
    except Exception as e:
        print(f"Plot skipped: {str(e)}")

    # ---------------------------
    # ✅ FIX: Slide-seq score calculation
    # ---------------------------
    # --- Check if ground truth exists and is valid ---
    if 'ground_truth' not in adata_h5.obs.columns:
        print(f"⚠️ {dataset} has no 'ground_truth' column. Setting all metrics to 0.")
        ari, nmi, ami, bhi = 0, 0, 0, 0
        sc_score = 0.0
    else:
        valid_mask = adata_h5.obs['ground_truth'].notna()
        n_valid = valid_mask.sum()
        print(f"🔍 {dataset}: {n_valid}/{len(adata_h5)} cells have valid ground truth labels")

        if n_valid < 2:
            print(f"⚠️ Not enough valid ground truth labels for evaluation. Setting all metrics to 0.")
            ari, nmi, ami, bhi = 0, 0, 0, 0
            sc_score = 0.0
        else:
            pred = adata_h5.obs["SpaceFlow_leiden"][valid_mask]
            gt = adata_h5.obs["ground_truth"][valid_mask]

            # --- Evaluate ARI/NMI/AMI/BHI ---
            ari, nmi, ami, bhi = eval_model(pred, gt)

            # --- Evaluate Silhouette Score ---
            valid_emb = adata_h5.obsm['SpaceFlow_emb'][valid_mask]
            if len(np.unique(pred)) > 1 and len(np.unique(gt)) > 1:
                sc_score = silhouette_score(valid_emb, pred)
            else:
                print(f"⚠️ Not enough unique clusters for silhouette score. Setting SC to 0.")
                sc_score = 0.0

    print(f"\n✅ {dataset} Scores: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")

    return {
        "dataset": dataset,
        "ari": round(ari,4),
        "nmi": round(nmi,4),
        "ami": round(ami,4),
        "bhi": round(bhi,4),
        "sc": round(sc_score,4)
    }

# ==========================
# MAIN RUNNER (ALL 7 CATEGORIES)
# ==========================
if __name__ == '__main__':
    # PATHS (EXACTLY LIKE CCST)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\SpaceFlow"
    mk_dir(OUTPUT_ROOT)

    # --------------------------
    # EXACT SAME DATASET LIST AS CCST
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
        
        # Slide-seq
        'Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855',
        # DLPFC
        '151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676',
        # 10X Visium
        'BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72',
        # ST
        "Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata",
        "Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130",
        
        # SNARE-seq2
        'LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788',
        # STARmap
        "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap",
        # Xenium
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
    ]

    # Reproducibility
    np.random.seed(42)

    # Store all results
    all_results = []

    # --------------------------
    # RUN ALL DATASETS
    # --------------------------
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING SPACEFLOW: {dataset}")
        print(f"==================================================")

        save_path = os.path.join(OUTPUT_ROOT, dataset)
        mk_dir(save_path)

        # ✅ LOAD DATA VIA YOUR UTILS (same as CCST)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)

        # Time & Memory tracking
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        # RUN SPACEFLOW
        res = run_SpaceFlow(
            adata_h5,
            save_path=save_path,
            dataset=dataset,
            n_cluster=n_cluster,
            epochs=200
        )

        # Calculate resources
        used_time = round(time.time() - start, 2)
        used_mem = round((psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) - start_mem, 4)
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = round(peak / 1024 / 1024 / 1024, 4)
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Get dataset category
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Build full result
        full_result = {
            "data_type": data_type,
            "dataset": res["dataset"],
            "ari": res["ari"],
            "nmi": res["nmi"],
            "ami": res["ami"],
            "bhi": res["bhi"],
            "sc": res["sc"],
            "time(s)": used_time,
            "mem_used(GB)": used_mem,
            "mem_peak(GB)": peak_gb
        }

        all_results.append(full_result)
        pd.DataFrame([full_result]).to_csv(
            os.path.join(save_path, f"{dataset}_metrics.csv"),
            index=False
        )

    # Save combined final results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "SpaceFlow_ALL_DATASETS.csv"), index=False)

    print("\n🎉 🎉 🎉 SPACEFLOW — ALL DATASETS COMPLETED!")