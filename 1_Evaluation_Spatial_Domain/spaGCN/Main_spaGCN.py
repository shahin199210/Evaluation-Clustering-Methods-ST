import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import psutil,tracemalloc
import time
import warnings
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,silhouette_score
from sklearn.metrics import homogeneity_score

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami=adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
        bhi = homogeneity_score(label_df["True"], label_df["Pred"])
    return  ari,nmi,ami,bhi

def run_SpaGCN(adata, dataset, random_seed = np.random.randint(100),
                device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
    """
    Original SpaGCN function (preserved for DLPFC) - works for ALL datasets
    Auto-detects spatial shape (hexagon/square) based on dataset type
    """
    import SpaGCN_raw as spg
    import random

    ##### Preprocessing
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    ##### Get coordinates
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    elif 'spatial_coords' in adata.obsm:
        spatial_coords = adata.obsm['spatial_coords']
    elif {'x','y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['x','y']].values
    elif {'spatial_x','spatial_y'}.issubset(adata.obs.columns):
        spatial_coords = adata.obs[['spatial_x','spatial_y']].values
    else:
        # Fallback dummy coordinates (like your SpatialMGCN code)
        n = adata.n_obs
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(np.arange(side), np.arange(side))
        dummy_coords = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        spatial_coords = dummy_coords[:n]
        adata.obsm['spatial'] = spatial_coords

    # Assign to x_array / y_array for SpaGCN
    adata.obs['x_array'] = spatial_coords[:, 0]
    adata.obs['y_array'] = spatial_coords[:, 1]

    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    adj = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)

    ##### Search l
    p=0.5
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    ##### Search resolution
    n_clusters = len(set(adata.obs["ground_truth"].dropna()))
    r_seed=t_seed=n_seed=random_seed
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1,
                       tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
                       t_seed=t_seed, n_seed=n_seed)

    ##### Train SpaGCN
    clf=spg.SpaGCN()
    clf.set_l(l)
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    
    clf.train(adata, adj, init_spa=True, init="louvain",
              res=res, tol=5e-3, lr=0.05, max_epochs=200)
    emb,y_pred, prob=clf.predict()

    # FIX: Use pd.Categorical instead of astype('category') on numpy array
    adata.obs["pred"] = pd.Categorical(y_pred)
    adata.obsm["embedding"]=emb

    ##### Auto-detect spatial shape for refinement (critical for non-DLPFC data)
    if dataset.startswith('15') or dataset in ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72']:
        shape = "hexagon"  # DLPFC + 10X Visium
    else:
        shape = "square"   # ST + Slide-seq + STARmap + Xenium + SNARE-seq2

    ##### Cluster refinement
    adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(),
                            pred=adata.obs["pred"].tolist(), dis=adj_2d, shape=shape)
    # FIX: Use pd.Categorical here too
    adata.obs["refined_pred"] = pd.Categorical(refined_pred)

    ##### Calculate all metrics
    ari, nmi, ami, bhi = eval_model(adata.obs['pred'], adata.obs['ground_truth'])
    SC = silhouette_score(adata.obsm["embedding"], adata.obs['pred'])

    used_adata = adata[adata.obs["ground_truth"].notna()]
    SC_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['ground_truth'])

    ari_r, nmi_r, ami_r, bhi_r = eval_model(adata.obs['refined_pred'], adata.obs['ground_truth'])
    SC_r = silhouette_score(adata.obsm["embedding"], adata.obs['refined_pred'])
    SC_r_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['refined_pred'])
    
    res = {
        "dataset": dataset,
        "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": SC, "SC_revise": SC_revise,
        "ari_r": ari_r, "nmi_r": nmi_r, "ami_r": ami_r, "bhi_r": bhi_r,
        "sc_r": SC_r, "SC_r_revise": SC_r_revise
    }
    return res, adata

# ====================== MAIN RUNNER (MATCHES CCST STRUCTURE) ======================
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    # PATHS (MATCH YOUR CCST CODE)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\SpaGCN"
    
    
    mk_dir(OUTPUT_ROOT)
    warnings.filterwarnings('ignore')

    # --------------------------
    # EXACT SAME DATA CATEGORIES AS CCST
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

    # FULL DATASET LIST (SAME AS CCST)
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

    # SEED
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    all_results = []

    # RUN ALL DATASETS
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING SPA GCN: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        mk_dir(save_path)

        # ✅ DATA LOADING (SAME AS CCST - uses your utils_for_all.py)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()
        random_seed = np.random.randint(100)

        # --------------------------
        # TIME & MEMORY TRACKING (SAME AS CCST)
        # --------------------------
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 /1024 /1024

        # RUN SPA GCN (ORIGINAL FUNCTION UNCHANGED)
        res, adata_result = run_SpaGCN(adata_h5.copy(), dataset, random_seed=random_seed)

        # End tracking
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss /1024/1024/1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak /1024/1024/1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Add time/memory to results
        res["time"] = used_time
        res["Memo"] = used_mem
        res["Memo_peak"] = peak_gb

        # --------------------------
        # Get data_type (match CCST)
        # --------------------------
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Final result with data type
        full_result = {"data_type": data_type, **res}
        all_results.append(full_result)

        # Save per-dataset outputs
        adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)
        print(f"✅ Completed: {dataset} | "
      f"ARI={res['ari']:.4f}, "
      f"NMI={res['nmi']:.4f}, "
      f"AMI={res['ami']:.4f}, "
      f"BHI={res['bhi']:.4f}, "
      f"SC={res['sc']:.4f},")

    # --------------------------
    # SAVE COMBINED RESULTS (ALL DATASETS)
    # --------------------------
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "SpaGCN_ALL_DATASETS.csv"), index=False)
    print(f"\n🎉 ALL DATASETS COMPLETED! Combined results saved to: {OUTPUT_ROOT}")