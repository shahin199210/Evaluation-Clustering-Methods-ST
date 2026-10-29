import sys
import os
import time
import psutil
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# --------------------------
# Basic Setup
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
# Metrics Import
# --------------------------
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.decomposition import PCA

# --------------------------
# GraphST Model Import
# --------------------------
from GraphST import GraphST

# --------------------------
# Evaluation Function
# --------------------------
def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami = adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
        return ari, nmi, ami
    return None, None, None

# --------------------------
# Resolution Search for Fixed Clusters
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# Core GraphST Function with PCA Embedding
# --------------------------
def run_GraphST(adata, dataset, random_seed=0):
    print("\nUsing PCA embedding (fixed for all datasets)")
    
    X_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    pca = PCA(n_components=50, random_state=random_seed)
    adata.obsm["emb"] = pca.fit_transform(X_mat)

    emb = adata.obsm["emb"]
    print(f"Embedding shape: {emb.shape}")

    # Clustering
    sc.pp.neighbors(adata, use_rep="emb", n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.8, random_state=random_seed)
    adata.obs["domain"] = adata.obs["leiden"].astype(str)

    # Calculate evaluation metrics
    if "ground_truth" in adata.obs and adata.obs["ground_truth"].notna().any():
        ground_truth = adata.obs["ground_truth"].dropna()
        pred = adata.obs["domain"][ground_truth.index]
        ari, nmi, ami = eval_model(pred, ground_truth)
        bhi = homogeneity_score(ground_truth, pred)
    else:
        ari, nmi, ami, bhi = None, None, None, None

    unique_clusters = adata.obs['domain'].nunique()
    if unique_clusters >= 2:
        SC = silhouette_score(emb, adata.obs['domain'])
    else:
        print(f"Warning: Only {unique_clusters} cluster found.")
        SC = float('nan')

    print(f"{dataset} Scores: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={SC:.4f}")

    res = {
        "dataset": dataset,
        "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": SC,
        "ari_r": ari, "nmi_r": nmi, "ami_r": ami, "sc_r": SC
    }
    adata.obs["pred_label"] = adata.obs['domain']
    return res, adata

# --------------------------
# Universal GraphST Pipeline
# --------------------------
def run_GraphST_Pipeline(adata_h5, save_path, dataset, n_cluster=10):
    print('Using device:', device)
    os.makedirs(save_path, exist_ok=True)

    adata = adata_h5.copy()
    adata.var_names_make_unique()

    # Run GraphST core function
    res, adata_result = run_GraphST(adata, dataset)

    # Copy ground truth and spatial information
    adata_result.obs['ground_truth'] = adata_h5.obs['ground_truth'].values
    if 'spatial' in adata_h5.obsm:
        adata_result.obsm['spatial'] = adata_h5.obsm['spatial']
        adata_result.uns['spatial'] = adata_h5.uns.get('spatial', {})

    # Save h5ad file (pred_label embedded inside)
    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    # Export cell-level predicted labels CSV
    pred_export_df = adata_result.obs[["pred_label"]].copy()
    if "ground_truth" in adata_result.obs.columns:
        pred_export_df["ground_truth"] = adata_result.obs["ground_truth"]
    pred_export_df.index.name = "cell_barcode"
    pred_csv_path = os.path.join(save_path, f"{dataset}_cell_pred_labels.csv")
    pred_export_df.to_csv(pred_csv_path)
    print(f"Predicted labels saved: {pred_csv_path}")

    # Optional save numpy label array
    pred_np = adata_result.obs["pred_label"].values
    np.save(os.path.join(save_path, f"{dataset}_pred_labels.npy"), pred_np)

    # ========== NEW ADD: Calculate actual prediction labels & re-verify metrics for STARmap_AD_13m_disease ==========
    if dataset == "STARmap_AD_13m_disease":
        print("\n========== VERIFY ACTUAL PREDICTION LABELS FOR STARmap_AD_13m_disease ==========")
        # Load exported label CSV
        label_df = pd.read_csv(pred_csv_path)
        true_labels = label_df["ground_truth"].values
        pred_labels = label_df["pred_label"].values

        # Recalculate metrics from actual saved prediction labels
        ari_recalc = adjusted_rand_score(true_labels, pred_labels)
        nmi_recalc = normalized_mutual_info_score(true_labels, pred_labels)
        ami_recalc = adjusted_mutual_info_score(true_labels, pred_labels)
        bhi_recalc = homogeneity_score(true_labels, pred_labels)

        # Count matching / mismatched cells
        total_cells = len(label_df)
        mismatch_mask = (label_df["pred_label"] != label_df["ground_truth"])
        mismatch_count = mismatch_mask.sum()
        match_count = total_cells - mismatch_count
        match_ratio = match_count / total_cells

        # Print detailed statistics
        print(f"Total Cells: {total_cells}")
        print(f"Matched Cells: {match_count} ({match_ratio:.2%})")
        print(f"Mismatched Cells: {mismatch_count} ({1-match_ratio:.2%})")
        print("\nRecomputed Metrics (from actual saved pred_label):")
        print(f"ARI = {ari_recalc:.6f}")
        print(f"NMI = {nmi_recalc:.6f}")
        print(f"AMI = {ami_recalc:.6f}")
        print(f"Homogeneity(BHI) = {bhi_recalc:.6f}")

        # Save mismatched cell list to separate CSV for manual check
        mismatch_df = label_df[mismatch_mask].copy()
        mismatch_csv = os.path.join(save_path, f"{dataset}_mismatched_cells.csv")
        mismatch_df.to_csv(mismatch_csv, index=False)
        print(f"Mismatched cell details saved to: {mismatch_csv}")

        # Save pure prediction labels only (no ground truth)
        pred_only_df = label_df[["cell_barcode", "pred_label"]].copy()
        pred_only_csv = os.path.join(save_path, f"{dataset}_only_prediction_labels.csv")
        pred_only_df.to_csv(pred_only_csv, index=False)
        print(f"Pure prediction labels exported separately to: {pred_only_csv}")
        print("==================================================================================\n")
    # =====================================================================================================

    # Spatial plot save logic
    plot_success = False
    if 'spatial' in adata_result.obsm:
        try:
            sc.pl.spatial(adata_result, color='pred_label', img_key="hires", show=False)
            plt.savefig(os.path.join(save_path, "GraphST_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
            plot_success = True
            print(f"Spatial plot saved successfully for {dataset}")
        except Exception as e:
            print(f"ERROR: Failed to save spatial plot for {dataset}, detailed reason: {str(e)}")
    else:
        print(f"Skip plot for {dataset}: adata.obsm['spatial'] does not exist")

    return res

# --------------------------
# Main Runner
# --------------------------
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\Ensamble_model\GraphST"
    usa.mk_dir(OUTPUT_ROOT)

    ALL_DATASETS = [      
       '151669',"BC_SPA130",'Kedney_HBM965.PSNC.855','LungR_HBM753.NSKJ.788',"STARmap_AD_13m_disease",'Lung_TENX141'
    ]

    def get_data_category(dataset_name):
        if dataset_name.startswith(('BC_TENX', 'Lung_TENX')):
            return '10X Visium'
        elif dataset_name.startswith(('BC_NCBI', 'Lung_NCBI', 'Colorectal_TENX', 'Ovarian_TENX', 'Skin_TENX')):
            return 'Xenium'
        elif dataset_name.startswith(('Patient', 'BC_SPA')):
            return 'ST'
        elif dataset_name.startswith('Kedney'):
            return 'Slide-seq'
        elif dataset_name.isdigit() or dataset_name.startswith('151'):
            return 'DLPFC'
        elif dataset_name.startswith(('STARmap', 'BOOST', 'Tutorial')):
            return 'STARmap'
        elif dataset_name.startswith(('LungL', 'LungR')):
            return 'SNARE-seq2'
        else:
            return 'Unknown'

    all_results = []

    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING GraphST: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)

        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 /1024 /1024

        res = run_GraphST_Pipeline(adata_h5, save_path, dataset, n_cluster=n_cluster)

        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss /1024/1024/1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak /1024/1024/1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        data_category = get_data_category(dataset)

        res.update({
            "data_type": data_category,
            "time": used_time,
            "Memo": used_mem,
            "Memo_peak": peak_gb
        })

        res_ordered = {"data_type": data_category}
        res_ordered.update(res)

        all_results.append(res_ordered)

    # Only integrated total metrics CSV saved at new root GraphST folder
    final_df = pd.DataFrame(all_results)
    final_csv_path = os.path.join(OUTPUT_ROOT, "Ensemble_GraphST_Integrated_All_Datasets_Results.csv")
    final_df.to_csv(final_csv_path, index=False)
    print(f"\n✅ All datasets finished! Integrated total results saved to:\n{final_csv_path}")