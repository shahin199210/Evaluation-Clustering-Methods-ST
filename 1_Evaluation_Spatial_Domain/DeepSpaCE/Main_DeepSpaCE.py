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
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, homogeneity_score,
                             silhouette_score)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

np.random.seed(0)
import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --------------------------
# DeepSpaCE Model Definition
# --------------------------
class DeepSpaCE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 64, 32]):
        super(DeepSpaCE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[2], hidden_dims[3])
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[3], hidden_dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[2], hidden_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# --------------------------
# Find resolution for fixed cluster number
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

# --------------------------
# Evaluation metrics
# --------------------------
def eval_model(pred, gt):
    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, nmi, ami

# --------------------------
# MAIN Universal DeepSpaCE Function (for ALL datasets)
# --------------------------
def run_DeepSpaCE(adata_h5, save_path, dataset, epochs=200, n_cluster=10):
    os.makedirs(save_path, exist_ok=True)
    print('===== Using device: ' + device)

    # Preprocessing
    sc.pp.filter_genes(adata_h5, min_cells=5)
    sc.pp.normalize_total(adata_h5, target_sum=1e4)
    sc.pp.log1p(adata_h5)
    sc.pp.scale(adata_h5)

    max_possible = min(adata_h5.n_obs, adata_h5.n_vars)
    n_comps = min(200, max_possible - 1)
    if n_comps < 1:
        adata_h5.obsm['X_pca'] = adata_h5.X.copy()
    else:
        sc.pp.pca(adata_h5, n_comps=n_comps, svd_solver='arpack')

    X = torch.tensor(adata_h5.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    model = DeepSpaCE(input_dim=n_comps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print("==== Start DeepSpaCE Training ====")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, X_recon = model(X)
        loss = criterion(X_recon, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embedding = model.encoder(X).cpu().numpy()

    adata_result = anndata.AnnData(embedding)
    adata_result.obs['ground_truth'] = adata_h5.obs['ground_truth'].values
    if 'spatial' in adata_h5.obsm:
        adata_result.obsm['spatial'] = adata_h5.obsm['spatial']
        adata_result.uns['spatial'] = adata_h5.uns.get('spatial', {})

    sc.pp.neighbors(adata_result, n_neighbors=20)
    sc.tl.umap(adata_result)
    eval_res = res_search_fixed_clus(adata_result, n_cluster)
    sc.tl.leiden(adata_result, key_added="DeepSpaCE_leiden", resolution=eval_res)

    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    try:
        sc.pl.spatial(adata_result, img_key="hires", color="DeepSpaCE_leiden", show=False)
        plt.savefig(os.path.join(save_path, "DeepSpaCE_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        print("⚠️ Spatial plot not available")

    valid_mask = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['DeepSpaCE_leiden'][valid_mask]
    gt = adata_result.obs['ground_truth'][valid_mask]

    if len(pred) == 0:
        ari, nmi, ami, bhi, sc_score = 0, 0, 0, 0, 0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata_result.X[valid_mask], pred) if len(np.unique(pred)) > 1 else 0

    print(f"✅ {dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")

    return {
        "dataset": dataset,
        "ari": ari,
        "nmi": nmi,
        "ami": ami,
        "bhi": bhi,
        "sc": sc_score
    }

# --------------------------
# Main execution (ALL DATASETS)
# --------------------------
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\DeepSpaCE"
    usa.mk_dir(OUTPUT_ROOT)

    # --------------------------
    # Define dataset categories (matches your GraphST/CCST outputs)
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

    # Run every dataset
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING DeepSpaCE: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        tracemalloc.start()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        res = run_DeepSpaCE(adata_h5, save_path, dataset, epochs=200, n_cluster=n_cluster)

        end_time = time.time()
        end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        current, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()

        used_time = end_time - start_time
        used_mem = end_mem - start_mem
        peak_gb = peak_mem / 1024 / 1024 / 1024

        print(f'Time: {used_time:.2f}s | Mem: {used_mem:.4f}GB | Peak: {peak_gb:.4f}GB')

        # --------------------------
        # Add data_type and build full result row
        # --------------------------
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Combine all fields, with data_type first
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

    # --------------------------
    # Final DataFrame with correct column order
    # --------------------------
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "DeepSpaCE_ALL_DATASETS.csv"), index=False)
    print("\n✅ ALL DATASETS COMPLETED FOR DeepSpaCE!")