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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
# DSTN Model (Original - NO CHANGES)
# --------------------------
class DSTN(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, nhead=4, hidden_dim=128, num_layers=2):
        super(DSTN, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2,
            activation='relu', batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        h = self.proj(x)
        h = self.transformer_encoder(h)
        h = h.squeeze(0)
        z = self.fc(h)
        x_recon = self.decoder(z)
        return z, x_recon

# --------------------------
# Resolution Search (Original - NO CHANGES)
# --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

# --------------------------
# Evaluation Metrics (Original - NO CHANGES)
# --------------------------
def eval_model(pred, gt):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    ami = adjusted_mutual_info_score(gt, pred)
    return ari, nmi, ami

# --------------------------
# UNIVERSAL DSTN RUN FUNCTION
# --------------------------
def run_DSTN(adata_h5, save_path, dataset, epochs=200, n_cluster=10):
    print('===== Using device:', device)
    os.makedirs(save_path, exist_ok=True)

    adata = adata_h5.copy()

    if "Kedney_" in dataset:
        print(f"Auto-subsampling Slide-seq dataset: {dataset} → 8000 cells")
        sc.pp.subsample(adata, n_obs=8000, random_state=0)

    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    max_possible = min(adata.n_obs, adata.n_vars)
    n_comps = min(200, max_possible - 1)
    if n_comps < 1:
        adata.obsm['X_pca'] = adata.X.copy()
    else:
        sc.pp.pca(adata, n_comps=n_comps, svd_solver='arpack')

    X = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)

    model = DSTN(input_dim=n_comps).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("==== Training DSTN ====")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_recon = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embedding = model(X)[0].cpu().numpy()

    adata_result = anndata.AnnData(embedding)
    adata_result.obs['ground_truth'] = adata.obs['ground_truth'].values
    if 'spatial' in adata.obsm:
        adata_result.obsm['spatial'] = adata.obsm['spatial']
        adata_result.uns['spatial'] = adata.uns.get('spatial', {})

    sc.pp.neighbors(adata_result, n_neighbors=20)
    sc.tl.umap(adata_result)
    res = res_search_fixed_clus(adata_result, n_cluster)
    sc.tl.leiden(adata_result, key_added="DSTN_leiden", resolution=res)

    adata_result.write_h5ad(os.path.join(save_path, f"{dataset}.h5ad"))

    try:
        if 'spatial' in adata_result.obsm:
            sc.pl.spatial(adata_result, color="DSTN_leiden", img_key="hires", show=False)
            plt.savefig(os.path.join(save_path, "DSTN_spatial_plot.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
    except:
        print("Spatial plot skipped")

    valid = adata_result.obs['ground_truth'].notna()
    pred = adata_result.obs['DSTN_leiden'][valid]
    gt = adata_result.obs['ground_truth'][valid]

    if len(pred) == 0:
        ari, nmi, ami, bhi, sc_score = 0,0,0,0,0
    else:
        ari, nmi, ami = eval_model(pred, gt)
        bhi = homogeneity_score(gt, pred)
        sc_score = silhouette_score(adata_result.X[valid], pred) if len(np.unique(pred))>1 else 0

    print(f"{dataset} | ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")
    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# --------------------------
# MAIN RUNNER (Only added data_category FIRST column - NO other changes)
# --------------------------
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa
    from sklearn.metrics import homogeneity_score, silhouette_score

    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\DSTN"
    usa.mk_dir(OUTPUT_ROOT)

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

    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING DSTN: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        usa.mk_dir(save_path)

        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 /1024 /1024

        res = run_DSTN(adata_h5, save_path, dataset, epochs=200, n_cluster=n_cluster)

        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss /1024/1024/1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak /1024/1024/1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        res.update({
            "time": used_time,
            "Memo": used_mem,
            "Memo_peak": peak_gb
        })

        # ======================
        # ONLY NEW CODE: Assign your 7 official categories
        # ======================
        if dataset.startswith(('BC_TENX','Lung_TENX')):
            cat = "10X Visium"
        elif dataset.startswith(('BC_NCBI','Lung_NCBI','Colorectal','Ovarian','Skin')):
            cat = "Xenium"
        elif dataset.startswith(('Patient','BC_SPA')):
            cat = "ST"
        elif dataset.startswith('Kedney'):
            cat = "Slide-seq"
        elif dataset.isdigit() or dataset.startswith('151'):
            cat = "DLPFC"
        elif dataset.startswith(('STARmap','BOOST','Tutorial')):
            cat = "STARmap"
        elif dataset.startswith(('LungL','LungR')):
            cat = "SNARE-seq2"
        else:
            cat = "Unknown"

        # Put data_category as FIRST COLUMN
        res_final = {"data_category": cat}
        res_final.update(res)
        
        all_results.append(res_final)
        pd.DataFrame([res_final]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "DSTN_ALL_DATASETS.csv"), index=False)
    print("\nALL DATASETS COMPLETED! Results saved to:", OUTPUT_ROOT)