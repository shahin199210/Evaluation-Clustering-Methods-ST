import sys
import os
import torch
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import time
import psutil
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================== YOUR ORIGINAL CODE ======================
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

def eval_model(pred, labels):
    label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
    ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
    nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
    ami = adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
    return ari, nmi, ami

class CCST(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 64], num_clusters=10, dropout=0.2):
        super(CCST, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU()
        )
        self.cluster_layer = torch.nn.Linear(hidden_dims[1], num_clusters)
        self.decoder = torch.nn.Linear(hidden_dims[1], input_dim)

    def forward(self, x):
        feat = self.encoder(x)
        recon = self.decoder(feat)
        cluster_prob = torch.softmax(self.cluster_layer(feat), dim=1)
        return feat, recon, cluster_prob

# ====================== FULLY FIXED train_ccst (Auto PCA + No Errors) ======================
def train_ccst(adata, epochs=5, lr=0.001, hidden_dim=64, num_clusters=10, device='cuda:0'):
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    max_possible = min(adata.n_obs, adata.n_vars)
    n_comps = min(200, max_possible - 1)
    
    if n_comps < 1:
        print(f"Dataset too small (min(n_obs, n_vars)={max_possible}), skipping PCA")
        adata.obsm['X_pca'] = adata.X.copy()
    else:
        print(f"Using PCA components: {n_comps} (max allowed for this dataset = {max_possible-1})")
        sc.pp.pca(adata, n_comps=n_comps, svd_solver='arpack')

    x = torch.tensor(adata.obsm['X_pca'].copy(), dtype=torch.float32).to(device)
    model = CCST(input_dim=n_comps, hidden_dims=[256, hidden_dim], num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        feat, recon, prob = model(x)
        recon_loss = mse_loss(recon, x)
        cluster_loss = -torch.mean(torch.sum(prob * torch.log(prob + 10e-10), dim=1))
        loss = recon_loss + 0.05 * cluster_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings, _, _ = model(x)
    adata.obsm['CCST_emb'] = embeddings.cpu().numpy()
    return adata

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        if len(adata.obs['leiden'].unique()) == fixed_clus_count:
            break
    return res

def run_CCST(adata_h5, save_path, dataset, epochs=200, n_cluster=10):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)

    mk_dir(save_path)
    adata_ccst = train_ccst(adata_h5.copy(), epochs=epochs, num_clusters=n_cluster, device=device)

    # Safe ground truth handling (FIXED)
    if 'ground_truth' in adata_h5.obs.columns:
        adata_ccst.obs['ground_truth'] = adata_h5.obs['ground_truth'].values
    else:
        raise ValueError("ERROR: No ground_truth found in adata.obs!")

    if 'spatial' in adata_h5.obsm:
        adata_ccst.obsm['spatial'] = adata_h5.obsm['spatial']
        adata_ccst.uns['spatial'] = adata_h5.uns.get('spatial', {})

    sc.pp.neighbors(adata_ccst, use_rep='CCST_emb', n_neighbors=20, random_state=0)
    sc.tl.umap(adata_ccst, random_state=0)
    eval_resolution = res_search_fixed_clus(adata_ccst, n_cluster)
    sc.tl.leiden(adata_ccst, key_added="CCST_leiden", resolution=eval_resolution, random_state=0)

    adata_ccst.write_h5ad(f'{save_path}/{dataset}.h5ad')

    try:
        if 'spatial' in adata_ccst.obsm:
            sc.pl.spatial(adata_ccst, img_key="hires", color=['CCST_leiden'], show=False)
            plt.savefig(os.path.join(save_path, "CCST_leiden_plot.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
    except:
        print("Plot skipped")

    valid_mask = adata_ccst.obs['ground_truth'].notna()
    pred = adata_ccst.obs['CCST_leiden'][valid_mask]
    gt = adata_ccst.obs['ground_truth'][valid_mask]

    ari, nmi, ami = eval_model(pred, gt)
    bhi = homogeneity_score(gt, pred)
    sc_score = silhouette_score(adata_ccst.obsm['CCST_emb'][valid_mask], pred) if len(np.unique(pred)) > 1 else 0.0

    print(f" {dataset} Scores: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}, BHI={bhi:.4f}, SC={sc_score:.4f}")

    return {"dataset": dataset, "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi, "sc": sc_score}

# ====================== MAIN RUNNER ======================
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa

    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\CCST"
    mk_dir(OUTPUT_ROOT)

    # ----------------------
    # All datasets (GROUPED AS IN YOUR IMAGE)
    # ----------------------
    # 10X Visium (DLPFC + commercial)
    Dataset_DLPFC = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    Dataset_10X_Visium = ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72']
    
    # ST
    Dataset_ST = [
        "Patient1_Biopsy1_rep1_chip1_d1_stdata",
        "Patient3_Biopsy2_rep2_Chip3_C2_stdata",
        "Patient3_Biopsy4_rep2_Chip3_E2_stdata",
        "BC_SPA130"
    ]
    
    # SNARE-seq2
    Dataset_SNARE_seq2 = [
        'LungL_HBM477.TMMN.763',
        'LungL_HBM582.RKVK.974',
        'LungR_HBM285.MWNX.593',
        'LungR_HBM753.NSKJ.788'
    ]
    
    # Slide-seq
    Dataset_SlideI_seq = [
        'Kedney_HBM232.MBNR.586',
        'Kedney_HBM478.VWXX.362',
        'Kedney_HBM965.PSNC.855'
    ]

    # ✅ UPDATED Xenium datasets (your 3 new ones + existing)
    Dataset_Xenium = [
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
    ]
    
    # STARmap
    Dataset_STARmap = [
        "STARmap_AD_13m_disease",
        "BOOST_HMI_starmap",
        "Tutorial_STARmap"
    ]

    # ✅ YOUR REQUESTED RUN ORDER: STARMAP FIRST
    ALL_DATASETS = (
        Dataset_STARmap
        + Dataset_SNARE_seq2
        + Dataset_Xenium
        + Dataset_ST
        + Dataset_SlideI_seq
        + Dataset_10X_Visium
        + Dataset_DLPFC
    )

    # ----------------------
    # Cluster settings
    # ----------------------
    n_clusters_map = {
        # DLPFC
        '151669':5,'151670':5,'151671':5,'151672':5,
        '151507':7,'151508':7,'151509':7,'151510':7,'151673':7,'151674':7,'151675':7,'151676':7,

        # Visium
        'BC_TENX13':15,'BC_TENX23':15,'BC_TENX39':15,'Lung_TENX62':15,'Lung_TENX72':15,

        # Xenium (including your 3 new downsampled datasets)
        'BC_NCBI784':15,'Lung_NCBI856':15,'Lung_NCBI858':15,'Lung_NCBI882':15,'Lung_TENX141':15,
        'Colorectal_TENX139':15, 'Ovarian_TENX142':15, 'Skin_TENX158':15,

        # Slide-seq
        'Kedney_HBM232.MBNR.586':15,
        'Kedney_HBM478.VWXX.362':15,
        'Kedney_HBM965.PSNC.855':15,

        # SNARE_seq2
        'LungL_HBM477.TMMN.763':15,
        'LungL_HBM582.RKVK.974':15,
        'LungR_HBM285.MWNX.593':15,
        'LungR_HBM753.NSKJ.788':15,
        
        # ST datasets
        **{dataset: 15 for dataset in Dataset_ST},

        # STARmap datasets
        "STARmap_AD_13m_disease": 15,
        "BOOST_HMI_starmap": 15,
        "Tutorial_STARmap": 15
    }

    warnings.filterwarnings('ignore')
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    all_results = []

    for idx, dataset in enumerate(ALL_DATASETS):
        print(f"\n==================================================")
        print(f"            RUNNING: {dataset}")
        print(f"==================================================")
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        mk_dir(save_path)
        n_cluster = n_clusters_map[dataset]

        # --------------------------
        # AUTOMATICALLY ASSIGN data_type and path
        # --------------------------
        if dataset in Dataset_SNARE_seq2:
            data_type = "SNARE_seq2"
            # ✅ FIX 2: FOLDER NAME → SNARE-seq2 (HYPHEN)
            subdir = "SNARE-seq2"
            read_data_path = os.path.join(BASE_DATA_PATH, subdir, dataset)
        elif dataset in Dataset_ST:
            data_type = "ST"
            subdir = "ST"
            st_folder = [
                "1_Patient1_Biopsy1_rep1_chip1_d1_stdata",
                "2_Patient3_Biopsy2_rep2_Chip3_C2_stdata",
                "3_Patient3_Biopsy4_rep2_Chip3_E2_stdata",
                "4_BC_SPA130"
            ][idx - len(Dataset_STARmap) - len(Dataset_SNARE_seq2) - len(Dataset_Xenium)]
            read_data_path = os.path.join(BASE_DATA_PATH, subdir, st_folder)
        elif dataset in Dataset_SlideI_seq:
            data_type = "Slide_Seq"
            subdir = "SlideI_seq"
            read_data_path = os.path.join(BASE_DATA_PATH, subdir, dataset)
        elif dataset in Dataset_10X_Visium:
            data_type = "10X_Visium"
            subdir = "10X_Visium"
            read_data_path = os.path.join(BASE_DATA_PATH, subdir, dataset)
        elif dataset in Dataset_Xenium:
            data_type = "Xenium"
            subdir = "Xenium"
            read_data_path = os.path.join(BASE_DATA_PATH, subdir, dataset)
        elif dataset in Dataset_DLPFC:
            data_type = "DLPFC"
            subdir = None
        elif dataset in Dataset_STARmap:
            data_type = "STARmap"
            read_data_path = os.path.join(BASE_DATA_PATH, "StarMAP")
        else:
            print(f"Unknown dataset type: {dataset}")
            continue

        # Load data
        if data_type != "DLPFC":
            if data_type == "ST":
                if dataset == "BC_SPA130":
                    labeled_file = "SPA130_with_labels.h5ad"
                else:
                    labeled_file = f"{dataset}.h5ad"
            elif data_type == "STARmap":
                if dataset == "STARmap_AD_13m_disease":
                    subfolder = "STARmap_AD"
                    labeled_file = "STARmap_AD_13m_disease_rep1_with_labels.h5ad"
                elif dataset == "BOOST_HMI_starmap":
                    subfolder = "BOOST_HMI"
                    labeled_file = "BOOST_HMI_starmap_final_fixed.h5ad"
                elif dataset == "Tutorial_STARmap":
                    # ✅ FIXED: YOUR REAL FOLDER NAME
                    subfolder = "20180505_BY3_1k"
                    labeled_file = "STARmap_20180505_BY3_1k.h5ad"
                read_data_path = os.path.join(BASE_DATA_PATH, "StarMAP", subfolder)
            elif data_type == "Xenium":
                # ✅ ONLY HERE I REPLACED TO YOUR PRE-MODIFIED FILES
                if dataset in ["Colorectal_TENX139", "Ovarian_TENX142", "Skin_TENX158"]:
                    labeled_file = f"{dataset.split('_')[-1]}_with_labels_downsampled_20k_with_realGT.h5ad"
                else:
                    labeled_file = f"{dataset.split('_', 1)[1]}_with_labels.h5ad"
            else:
                labeled_file = f"{dataset.split('_', 1)[1]}_with_labels.h5ad"

            full_path = os.path.join(read_data_path, labeled_file)
            print(f"Loading file: {full_path}")
            adata_h5 = sc.read_h5ad(full_path)
            
            if dataset == "Tutorial_STARmap" and "label" in adata_h5.obs.columns:
                adata_h5.obs["ground_truth"] = adata_h5.obs["label"].astype(str)
            
            adata_h5.var_names_make_unique()
        else:
            adata_h5, n_cluster = usa.get_adata(dataset)
            adata_h5.var_names_make_unique()

        # Run pipeline
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        res = run_CCST(adata_h5, save_path, dataset, epochs=200, n_cluster=n_cluster)

        used_time = time.time() - start
        used_memo = (psutil.Process(os.getpid()).memory_info().rss / 1024/1024/1024) - start_MB
        current, peak = tracemalloc.get_traced_memory()
        peak = peak / 1024.0 / 1024.0 / 1024.0
        tracemalloc.stop()
        tracemalloc.clear_traces()

        res = {
            "data_type": data_type,
            **res
        }

        res.update({"time": used_time, "Memo": used_memo, "Memo_peak": peak})
        all_results.append(res)
        pd.DataFrame([res]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    # Save combined results with data_type
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "CCST_ALL_DATASETS.csv"), index=False)
    print("\n ALL DATASETS DONE — NO MORE ERRORS!")