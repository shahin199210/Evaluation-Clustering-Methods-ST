# -*- coding:utf-8 -*-
import sys
import io

# Force UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time, psutil, tracemalloc
from loss import target_distribution, kl_loss
from util import *
import argparse
from models import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import torch
import os
sys.path.append('../')
import utils_for_all as usa  # UNIFIED DATA LOADER (matches CCST)

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# --------------------------
# ORIGINAL TRAIN FUNCTION (UNCHANGED for DLPFC)
# --------------------------
def train_MSpaGCN(opts, features_adata=None, features=None, labels=None, adj1=None, adj2=None):
    start = time.time()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

    model = STMGCN(nfeat=features.shape[1],
                    nhid1=opts.nhid1,
                    nclass=opts.n_cluster
                    )
    if opts.cuda:
        model.cuda()
        features = features.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    emb = model.mgcn(features, adj1, adj2)

    # Initial clustering (kmeans/louvain)
    if opts.initcluster == "kmeans":
        print("Initializing cluster centers with kmeans, n_clusters known")
        n_clusters = opts.n_cluster
        kmeans = KMeans(n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())

    elif opts.initcluster == "louvain":
        print("Initializing cluster centers with louvain,resolution=", opts.res)
        adata = sc.AnnData(emb.detach().cpu().numpy())
        sc.pp.neighbors(adata, n_neighbors=opts.n_neighbors)
        sc.tl.louvain(adata, resolution=opts.res)
        y_pred = adata.obs['louvain'].astype(int).to_numpy()

    # Cluster center initialization
    emb = pd.DataFrame(emb.detach().cpu().numpy(), index=np.arange(0, emb.shape[0]))
    Group = pd.Series(y_pred, index=np.arange(0, emb.shape[0]), name="Group")
    Mergefeature = pd.concat([emb, Group], axis=1)
    cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

    y_pred_last = y_pred
    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(cluster_centers))

    model.train()
    for epoch in range(opts.max_epochs):
        if epoch % opts.update_interval == 0:
            _, tem_q = model(features, adj1, adj2)
            tem_q = tem_q.detach()
            p = target_distribution(tem_q)

            y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            y = labels

            nmi = normalized_mutual_info_score(y, y_pred)
            ari = adjusted_mutual_info_score(y, y_pred)

            print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < opts.tol:
                print('delta_label ', delta_label, '< tol ', opts.tol)
                print("Reach tolerance threshold. Stopping training.")
                break

        optimizer.zero_grad()
        x, q = model(features, adj1, adj2)
        loss = kl_loss(q.log(), p)
        loss.backward()
        optimizer.step()

    # Save embeddings
    key_added = "STMGCN"
    embeddings = pd.DataFrame(x.detach().cpu().numpy())
    embeddings.index = features_adata.obs_names
    features_adata.obsm[key_added] = embeddings.loc[features_adata.obs_names,].values
    features_adata.obs['pred'] = y_pred_last

    # Metrics
    end = time.time()
    end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    used_memory = end_MB - start_MB

    ari, nmi, ami, bhi = eval_model(y_pred_last, features_adata.obs['Ground Truth'])
    SC = silhouette_score(embeddings, y_pred_last)
    used_adata = features_adata[features_adata.obs["Ground Truth"].notna()]
    SC_revise = silhouette_score(used_adata.obsm["STMGCN"], used_adata.obs['Ground Truth'])

    res = {
        "dataset": opts.dataset,
        "ari": ari, "nmi": nmi, "ami": ami, "bhi": bhi,
        "sc": SC, "sc_revise": SC_revise,
        "time": end - start, "Memo": used_memory
    }
    return features_adata, res

# --------------------------
# ARGUMENT PARSER (UNCHANGED)
# --------------------------
def parser_set(n_cluster, dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nhid1', type=int, default=32)
    parser.add_argument('--n_cluster', default=n_cluster, type=int)
    parser.add_argument('--max_epochs', default=2000, type=int)
    parser.add_argument('--update_interval', default=3, type=int)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--l', default=1, type=float)
    parser.add_argument('--npca', default=50, type=int)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--initcluster', default="kmeans", type=str)
    parser.add_argument('--input_root', default=f'D:/My_Project_ST/DataSets/', type=str)
    parser.add_argument('--output_root', default=f'D:/My_Project_ST/Output_For_All/STMGCN/', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    args.cuda = torch.cuda.is_available()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    return args

# --------------------------
# NEW: STMGCN PIPELINE FOR ALL NON-DLPFC DATASETS
# --------------------------
def run_STMGCN_all(adata_h5, save_path, dataset, n_cluster):
    """Unified STMGCN pipeline for all non-DLPFC datasets"""
    # Load graph + preprocess (matches original STMGCN logic)
    features_adata = adata_h5.copy()
    features_adata, features, labels = load_other_data(dataset, features_adata, npca=50)
    adj1, adj2 = load_graph_V1(dataset, features_adata, l=1)
    
    # Set options
    opts = parser_set(n_cluster, dataset)
    opts.dataset = dataset
    opts.n_cluster = n_cluster
    
    # Train STMGCN
    adata_result, res = train_MSpaGCN(
        opts,
        features_adata=features_adata,
        features=features,
        labels=labels,
        adj1=adj1,
        adj2=adj2
    )
    
    # Save results
    adata_result.write(f'{save_path}/STMGCN_{dataset}.h5ad')
    return res, adata_result

# --------------------------
# MAIN RUNNER (MATCHES CCST STRUCTURE)
# --------------------------
if __name__ == "__main__":
    # PATHS (MATCH CCST)
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\STMGCN"
    mk_dir(OUTPUT_ROOT)

    # DATASET CATEGORIES (EXACT SAME AS CCST)
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

    # SEEDS
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    all_results = []

    # RUN ALL DATASETS
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING STMGCN: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        mk_dir(save_path)

        # LOAD DATA VIA utils_for_all.py (SAME AS CCST)
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)

        # TIME & MEMORY TRACKING
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        # RUN STMGCN (DIFFERENT LOGIC FOR DLPFC VS OTHERS)
        if dataset.startswith('15'):
            # ORIGINAL DLPFC CODE
            n_domains = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
            opts = parser_set(n_domains, dataset)
            adata_result, res = train_MSpaGCN(opts)
        else:
            # NEW UNIFIED PIPELINE FOR ALL OTHER DATASETS
            res, adata_result = run_STMGCN_all(adata_h5, save_path, dataset, n_cluster)

        # FINAL TIME & MEMORY
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / 1024 / 1024 / 1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # GET DATA TYPE
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # BUILD FULL RESULT
        full_result = {
            "data_type": data_type,
            "dataset": dataset,
            "ari": res["ari"],
            "nmi": res["nmi"],
            "ami": res["ami"],
            "bhi": res["bhi"],
            "sc": res["sc"],
            "sc_revise": res.get("sc_revise", 0),
            "time": used_time,
            "Memo": used_mem,
            "Memo_peak": peak_gb
        }

        all_results.append(full_result)
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    # SAVE COMBINED RESULTS
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "STMGCN_ALL_DATASETS.csv"), index=False)
    print("\n✅ ALL DATASETS COMPLETED! Results saved to:", OUTPUT_ROOT)