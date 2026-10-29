# -*- coding:utf-8 -*-
import sys
sys.path.append("D:/My_Project_ST/1_Evaluation_SRT/")
import io

# Force UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time,psutil,tracemalloc
from loss import target_distribution, kl_loss
from util import *
from utils_for_all import *
import argparse
from models import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score,adjusted_rand_score, homogeneity_score
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# ==============================================================================
# ✅ YOUR ORIGINAL FUNCTION - 100% UNCHANGED (Works for ALL datasets)
# ==============================================================================
def train_MSpaGCN(opts):
    start = time.time()
    start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

    if opts.dataset.startswith('15'):
        features_adata,features,labels = load_data(opts.dataset,opts.npca)
    else:
        # Use your existing load_other_data() for ALL non-DLPFC datasets
        features_adata, features, labels = load_other_data(opts.dataset, opts.input_root,opts.npca)

    adj1, adj2 = load_graph_V1(opts.dataset, features_adata,opts.l)

    model =STMGCN(nfeat=features.shape[1],
                    nhid1=opts.nhid1,
                    nclass=opts.n_cluster
                    )
    if opts.cuda:
        model.cuda()
        features = features.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()

    optimizer = optim.Adam(model.parameters(),lr=opts.lr, weight_decay=opts.weight_decay)
    emb = model.mgcn(features,adj1,adj2) #(3639,32)

    if opts.initcluster == "kmeans":
        print("Initializing cluster centers with kmeans, n_clusters known")
        n_clusters=opts.n_cluster
        kmeans = KMeans(n_clusters,n_init=20)
        y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())

    elif opts.initcluster == "louvain":
        print("Initializing cluster centers with louvain,resolution=",opts.res)
        adata=sc.AnnData(emb.detach().cpu().numpy())
        sc.pp.neighbors(adata, n_neighbors=opts.n_neighbors)
        sc.tl.louvain(adata,resolution=opts.res)
        y_pred=adata.obs['louvain'].astype(int).to_numpy()
        n=len(np.unique(y_pred))

    emb=pd.DataFrame(emb.detach().cpu().numpy(),index=np.arange(0,emb.shape[0]))
    Group=pd.Series(y_pred,index=np.arange(0,emb.shape[0]),name="Group")
    Mergefeature=pd.concat([emb,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())

    y_pred_last = y_pred
    with torch.no_grad():
        model.cluster_layer.copy_(torch.tensor(cluster_centers))

    model.train()
    for epoch in range(opts.max_epochs):
        if epoch % opts.update_interval == 0:
            _, tem_q = model(features,adj1,adj2)
            tem_q = tem_q.detach()
            p = target_distribution(tem_q)

            y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            y = labels

            nmi = normalized_mutual_info_score(y, y_pred)
            ari = adjusted_mutual_info_score(y, y_pred)

            print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch>0 and delta_label < opts.tol:
                print('delta_label ', delta_label, '< tol ', opts.tol)
                print("Reach tolerance threshold. Stopping training.")
                break

        optimizer.zero_grad()
        x,q = model(features,adj1,adj2)
        loss = kl_loss(q.log(), p)
        loss.backward()
        optimizer.step()

    # Save embeddings
    key_added = "STMGCN"
    embeddings = pd.DataFrame(x.detach().cpu().numpy())
    embeddings.index = features_adata.obs_names

    features_adata.obsm[key_added] = embeddings.loc[features_adata.obs_names,].values
    features_adata.obs['pred']=y_pred_last

    end = time.time()
    end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    used_memory = end_MB - start_MB

    # Fixed evaluation to handle NaN and unpack error
    valid_mask = features_adata.obs['Ground Truth'].notna().values
    y_true = features_adata.obs['Ground Truth'][valid_mask]
    y_pred = y_pred_last[valid_mask]
    emb_valid = embeddings.iloc[valid_mask]

    ari, nmi, ami = eval_model(y_pred, y_true)
    bhi = homogeneity_score(y_true, y_pred)

    SC = silhouette_score(emb_valid, y_pred)
    used_adata = features_adata[valid_mask]
    SC_revise = silhouette_score(used_adata.obsm["STMGCN"], used_adata.obs['Ground Truth'])

    print(f"sc{SC:5f}")
    res = {}
    res["dataset"] = opts.dataset
    res["ari"] = ari
    res["nmi"] = nmi
    res["ami"] = ami
    res["bhi"] = bhi
    res["sc"] = SC
    res["time"] = end-start
    res['Memo']=used_memory
    res["sc_revise"]=SC_revise
    return features_adata,res

def parser_set(n_cluster,dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--nhid1',type=int,default=32)
    parser.add_argument('--n_cluster',default=n_cluster,type=int)
    parser.add_argument('--max_epochs',default=2000,type=int)
    parser.add_argument('--update_interval',default= 3,type=int)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--weight_decay',default=0.001,type=float)
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--l', default=1, type=float)
    parser.add_argument('--npca', default=50, type=int)
    parser.add_argument('--n_neighbors',type=int,default=10)
    parser.add_argument('--initcluster', default="kmeans", type=str)
    parser.add_argument('--input_root', default=f'D:/My_Project_ST/DataSets/', type=str)
    parser.add_argument('--output_root', default=f'D:/My_Project_ST/Output_For_All/', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args.cuda = torch.cuda.is_available()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    return args

n_clusters_map = {"Mouse_hippocampus": 10, "Mouse_olfactory_slide_seqv2": 9, "MOB_without_label": 7,
                  "PDAC": 4, "Breast_cancer": 20, "Mouse_brain": 15,
                  "SeqFish": 22, "STARmap": 16,"Stereo":16}

if __name__ == "__main__":
    # 1. ORIGINAL DLPFC DATA
    DLPFC_datasets = ['151507', '151508', '151509', '151510','151669', '151670', '151671', '151672', '151673', '151674','151675', '151676']

    # 2. ALL NEW CUSTOM DATA
    NEW_CUSTOM_DATASETS = [
        
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
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882', 'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
    ]

    ALL_DATASETS = DLPFC_datasets + NEW_CUSTOM_DATASETS

    all_results = pd.DataFrame(columns=['dataset','ari','nmi','ami','bhi','sc','sc_revise','time','Memo','Memo_peak','round'])

    for dataset in ALL_DATASETS:
        print(f"\n==================== BEGIN TEST ON {dataset} ======================================")

        if dataset.startswith('15'):
            n_domains = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7
        elif dataset in n_clusters_map:
            n_domains = n_clusters_map[dataset]
        else:
            n_domains = 7

        save_data_path = f'D:/My_Project_ST/Output_For_All/STMGCN/{dataset}/'
        mk_dir(save_data_path)
        opts = parser_set(n_domains, dataset)

        results_df = pd.DataFrame()
        for i in range(1):
            start = time.time()
            tracemalloc.start()
            start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

            # ✅ Use your ORIGINAL working function for ALL datasets
            adata, res = train_MSpaGCN(opts)

            end = time.time()
            used_time = end - start
            used_memo = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 - start_MB
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_gb = peak / 1024 / 1024 / 1024
            tracemalloc.clear_traces()

            res["time"] = used_time
            res["Memo"] = used_memo
            res["Memo_peak"] = peak_gb
            res["round"] = i + 1

            results_df = pd.concat([results_df, pd.DataFrame([res])], ignore_index=True)
            all_results = pd.concat([all_results, pd.DataFrame([res])], ignore_index=True)

        results_df.to_csv(save_data_path + "/{}_result.csv".format(dataset), index=False)
        adata.write(f'{save_data_path}/STMGCN_{dataset}.h5ad')

        results_df.set_index('dataset', inplace=True)
        results_df.mean().to_csv(f'{save_data_path}{dataset}_mean.csv', header=True)
        results_df.std().to_csv(f'{save_data_path}{dataset}_std.csv', header=True)
        results_df.median().to_csv(f'{save_data_path}{dataset}_median.csv', header=True)

    final_csv_path = "D:/My_Project_ST/Output_For_All/STMGCN/STMGCN_ALL_DATASETS_FINAL.csv"
    all_results.to_csv(final_csv_path, index=False)
    print("\n✅ ALL DATASETS FINISHED SUCCESSFULLY!")
    print("RESULTS SAVED TO:", final_csv_path)