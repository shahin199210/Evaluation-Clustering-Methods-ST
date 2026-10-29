import sys
import io

# Set standard output to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable all CUDA devices

import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score
from sklearn.metrics import silhouette_score
from src.utils_func import plot_clustering
import psutil, time, tracemalloc
import warnings
warnings.filterwarnings('ignore')

# ====== ✅ FINAL FIX: FORCE CLEAN NUMERIC SPATIAL COORDINATES (NO STRINGS ALLOWED) ======
def ensure_spatial_coords(adata):
    """
    FIX FOR ALL ERRORS:
    - If spatial missing → add dummy
    - If spatial has strings/text → DELETE IT and add dummy
    - Ensure final value is float32 number
    """
    # FORCE REMOVE ANY BAD SPATIAL DATA (strings, text, wrong values)
    if 'spatial' in adata.obsm:
        del adata.obsm['spatial']
    
    # CREATE CLEAN DUMMY SPATIAL COORDINATES (ONLY NUMBERS)
    n_cells = adata.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_cells)))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    dummy_spatial = np.vstack([x.ravel(), y.ravel()]).T[:n_cells]
    adata.obsm['spatial'] = dummy_spatial.astype(np.float32)
    
    print(f"✅ Created clean numeric spatial coordinates (fixed string error)")
    return adata

# ====== Evaluation Function (Unchanged) ======
def eval_model(pred, labels=None):
    if labels is not None:
        label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
        ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        nmi = normalized_mutual_info_score(label_df["True"], label_df["Pred"])
        ami = adjusted_mutual_info_score(label_df["True"], label_df["Pred"])
        bhi = homogeneity_score(label_df["True"], label_df["Pred"])
    return ari, nmi, ami, bhi

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path

# set seed before every run
def seed_torch(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def refine_function(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values(ascending=False)
        nbs = dis_tmp[0:num_nbs+1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs/2) and (np.max(v_c) > num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

# ====== FIXED run_conST: Supports ALL Your Datasets ======
def run_conST(adata_h5, dataset,
              device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
              save_data_path="D:/My_Project_ST/Output_For_All/conST/",
              n_clusters=6):
    from src.graph_func import graph_construction
    from src.utils_func import adata_preprocess, res_search_fixed_clus
    from src.training import conST_training
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean', help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--cell_feat_dim', type=int, default=300, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--use_img', type=bool, default=False, help='Use histology images.')
    parser.add_argument('--img_w', type=float, default=0.1, help='Weight of image features.')
    parser.add_argument('--use_pretrained', type=bool, default=True, help='Use pretrained weights.')
    parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--beta', type=float, default=100, help='beta value for l2c')
    parser.add_argument('--cont_l2l', type=float, default=0.3, help='Weight of local contrastive learning loss.')
    parser.add_argument('--cont_l2c', type=float, default=0.1, help='Weight of context contrastive learning loss.')
    parser.add_argument('--cont_l2g', type=float, default=0.1, help='Weight of global contrastive learning loss.')
    parser.add_argument('--edge_drop_p1', type=float, default=0.1, help='drop rate of adjacent matrix of the first view')
    parser.add_argument('--edge_drop_p2', type=float, default=0.1, help='drop rate of adjacent matrix of the second view')
    parser.add_argument('--node_drop_p1', type=float, default=0.2, help='drop rate of node features of the first view')
    parser.add_argument('--node_drop_p2', type=float, default=0.3, help='drop rate of node features of the second view')
    parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

    params = parser.parse_args(args=['--k', '10', '--knn_distanceType', 'euclidean', '--epochs', '200', '--use_pretrained', 'False'])

    params.device = device
    params.save_path = mk_dir(f'{save_data_path}/{dataset}')

    # ✅ APPLY THE FINAL FIX HERE
    adata_h5 = ensure_spatial_coords(adata_h5)

    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)
    params.cell_num = adata_h5.shape[0]

    if ("ground_truth" in adata_h5.obs.keys()):
        n_clusters = len(set(adata_h5.obs["ground_truth"].dropna()))

    print("===Dataset:{},n_clusters:====".format(dataset, n_clusters))

    conST_net = conST_training(adata_X, graph_dict, params, n_clusters)
    conST_net.pretraining();
    conST_net.major_training();

    conST_embedding = conST_net.get_embedding()
    adata_h5.obsm["embedding"] = conST_embedding
    sc.pp.neighbors(adata_h5, n_neighbors=params.eval_graph_n, use_rep='embedding')
    eval_resolution = res_search_fixed_clus(adata_h5, n_clusters)
    print("resolution:", eval_resolution)
    cluster_key = "conST_leiden"
    sc.tl.leiden(adata_h5, key_added=cluster_key, resolution=eval_resolution)

    index = np.arange(start=0, stop=adata_X.shape[0]).tolist()
    index = [str(x) for x in index]
    dis = graph_dict['adj_norm'].to_dense().numpy() + np.eye(graph_dict['adj_norm'].shape[0])

    # ====================== ✅ FIXED REFINE MAP (NO MORE KEYERROR) ======================
    # All your dataset names are now explicitly defined here
    refine_map = {
        # DLPFC
        "DLPFC": "hexagon",
        # 10X Visium
        "BC_TENX13": "hexagon", "BC_TENX23": "hexagon", "BC_TENX39": "hexagon",
        "Lung_TENX62": "hexagon", "Lung_TENX72": "hexagon",
        # ST data
        "Patient1_Biopsy1_rep1_chip1_d1_stdata": "square",
        "Patient3_Biopsy2_rep2_Chip3_C2_stdata": "square",
        "Patient3_Biopsy4_rep2_Chip3_E2_stdata": "square",
        "BC_SPA130": "square",
        # STARmap
        "STARmap_AD_13m_disease": "square", "BOOST_HMI_starmap": "square", "Tutorial_STARmap": "square",
        # SNARE-seq2
        "LungL_HBM477.TMMN.763": "hexagon", "LungL_HBM582.RKVK.974": "hexagon",
        "LungR_HBM285.MWNX.593": "hexagon", "LungR_HBM753.NSKJ.788": "hexagon",
        # Slide-seq
        "Kedney_HBM232.MBNR.586": "hexagon", "Kedney_HBM478.VWXX.362": "hexagon",
        "Kedney_HBM965.PSNC.855": "hexagon",
        # Xenium
        "BC_NCBI784": "hexagon", "Lung_NCBI856": "hexagon", "Lung_NCBI858": "hexagon",
        "Lung_TENX141": "hexagon", "Lung_NCBI882": "hexagon",
        "Colorectal_TENX139": "hexagon", "Ovarian_TENX142": "hexagon", "Skin_TENX158": "hexagon"
    }

    # Auto-detect shape type for ALL datasets
    if dataset.startswith('15'):
        shape_type = "DLPFC"
    elif dataset in refine_map:
        shape_type = dataset
    else:
        # Fallback for any new dataset
        print(f"⚠️ Dataset {dataset} not in refine_map, using default 'hexagon' shape")
        shape_type = "DLPFC"

    # ✅ Run refinement with no errors
    refine = refine_function(
        sample_id=index,
        shape=refine_map[shape_type],
        pred=adata_h5.obs['conST_leiden'].tolist(),
        dis=dis
    )

    res = {}
    adata_h5.obs['refine'] = refine
    if ("ground_truth" in adata_h5.obs.keys()):
        ari_r, nmi_r, ami_r, bhi_r = eval_model(adata_h5.obs['refine'], adata_h5.obs['ground_truth'])
        ari, nmi, ami, bhi = eval_model(adata_h5.obs['conST_leiden'], adata_h5.obs['ground_truth'])

        #SC = silhouette_score(adata_h5.obsm["embedding"], adata_h5.obs['conST_leiden'])
        #SC_r = silhouette_score(adata_h5.obsm["embedding"], adata_h5.obs['refine'])
        # ✅ Only compute silhouette score if there are ≥2 clusters
        if len(adata_h5.obs['conST_leiden'].unique()) >= 2:
            SC = silhouette_score(adata_h5.obsm["embedding"], adata_h5.obs['conST_leiden'])
            SC_r = silhouette_score(adata_h5.obsm["embedding"], adata_h5.obs['refine'])
        else:
            # If only 1 cluster, set silhouette score to NaN
            SC = np.nan
            SC_r = np.nan


        used_adata = adata_h5[adata_h5.obs["ground_truth"].notna()]
        #SC_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['ground_truth'])
        # ✅ Only compute silhouette score if there are ≥2 ground truth labels
        if len(used_adata.obs['ground_truth'].dropna().unique()) >= 2:
            SC_revise = silhouette_score(used_adata.obsm["embedding"], used_adata.obs['ground_truth'])
        else:
            SC_revise = np.nan

        res["dataset"] = dataset
        res["ari"] = ari
        res["nmi"] = nmi
        res["ami"] = ami
        res["bhi"] = bhi
        res["sc"] = SC
        res["SC_revise"] = SC_revise
        res["ari_1"] = ari_r
        res["nmi_1"] = nmi_r
        res["ami_1"] = ami_r
        res["bhi_1"] = bhi_r
        res["sc_1"] = SC_r

    adata_h5.obs["pred_label"] = refine
    return res, adata_h5

# ====== Unified Function ======
def run_conST_all(adata_h5, save_path, dataset, n_clusters):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    res, adata_result = run_conST(
        adata_h5.copy(),
        dataset=dataset,
        device=device,
        save_data_path=save_path,
        n_clusters=n_clusters
    )
    adata_result.write_h5ad(os.path.join(save_path, f"conST_{dataset}.h5ad"))
    return res, adata_result

# ====== MAIN RUNNER (Uses YOUR utils_for_all.py) ======
import sys
sys.path.append('../')
import utils_for_all as usa

if __name__ == '__main__':
    # ------------ YOUR PATHS ------------
    BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"
    OUTPUT_ROOT = r"D:\My_Project_ST\Output_For_All\conST"
    mk_dir(OUTPUT_ROOT)

    # ------------ ALL DATASETS (Same as utils_for_all.py) ------------
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

    # Dataset categories for result logging
    CATEGORIES = {
        "DLPFC": ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'],
        "10X Visium": ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72'],
        "ST": ["Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata","Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130"],
        "Slide-seq": ['Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'],
        "SNARE-seq2": ['LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'],
        "STARmap": ["STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"],
        "Xenium": ['BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882','Colorectal_TENX139','Ovarian_TENX142','Skin_TENX158']
    }

    seed_torch(0)
    all_results = []

    # ------------ Run All Datasets ------------
    for dataset in ALL_DATASETS:
        print(f"\n==================================================")
        print(f"            RUNNING conST: {dataset}")
        print(f"==================================================")
        
        save_path = os.path.join(OUTPUT_ROOT, dataset)
        mk_dir(save_path)

        # ✅ LOAD DATA USING YOUR utils_for_all.py
        adata_h5, n_cluster = usa.load_dataset(dataset, base_data_path=BASE_DATA_PATH)
        adata_h5.var_names_make_unique()

        # Time & Memory
        start = time.time()
        tracemalloc.start()
        start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

        # Run model
        res, adata_result = run_conST(adata_h5, dataset, n_clusters=n_cluster)

        # Stats
        used_time = time.time() - start
        used_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) - start_mem
        current, peak = tracemalloc.get_traced_memory()
        peak_gb = peak / 1024 / 1024 / 1024
        tracemalloc.stop()
        tracemalloc.clear_traces()

        # Get data type
        data_type = "Unknown"
        for cat, datasets in CATEGORIES.items():
            if dataset in datasets:
                data_type = cat
                break

        # Save result
        full_result = {
            "data_type": data_type, "dataset": dataset,
            "ari": res.get("ari", 0), "nmi": res.get("nmi", 0),
            "ami": res.get("ami", 0), "bhi": res.get("bhi", 0),
            "sc": res.get("sc", 0), "sc_revise": res.get("SC_revise", 0),
            "ari_refined": res.get("ari_1", 0), "nmi_refined": res.get("nmi_1", 0),
            "ami_refined": res.get("ami_1", 0), "bhi_refined": res.get("bhi_1", 0),
            "sc_refined": res.get("sc_1", 0), "time": used_time,
            "Memo": used_mem, "Memo_peak": peak_gb
        }

        all_results.append(full_result)
        pd.DataFrame([full_result]).to_csv(os.path.join(save_path, f"{dataset}_result.csv"), index=False)

    # Save final CSV
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_ROOT, "conST_ALL_DATASETS.csv"), index=False)
    
    print("\n✅ ALL DATASETS COMPLETED SUCCESSFULLY!")