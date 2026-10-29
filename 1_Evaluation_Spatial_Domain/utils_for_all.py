import scanpy as sc
import pandas as pd
import os
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

# ====================== HELPER FUNCTIONS (ONLY FOR CCST) ======================
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
    return ari, nmi, ami

# ====================== CENTRAL DATA LOADER (ALL DATASETS) ======================
def load_dataset(dataset, base_data_path="D:/My_Project_ST/DataSets"):
    # --------------------------
    # Dataset Groups
    # --------------------------
    Dataset_DLPFC = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    Dataset_10X_Visium = ['BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72']
    Dataset_ST = [
        "Patient1_Biopsy1_rep1_chip1_d1_stdata",
        "Patient3_Biopsy2_rep2_Chip3_C2_stdata",
        "Patient3_Biopsy4_rep2_Chip3_E2_stdata",
        "BC_SPA130"
    ]
    Dataset_SNARE_seq2 = [
        'LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788'
    ]
    Dataset_SlideI_seq = [
        'Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855'
    ]
    Dataset_Xenium = [
        'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_NCBI882','Lung_TENX141',
        'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
    ]
    Dataset_STARmap = [
        "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap"
    ]

    # --------------------------
    # Load by Dataset Type
    # --------------------------
    if dataset in Dataset_DLPFC:
        # DLPFC Special Loader
        file_fold = f"{base_data_path}/DLPFC/{dataset}/"
        raw = sc.read_visium(path=file_fold, count_file='filtered_feature_bc_matrix.h5')
        
        spatial = pd.read_csv(file_fold + "spatial/tissue_positions_list.csv",
                              sep=",", header=None, na_filter=False, index_col=0)
        spatial.columns = ["X0", "X1", "X2", "X3", "X4"]
        raw.obs["x_array"] = spatial.loc[raw.obs_names, "X1"]
        raw.obs["y_array"] = spatial.loc[raw.obs_names, "X2"]
        raw.obs["x_pixel"] = spatial.loc[raw.obs_names, "X3"]
        raw.obs["y_pixel"] = spatial.loc[raw.obs_names, "X4"]

        Ann_df = pd.read_csv(f"{file_fold}metadata.tsv", sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        Ann_df = Ann_df.set_index('barcode')
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']
        raw = raw[raw.obs['ground_truth'].notna()].copy()
        
        n_clusters = 5 if dataset in ['151669','151670','151671','151672'] else 7

    elif dataset in Dataset_STARmap:
        # STARmap Loader
        if dataset == "STARmap_AD_13m_disease":
            subfolder = "STARmap_AD"
            labeled_file = "STARmap_AD_13m_disease_rep1_with_labels.h5ad"
        elif dataset == "BOOST_HMI_starmap":
            subfolder = "BOOST_HMI"
            labeled_file = "BOOST_HMI_starmap_final_fixed.h5ad"
        elif dataset == "Tutorial_STARmap":
            subfolder = "20180505_BY3_1k"
            labeled_file = "STARmap_20180505_BY3_1k.h5ad"
        
        full_path = os.path.join(base_data_path, "StarMAP", subfolder, labeled_file)
        raw = sc.read_h5ad(full_path)
        
        if dataset == "Tutorial_STARmap" and "label" in raw.obs.columns:
            raw.obs["ground_truth"] = raw.obs["label"].astype(str)
        n_clusters = 15

    elif dataset in Dataset_SNARE_seq2:
        # SNARE-seq2 Loader
        subdir = "SNARE-seq2"
        labeled_file = f"{dataset.split('_', 1)[1]}_with_labels.h5ad"
        full_path = os.path.join(base_data_path, subdir, dataset, labeled_file)
        raw = sc.read_h5ad(full_path)
        n_clusters = 15

    elif dataset in Dataset_Xenium:
        # Xenium Loader ✅ FULLY FIXED PATH
        subdir = "Xenium"
        if dataset in ["Colorectal_TENX139", "Ovarian_TENX142", "Skin_TENX158"]:
            labeled_file = f"{dataset.split('_')[-1]}_with_labels_downsampled_20k_with_realGT.h5ad"
        else:
            labeled_file = f"{dataset.split('_', 1)[1]}_with_labels.h5ad"
        # ✅ FIXED: Added dataset folder (BC_NCBI784) to path
        full_path = os.path.join(base_data_path, subdir, dataset, labeled_file)
        raw = sc.read_h5ad(full_path)
        n_clusters = 15

    elif dataset in Dataset_ST:
        # ST Loader
        subdir = "ST"
        st_folders = [
            "1_Patient1_Biopsy1_rep1_chip1_d1_stdata",
            "2_Patient3_Biopsy2_rep2_Chip3_C2_stdata",
            "3_Patient3_Biopsy4_rep2_Chip3_E2_stdata",
            "4_BC_SPA130"
        ]
        st_idx = Dataset_ST.index(dataset)
        st_folder = st_folders[st_idx]
        
        if dataset == "BC_SPA130":
            labeled_file = "SPA130_with_labels.h5ad"
        else:
            labeled_file = f"{dataset}.h5ad"
        
        full_path = os.path.join(base_data_path, subdir, st_folder, labeled_file)
        raw = sc.read_h5ad(full_path)
        n_clusters = 15

    elif dataset in Dataset_SlideI_seq:
        # Slide-seq Loader
        subdir = "SlideI_seq"
        labeled_file = f"{dataset.split('_', 1)[1]}_with_labels.h5ad"
        full_path = os.path.join(base_data_path, subdir, dataset, labeled_file)
        raw = sc.read_h5ad(full_path)
        n_clusters = 15

    elif dataset in Dataset_10X_Visium:
        # 10X Visium Loader
        subdir = "10X_Visium"
        labeled_file = f"{dataset.split('_', 1)[1]}_with_labels.h5ad"
        full_path = os.path.join(base_data_path, subdir, dataset, labeled_file)
        raw = sc.read_h5ad(full_path)
        n_clusters = 15

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    raw.var_names_make_unique()
    print(f"Loaded: {dataset} | Shape: {raw.shape}")
    return raw, n_clusters

