# Add THESE 2 LINES FIRST, before all other imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

import pandas as pd
import scanpy as sc
import numpy as np
import stlearn as st
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from h5py import Dataset, Group

def read_h5(f, i=0):
    for k in f.keys():
        if isinstance(f[k], Group):
            print('Group', f[k])
            print('-'*(10-5*i))
            read_h5(f[k], i=i+1)
            print('-'*(10-5*i))
        elif isinstance(f[k], Dataset):
            print('Dataset', f[k])
            print(f[k][()])
        else:
            print('Name', f[k].name)

def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X

def get_adj(generated_data_fold, args):
    import numpy as np
    from scipy import sparse
    import pickle
    from sklearn.metrics.pairwise import euclidean_distances

    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)

    cell_num = len(coordinates)
    distance_list = []
    print ('Calculating distance matrix...')

    for j in range(cell_num):
        for i in range(cell_num):
            if i != j:
                distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))

    distance_array = np.array(distance_list)

    if args.data_name in ['SeqFish','SlideV2_mouse_embryo_E8.5']:
        r=0.1
    elif args.data_name in ['Stereo']:
        r=50
    else:
        r=150

    for threshold in [r]:
        num_big = np.where(distance_array < threshold)[0].shape[0]
        print(f"Radius: {threshold} | Avg neighbors: {num_big/(cell_num*2)}")

        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)

        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1

        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I)
        distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)

        with open(generated_data_fold + 'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_threshold_I_N_crs, fp)

def get_type(args, cell_types, generated_data_fold):
    types_dic = []
    types_idx = []
    for t in cell_types:
        if not t in types_dic:
            types_dic.append(t)
        id = types_dic.index(t)
        types_idx.append(id)

    n_types = max(types_idx) + 1

    if args.data_name == 'Breast_cancer':
        types_dic_sorted = ['Healthy_1', 'Healthy_2', 'Tumor_edge_1', 'Tumor_edge_2',
            'Tumor_edge_3', 'Tumor_edge_4', 'Tumor_edge_5', 'Tumor_edge_6',
            'DCIS/LCIS_1', 'DCIS/LCIS_2', 'DCIS/LCIS_3', 'DCIS/LCIS_4', 'DCIS/LCIS_5',
            'IDC_1', 'IDC_2', 'IDC_3', 'IDC_4', 'IDC_5', 'IDC_6', 'IDC_7']
        relabel_map = {i: types_dic_sorted.index(types_dic[i]) for i in range(n_types)}
        cell_types_relabel = [relabel_map[old] for old in types_idx]

        np.save(generated_data_fold+'cell_types.npy', np.array(cell_types_relabel))
        np.savetxt(generated_data_fold+'types_dic.txt', np.array(types_dic_sorted), fmt='%s', delimiter='\t')
    else:
        np.save(generated_data_fold+'cell_types.npy', np.array(types_idx))
        np.savetxt(generated_data_fold+'types_dic.txt', np.array(types_dic), fmt='%s', delimiter='\t')

def draw_map(generated_data_fold):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    cell_types = np.load(generated_data_fold+'cell_types.npy', allow_pickle=True)
    n_types = max(cell_types) + 1

    types_dic = np.loadtxt(generated_data_fold+'types_dic.txt', dtype='|S15', delimiter='\t').tolist()
    types_dic = [tmp.decode() for tmp in types_dic]

    plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_types, cmap='rainbow')
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title('Annotation')
    plt.savefig(generated_data_fold+'/spacial.png', dpi=400, bbox_inches='tight')
    plt.clf()

# ===================== MAIN FUNCTION =====================
def main(args):
    # ================== DLPFC DATA ==================
    if args.data_name == 'DLPFC':
        proj_list = ['151507']
        for sample_name in proj_list:
            print(f"\n===== Processing DLPFC : {sample_name} =====")

            # Input path
            sample_full_path = os.path.join(args.data_path, "DLPFC", sample_name)
            # OUTPUT PATH
            generated_data_fold = os.path.join(args.generated_data_path, "DLPFC", sample_name) + "\\"

            os.makedirs(generated_data_fold, exist_ok=True)

            # Read 10X data
            adata_h5 = sc.read_10x_h5(os.path.join(sample_full_path, "filtered_feature_bc_matrix.h5"))

            # Read spatial positions
            spatial_path = os.path.join(sample_full_path, "spatial", "tissue_positions_list.csv")
            spatial_df = pd.read_csv(spatial_path, header=None,
                names=["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
            spatial_df = spatial_df.set_index("barcode")
            adata_h5 = adata_h5[adata_h5.obs_names.isin(spatial_df.index)]
            spatial_df = spatial_df.loc[adata_h5.obs_names]
            adata_h5.obsm["spatial"] = spatial_df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values

            # Save features & coordinates
            features = adata_preprocess(adata_h5, min_cells=args.min_cells, pca_n_comps=args.Dim_PCA)
            coordinates = adata_h5.obsm['spatial']
            np.save(generated_data_fold + 'features.npy', features)
            np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))

            # Load ground truth labels
            df_meta = pd.read_csv(os.path.join(sample_full_path, 'metadata.tsv'), sep='\t')
            groud_df = df_meta[~pd.isnull(df_meta['layer_guess'])]
            cell_types = groud_df['layer_guess']

            # Generate graph and labels
            get_adj(generated_data_fold, args)
            get_type(args, cell_types, generated_data_fold)

    # ================== 10X Visium DATA ==================
    if args.data_name in ['BC_TENX13', 'BC_TENX23', 'BC_TENX39', 'Lung_TENX62', 'Lung_TENX72']:
        print(f"\n===== Processing 10X Visium : {args.data_name} =====")

        # Input path
        sample_full_path = os.path.join(args.data_path, "10X_Visium", args.data_name)
        # OUTPUT PATH
        generated_data_fold = os.path.join(args.generated_data_path, "10X_Visium", args.data_name) + "\\"

        os.makedirs(generated_data_fold, exist_ok=True)

        # Read h5ad file
        h5ad_files = [f for f in os.listdir(sample_full_path) if f.endswith('.h5ad')]
        adata_h5 = sc.read_h5ad(os.path.join(sample_full_path, h5ad_files[0]))

        # Save features & coordinates
        features = adata_preprocess(adata_h5, min_cells=args.min_cells, pca_n_comps=args.Dim_PCA)
        coordinates = adata_h5.obsm['spatial']
        np.save(generated_data_fold + 'features.npy', features)
        np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))

        # Get labels
        if 'ground_truth' in adata_h5.obs.columns:
            cell_types = adata_h5.obs['ground_truth']
        elif 'label' in adata_h5.obs.columns:
            cell_types = adata_h5.obs['label']
        elif 'annotation' in adata_h5.obs.columns:
            cell_types = adata_h5.obs['annotation']
        else:
            print("❌ No labels found!")
            return

        get_adj(generated_data_fold, args)
        get_type(args, cell_types, generated_data_fold)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_cells', type=float, default=5)
    parser.add_argument('--Dim_PCA', type=int, default=200)

    # Fixed paths (using double backslashes)
    parser.add_argument('--data_path', type=str, default=r'D:\\My_Project_ST\\DataSets\\')
    parser.add_argument('--generated_data_path',
        default=r'D:\\My_Project_ST\\DataSets\\Generate_data\\generate_CCST\\')

    parser.add_argument('--data_name', type=str, default='DLPFC')

    args = parser.parse_args()
    main(args)