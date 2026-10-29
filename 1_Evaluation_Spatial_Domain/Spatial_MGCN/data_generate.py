from __future__ import division
from __future__ import print_function

from utils import features_construct_graph, spatial_construct_graph1
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from config import Config
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, radius, data_path, sq=None):

    if dataset.startswith('15'):  # DLPFC dataset
        print("load DLPFC dataset:")
        file_fold = f"{data_path}"
        
        # ✅ FIXED INPUT PATH
        raw = sc.read_visium(path=f'D:/My_Project_ST/DataSets/DLPFC/{dataset}/', count_file='filtered_feature_bc_matrix.h5')
        Ann_df = pd.read_csv(f'D:/My_Project_ST/DataSets/DLPFC/{dataset}/metadata.tsv', sep='\t')
        
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        raw.obs['ground_truth'] = Ann_df.loc[raw.obs_names, 'Ground Truth']
        
        # ✅ FIXED SPATIAL PATH
        spatial = pd.read_csv(f'D:/My_Project_ST/DataSets/DLPFC/{dataset}/spatial/tissue_positions_list.csv',
                              sep=",",
                              header=None,
                              na_filter=False,
                              index_col=0,
                              )
                              
        spatial.columns = ["X0", "X1", "X2", "X3", "X4"]
        raw.obs["x_array"] = spatial.loc[raw.obs_names, "X1"]
        raw.obs["y_array"] = spatial.loc[raw.obs_names, "X2"]
        raw.obs["x_pixel"] = spatial.loc[raw.obs_names, "X3"]
        raw.obs["y_pixel"] = spatial.loc[raw.obs_names, "X4"]
        x_array = raw.obs["x_array"].tolist()
        y_array = raw.obs["y_array"].tolist()

        n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)
        raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        raw.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        raw.obs['ground_truth'] = df_meta.loc[raw.obs_names, 'fine_annot_type'].values
        print("Breast_cancer的类别数", len(raw.obs['ground_truth'].unique()))

        spatial = pd.read_csv(file_fold + '/spatial/tissue_positions_list.csv',
                              sep=",",
                              header=None,
                              na_filter=False,
                              index_col=0,
                              )
        spatial.columns = ["X0", "X1", "X2", "X3", "X4"]
        raw.obs["x_array"] = spatial.loc[raw.obs_names, "X1"]
        raw.obs["y_array"] = spatial.loc[raw.obs_names, "X2"]
        raw.obs["x_pixel"] = spatial.loc[raw.obs_names, "X3"]
        raw.obs["y_pixel"] = spatial.loc[raw.obs_names, "X4"]
        x_array = raw.obs["x_array"].tolist()
        y_array = raw.obs["y_array"].tolist()
        n_clusters = 20

    if dataset == "Mouse_brain":
        raw = sq.datasets.visium_hne_adata()
        raw.var_names_make_unique()
        raw.obs['ground_truth'] = raw.obs["cluster"]
        print(f"Mouse_brain数据大小：{raw.shape}，类别数：{len(raw.obs['ground_truth'].unique())}")

        x_array = raw.obs['array_row']
        y_array = raw.obs['array_col']
        raw.obs["x_array"] = x_array
        raw.obs["y_array"] = y_array
        n_clusters = 15

    if dataset == "PDAC":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/PDAC_raw_428_19736.h5ad")
        raw.obs['ground_truth'] = raw.obs['Ground Truth']
        x_array = raw.obs['x_array']
        y_array = raw.obs['y_array']
        n_clusters = 4

    if dataset == 'Stereo':
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + '/Adult_stereo.h5ad')
        raw.obs["ground_truth"] = raw.obs['Annotation']
        n_clusters = 16
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == 'SeqFish':
        raw = sq.datasets.seqfish()
        raw.obs['ground_truth'] = raw.obs['celltype_mapped_refined']
        n_clusters = 22
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        raw = sc.read(file_fold + "/STARmap_1207_1020.h5ad")
        n_clusters = 16
        image_coor = raw.obsm["spatial"]
        x_array = image_coor[:, 0]
        y_array = image_coor[:, 1]
        raw.obs["x_array"] = image_coor[:, 0]
        raw.obs["y_array"] = image_coor[:, 1]

    adata=raw
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    dataset2 = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    Dataset_test = ['151673']

    for dataset in dataset2:
        if dataset.startswith('15'):
            # ✅ FIXED OUTPUT SAVE PATH
            savepath = "D:/My_Project_ST/DataSets/Generate_data/Spatial_MGCN/" + dataset + "/"
            # ✅ FIXED INPUT PATH
            input=f'D:/My_Project_ST/DataSets/DLPFC/{dataset}/'
            config_file = './config/DLPFC.ini'

        else:
            # ✅ FIXED OUTPUT PATH
            savepath = "D:/My_Project_ST/DataSets/Generate_data/Spatial_MGCN/" + dataset + "/"
            input = f'D:/My_Project_ST/DataSets/{dataset}/'
            config_file = './config/' + dataset + '.ini'

        mk_dir(savepath)
        
        import os
        print(f"Current working directory: {os.getcwd()}")
        print(f"Config file path passed: {config_file}")
        print(f"Full absolute path to config: {os.path.abspath(config_file)}")
        print(f"File exists? {os.path.exists(config_file)}")
        
        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius, input)
        print("saving", adata.shape)

        adata.write(f'{savepath}Spatial_MGCN_{dataset}.h5ad')
        print("done!")