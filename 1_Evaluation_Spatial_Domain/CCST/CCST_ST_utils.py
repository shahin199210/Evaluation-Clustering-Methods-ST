import os
import sys
import matplotlib
import networkx as nx
import scipy.sparse as sp

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import distance
from sklearn import metrics
from scipy.interpolate import interp1d as interp
from sklearn.metrics import roc_curve, auc, roc_auc_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_samples

import numpy as np
from scipy import sparse
import pickle
import pandas as pd
import scanpy as sc
import anndata as ad
import squidpy as sq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader

from CCST import get_graph, train_DGI, PCA_process, Kmeans_cluster

# ==================== PATH FIX FOR WINDOWS ====================
CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(CODE_ROOT)

##### Load raw spatial transcriptomics data
def read_data(dataset, data_path='../../Dataset/', sample_name='1'):
    if dataset == "STARmap":
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + "/STARmap_20180505_BY3_1k.h5ad")
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/Annotation_STARmap_20180505_BY3_1k.h5ad.txt', sep='\t', index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'Annotation'].values

    if dataset == "Mouse_brain":
        adata = sq.datasets.visium_hne_adata()
        adata.var_names_make_unique()
        adata.obs['ground_truth'] = adata.obs["cluster"]

    if dataset == "DLPFC":
        data_name = sample_name
        adata = load_ST_file(f'{data_path}/{dataset}/{data_name}/', count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        Ann_df = pd.read_csv(f'{data_path}/{dataset}/{data_name}/metadata.tsv', sep='\t')
        Ann_df['Ground Truth'] = Ann_df['layer_guess']
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    if dataset == "Breast_cancer":
        file_fold = data_path + str(dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        df_meta = pd.read_table(file_fold + '/metadata.tsv', sep='\t', index_col=0)
        adata.obs['ground_truth'] = df_meta.loc[adata.obs_names, 'annot_type'].values

    if dataset == "Mouse_hippocampus":
        adata = sq.datasets.slideseqv2()
        adata.var_names_make_unique()

    if dataset in ["Mouse_olfactory", "Mouse_brain_section_2", "Mouse_brain_section_1"]:
        file_fold = data_path + str(dataset)
        adata = sc.read(file_fold + '/filtered_feature_bc_matrix.h5ad')
        adata.var_names_make_unique()
    return adata

def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if not load_images:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode', 'in_tissue', 'array_row', 'array_col',
            'pxl_col_in_fullres', 'pxl_row_in_fullres'
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print(f'adata: ({adata_h5.shape[0]}, {adata_h5.shape[1]})')
    return adata_h5

# ==================== FULLY FIXED get_data() FUNCTION ====================
# SUPPORTS BOTH RAW DLPFC (10X) + 10X Visium (.h5ad) – NO MISSING FILES ERROR
def get_data(args):
    data_dir = args.data_path
    dataset_name = args.data_name
    print(f"Loading data from: {data_dir}")

    # --------------------------
    # Option 1: Load .h5ad (10X Visium: BC_TENX13, Lung_TENX62, etc.)
    # --------------------------
    h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
    if h5ad_files:
        h5ad_path = os.path.join(data_dir, h5ad_files[0])
        print(f" Found 10X Visium .h5ad file: {h5ad_files[0]}")
        adata = sc.read_h5ad(h5ad_path)
        adata.var_names_make_unique()

        # Get labels from .h5ad obs
        if 'ground_truth' in adata.obs.columns:
            cell_type_indeces = adata.obs['ground_truth'].values
        elif 'label' in adata.obs.columns:
            cell_type_indeces = adata.obs['label'].values
        elif 'annotation' in adata.obs.columns:
            cell_type_indeces = adata.obs['annotation'].values
        else:
            cell_type_indeces = np.array(['unknown'] * adata.n_obs)

    # --------------------------
    # Option 2: Load raw DLPFC (10X Visium folder structure)
    # --------------------------
    else:
        print(f" Loading raw DLPFC 10X Visium data")
        adata = load_ST_file(data_dir, count_file='filtered_feature_bc_matrix.h5')
        # Load DLPFC labels from metadata.tsv
        meta_path = os.path.join(data_dir, "metadata.tsv")
        meta = pd.read_csv(meta_path, sep="\t")
        cell_type_indeces = meta["layer_guess"].values

    # --------------------------
    # Standard preprocessing
    # --------------------------
    X_data = adata_preprocess(adata, min_cells=5, pca_n_comps=200)

    # Build spatial graph
    graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], args)
    adj_0 = graph_dict["adj_org"]
    adj_norm = graph_dict["adj_norm"]

    # Lambda_I integration
    num_points = X_data.shape[0]
    adj_I = sparse.csr_matrix(np.eye(num_points))
    adj = (1 - args.lambda_I) * adj_norm + args.lambda_I * adj_I

    print(f"Data loaded: {num_points} spots | Feature dim: {X_data.shape[1]}")
    return adj_0, adj, X_data, cell_type_indeces

def get_process_data(args):
    if args.data_name == "Breast_cancer":
        data_file = args.data_path + args.data_name + '/'
        with open(data_file + 'Adjacent', 'rb') as fp:
            adj_0 = pickle.load(fp)
        X_data = np.load(data_file + 'features.npy')
        num_points = X_data.shape[0]
        adj_I = np.eye(num_points)
        adj_I = sparse.csr_matrix(adj_I)
        adj = (1 - args.lambda_I) * adj_0 + args.lambda_I * adj_I
        cell_type_indeces = np.load(data_file + 'cell_types.npy', allow_pickle=True)

    if args.data_name == "DLPFC":
        data_path = args.data_path
        data_name = args.sample_name
        adata = load_ST_file(f'{data_path}/{data_name}/', count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        Ann_df = pd.read_csv(f'{data_path}/{data_name}/metadata.tsv', sep='\t')
        adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'layer_guess']
        X_data = adata_preprocess(adata, min_cells=5, pca_n_comps=200)
        graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], args)
        adj_0 = graph_dict["adj_org"]
        adj = graph_dict["adj_norm"]
        cell_type_indeces = Ann_df['layer_guess'].values

    return adj_0, adj, X_data, cell_type_indeces

def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X

def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []
    return graphdict

def graph_computing(adj_coo, cell_num, params):
    edgeList = []
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        res = distMat.argsort()[:params.k + 1]
        tmpdist = distMat[0, res[0][1:params.k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, params.k + 1):
            weight = 1.0 if distMat[0, res[0][j]] <= boundary else 0.0
            edgeList.append((node_idx, res[0][j], weight))
    return edgeList

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

def graph_construction(adj_coo, cell_N, args):
    params = args
    adata_Adj = graph_computing(adj_coo, cell_N, params)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }
    return graph_dict

def clean_labels(gt_labels, cluster_labels, NAN_idx):
    cleaned_gt_labels, cleaned_cluster_labels = [], []
    for i, tmp in enumerate(gt_labels):
        if tmp != NAN_idx:
            cleaned_gt_labels.append(tmp)
            cleaned_cluster_labels.append(cluster_labels[i])
    print('cleaned length', len(cleaned_gt_labels), len(cleaned_cluster_labels))
    return np.array(cleaned_gt_labels), np.array(cleaned_cluster_labels)

def compare_labels(save_path, gt_labels, cluster_labels):
    if max(gt_labels) == max(cluster_labels):
        matrix = np.zeros([max(gt_labels) + 1, max(cluster_labels) + 1], dtype=int)
        n_samples = len(cluster_labels)
        for i in range(n_samples):
            matrix[gt_labels[i], cluster_labels[i]] += 1
        matrix_size = max(gt_labels) + 1
        order_seq = np.arange(matrix_size)
        matrix = np.array(matrix)
        norm_matrix = matrix / matrix.sum(1).reshape(-1, 1)
        norm_matrix_2_arr = norm_matrix.flatten()
        sort_index = np.argsort(-norm_matrix_2_arr)
        sort_row, sort_col = [], []
        for tmp in sort_index:
            sort_row.append(int(tmp / matrix_size))
            sort_col.append(int(tmp % matrix_size))
        sort_row = np.array(sort_row)
        sort_col = np.array(sort_col)
        done_list = []
        for j in range(len(sort_index)):
            if len(done_list) == matrix_size:
                break
            if (sort_row[j] in done_list) or (sort_col[j] in done_list):
                continue
            done_list.append(sort_row[j])
            tmp = sort_col[j]
            sort_col[sort_col == tmp] = -1
            sort_col[sort_col == sort_row[j]] = tmp
            sort_col[sort_col == -1] = sort_row[j]
            order_seq[sort_row[j]], order_seq[tmp] = order_seq[tmp], order_seq[sort_row[j]]

        reorder_cluster_labels = []
        for k in cluster_labels:
            reorder_cluster_labels.append(order_seq.tolist().index(k))
        matrix = matrix[:, order_seq]
        norm_matrix = norm_matrix[:, order_seq]
        plt.imshow(norm_matrix)
        plt.savefig(save_path + '/compare_labels_Matrix.png')
        plt.close()
        np.savetxt(save_path + '/compare_labels_Matrix.txt', matrix, fmt='%3d', delimiter='\t')
        reorder_cluster_labels = np.array(reorder_cluster_labels, dtype=int)
    else:
        print('not square matrix!!')
        reorder_cluster_labels = cluster_labels
    return reorder_cluster_labels

def draw_map(args, adj_0, barplot=False):
    data_folder = args.data_path
    save_path = args.result_path
    f = open(args.data_path + 'type.txt')
    line = f.readline()
    cell_cluster_type_list = []

    while line:
        tmp = line.split('\t')
        cell_id = int(float(tmp[0]))
        cell_cluster_type = int(float(tmp[2]))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline()
    f.close()

    n_clusters = max(cell_cluster_type_list) + 1
    print('n clusters in drawing:', n_clusters)
    coordinates = np.load(data_folder + 'coordinates.npy')
    plt.scatter(x=coordinates[:, 0], y=-coordinates[:, 1], s=5, c=cell_cluster_type_list, cmap='rainbow')
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title('CCST')
    plt.savefig(save_path + '/spacial.png', dpi=400, bbox_inches='tight')
    plt.clf()

    if barplot:
        total_cell_num = len(cell_cluster_type_list)
        barplot = np.zeros([n_clusters, n_clusters], dtype=int)
        source_cluster_type_count = np.zeros(n_clusters, dtype=int)
        p1, p2 = adj_0.nonzero()

        def get_all_index(lst=None, item=''):
            return [i for i in range(len(lst)) if lst[i] == item]

        for i in range(total_cell_num):
            source_cluster_type_index = cell_cluster_type_list[i]
            edge_indeces = get_all_index(p1, item=i)
            paired_vertices = p2[edge_indeces]
            for j in paired_vertices:
                neighbor_type_index = cell_cluster_type_list[j]
                barplot[source_cluster_type_index, neighbor_type_index] += 1
                source_cluster_type_count[source_cluster_type_index] += 1

        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot.txt', barplot, fmt='%3d', delimiter='\t')
        norm_barplot = barplot / (source_cluster_type_count.reshape(-1, 1))
        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot_normalize.txt', norm_barplot, fmt='%3f', delimiter='\t')

        for clusters_i in range(n_clusters):
            plt.bar(range(n_clusters), norm_barplot[clusters_i], label='graph ' + str(clusters_i))
            plt.xlabel('cell type index')
            plt.ylabel('value')
            plt.title('barplot_' + str(clusters_i))
            plt.savefig(save_path + '/barplot_sub' + str(clusters_i) + '.jpg')
            plt.clf()
    return

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

def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.02):
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == fixed_clus_count:
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
    return res

def CCST_on_ST(args):
    lambda_I = args.lambda_I
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------
    # Use fixed get_data()
    # --------------------------
    adj_0, adj, X_data, cell_type_indeces = get_data(args)

    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj shape:', adj.shape, 'Edges:', len(adj.data))
    print('Feature matrix shape:', X_data.shape)
    n_clusters = args.n_clusters

    if args.DGI and (lambda_I >= 0):
        print("----------- Deep Graph Infomax -------------")
        data_list = get_graph(adj, X_data)
        data_loader = DataLoader(data_list, batch_size=batch_size)
        DGI_model = train_DGI(args, data_loader=data_loader, in_channels=num_feature)

        for data in data_loader:
            data.to(device)
            X_embedding, _, _ = DGI_model(data)
            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_filename = args.embedding_data_path + 'lambdaI' + str(lambda_I) + '_epoch' + str(
                args.num_epoch) + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)

    if args.cluster:
        cluster_type = 'kmeans'
        print("----------- Clustering -------------")
        X_embedding_filename = args.embedding_data_path + 'lambdaI' + str(lambda_I) + '_epoch' + str(
            args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)
        adata = ad.AnnData(X_embedding)

        if cluster_type == 'kmeans':
            X_embedding = PCA_process(X_embedding, nps=30)

            print('Shape of data for clustering:', X_embedding.shape)

            cluster_labels, score = Kmeans_cluster(X_embedding, args.n_clusters)
            SC_Score = silhouette_score(X_embedding, cluster_labels)
            print("Silhouette Score (SC):", SC_Score)

            ASW = np.mean(silhouette_samples(X_embedding, cluster_labels))
            print("Average Silhouette Width (ASW):", ASW)

            # Save predictions
            result_df = pd.DataFrame()
            result_df["Ground Truth"] = cell_type_indeces
            result_df['CCST_Kmens_predict'] = cluster_labels
            result_df.to_csv(args.result_path + '/predict_types.csv', index=False)

            # Evaluate
            ari, nmi, ami = eval_model(cell_type_indeces, cluster_labels)
            print("******************************** Results **********************")
            print("ARI, NMI, AMI, SC, ASW:", ari, nmi, ami, SC_Score, ASW)

        else:
            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=20, n_pcs=50)
            eval_resolution = res_search_fixed_clus(cluster_type, adata, n_clusters)
            if cluster_type == 'leiden':
                sc.tl.leiden(adata, key_added="CCST_leiden", resolution=eval_resolution)
                cluster_labels = np.array(adata.obs['leiden'])
            if cluster_type == 'louvain':
                sc.tl.louvain(adata, key_added="CCST_louvain", resolution=eval_resolution)
                cluster_labels = np.array(adata.obs['louvain'])

            cluster_labels = [int(x) for x in cluster_labels]
            score = False

        results_file = f'{args.result_path}/CCST_{args.data_name}.h5ad'
        adata.write(results_file)
        return SC_Score

    if args.draw_map:
        print("----------- Drawing spatial plot -------------")
        draw_map(args, adj_0)