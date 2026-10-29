import os
import anndata
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

def mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_adata(dataset):
    # Your existing function to load 12 DLPFC datasets (151507-151676)
    # This is the same function you use in SEDR, no changes needed
    pass  # Replace with your actual implementation

def eval_model(y_pred, y_true):
    # Compute ARI, NMI, AMI (exact same as your SEDR code)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    return ari, nmi, ami