import scanpy as sc
import numpy as np

def preprocess_data(adata):
    """
    Standard preprocessing for spatial transcriptomics (SpaNCMG-style)
    """
    adata = adata.copy()
    
    # Basic QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Normalize & log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    
    # Scale
    sc.pp.scale(adata, max_value=10)
    
    # PCA for initial embedding
    sc.tl.pca(adata, n_comps=50)
    
    return adata