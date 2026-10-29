import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import torch
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# -------------------------- CCST Model Definition --------------------------
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

def train_ccst(adata, epochs=200, lr=0.001, hidden_dim=64, num_clusters=10, device='cuda:0'):
    # Preprocess
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=200)
    
    # Train
    x = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)
    model = CCST(input_dim=200, hidden_dims=[256, hidden_dim], num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        feat, recon, prob = model(x)
        recon_loss = mse_loss(recon, x)
        cluster_loss = -torch.mean(torch.sum(prob * torch.log(prob + 1e-10), dim=1))
        loss = recon_loss + 0.1 * cluster_loss
        
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

# -------------------------- Fixed Cluster Search --------------------------
def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

# -------------------------- ONLY FOR DLPFC 151507 + CLEAN UMAP --------------------------
def run_CCST_151507_with_Clean_UMAP(adata_h5, save_path, epochs=200, n_cluster=10):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)
    os.makedirs(save_path, exist_ok=True)
    
    # Train CCST
    adata_ccst = train_ccst(
        adata_h5.copy(),
        epochs=epochs,
        num_clusters=n_cluster,
        device=device
    )
    
    # Copy metadata
    adata_ccst.obs['ground_truth'] = adata_h5.obs['ground_truth'].values
    
    # Compute UMAP from CCST embeddings
    sc.pp.neighbors(adata_ccst, use_rep='CCST_emb', n_neighbors=20, random_state=0)
    sc.tl.umap(adata_ccst, random_state=0)
    
    # Leiden clustering (only using ARI-tuned resolution)
    eval_resolution = res_search_fixed_clus(adata_ccst, n_cluster)
    sc.tl.leiden(adata_ccst, key_added="CCST_leiden", resolution=eval_resolution, random_state=0)
    
    # -------------------------- CLEAN UMAP PLOTS --------------------------
    # Set plot style for clarity
    plt.rcParams.update({
        'figure.figsize': (6, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    })
    
    # 1. UMAP colored by PREDICTED CLUSTERS (CCST_leiden)
    fig, ax = plt.subplots()
    sc.pl.umap(
        adata_ccst,
        color='CCST_leiden',
        title='CCST UMAP | Predicted Clusters (151507)',
        palette='tab20',
        show=False,
        ax=ax
    )
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "CCST_UMAP_Predicted_Clusters.pdf"))
    plt.close()
    
    # 2. UMAP colored by GROUND TRUTH (for comparison)
    fig, ax = plt.subplots()
    sc.pl.umap(
        adata_ccst,
        color='ground_truth',
        title='CCST UMAP | Ground Truth (151507)',
        palette='tab20',
        show=False,
        ax=ax
    )
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "CCST_UMAP_Ground_Truth.pdf"))
    plt.close()
    
    print(f"\n✅ Clean UMAP plots saved to: {save_path}")
    return adata_ccst

# -------------------------- MAIN --------------------------
if __name__ == '__main__':
    sys.path.append('../')
    import utils_for_all as usa
    warnings.filterwarnings('ignore')

    # Only run on 151507
    dataset = "151507"
    output_root = "D:/My_Project_ST/Output/CCST/"
    save_path = os.path.join(output_root, dataset)
    
    # Load data
    adata_h5, n_cluster = usa.get_adata(dataset)
    adata_h5.var_names_make_unique()
    
    # Run CCST + Generate clean UMAPs
    adata_result = run_CCST_151507_with_Clean_UMAP(
        adata_h5,
        save_path=save_path,
        epochs=200,
        n_cluster=n_cluster
    )
    
    # Save final AnnData
    adata_result.write_h5ad(os.path.join(save_path, "151507_CCST_with_UMAP.h5ad"))