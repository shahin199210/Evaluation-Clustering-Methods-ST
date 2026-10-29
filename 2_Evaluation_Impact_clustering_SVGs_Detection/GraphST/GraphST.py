import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

# Import required layers
from layers import GraphConvolution

# --------------------------
# Low-level GCN Model (Dynamic)
# --------------------------
class GraphST_GCN(nn.Module):
    def __init__(self, nfeat, nhid1=128, nhid2=64, dropout=0.1):
        super(GraphST_GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        # Decoder: Reconstructs from the embedding (nhid2) back to input features (nfeat)
        self.decoder = nn.Linear(nhid2, nfeat) 
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc2(x, adj))
        emb = x
        # Reconstruct input
        output = self.decoder(emb) 
        return output, emb

# --------------------------
# High-level GraphST Wrapper (Fixed for your script)
# --------------------------
class GraphST:
    def __init__(self, adata, device, epochs=600, random_seed=42):
        self.adata = adata
        self.device = device
        self.epochs = epochs
        self.random_seed = random_seed
        
        # Reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        # Prepare data and graph
        self._prepare_data()
        self._build_graph()
        
        # Initialize model with DYNAMIC input size
        nfeat = self.x.shape[1] # Number of genes/features
        self.model = GraphST_GCN(nfeat=nfeat, nhid1=128, nhid2=64, dropout=0.1).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _prepare_data(self):
        """Prepare input features from adata"""
        if sp.issparse(self.adata.X):
            self.x = torch.FloatTensor(self.adata.X.toarray()).to(self.device)
        else:
            self.x = torch.FloatTensor(self.adata.X).to(self.device)

    def _build_graph(self):
        """Build k-nearest neighbor graph from spatial coordinates"""
        coords = self.adata.obsm['spatial']
        adj = kneighbors_graph(coords, n_neighbors=10, mode='connectivity', include_self=False)
        adj = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = adj.tocoo()
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        self.adj = torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(values),
            torch.Size(adj.shape)
        ).to(self.device)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def train(self):
        """Full training loop"""
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output, emb = self.model(self.x, self.adj)
            
            # 🔥 FIX: Ensure output and input shapes match
            # The decoder now maps from embedding (64) back to feature count (33538)
            loss = nn.functional.mse_loss(output, self.x)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        
        # Get final embeddings
        self.model.eval()
        with torch.no_grad():
            _, emb = self.model(self.x, self.adj)
            self.adata.obsm['emb'] = emb.cpu().numpy()
        
        return self.adata