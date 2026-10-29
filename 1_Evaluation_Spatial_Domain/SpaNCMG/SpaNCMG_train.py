import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

class SpaNCMG_model(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def SpaNCMG_train(adata, n_clusters, device, epochs=200, lr=1e-3):
    """
    Train a simplified SpaNCMG model
    """
    adata = adata.copy()
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=torch.float32).to(device)
    
    model = SpaNCMG_model(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_recon, z = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()

    # Get embedding
    model.eval()
    with torch.no_grad():
        _, z = model(X)
    embedding = z.cpu().numpy()

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(embedding)

    # Save to adata
    adata.obsm["SpaNCMG_embedding"] = embedding
    adata.obs["SpaNCMG_cluster"] = clusters.astype(str)

    return adata