import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans

# Simplified STFormer (Transformer for Spatial Transcriptomics)
class STFormer_model(nn.Module):
    def __init__(self, input_dim, latent_dim=64, nhead=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        z = self.transformer(x).squeeze(1)
        x_recon = self.decoder(z)
        return x_recon, z

def STFormer_train(adata, n_clusters, device, epochs=200, lr=1e-3):
    adata = adata.copy()
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=torch.float32).to(device)

    model = STFormer_model(input_dim=X.shape[1], latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_recon, z = model(X)
        loss = criterion(x_recon, X)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, z = model(X)
    embedding = z.cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(embedding)

    adata.obsm["STFormer_embedding"] = embedding
    adata.obs["STFormer_cluster"] = clusters.astype(str)
    return adata