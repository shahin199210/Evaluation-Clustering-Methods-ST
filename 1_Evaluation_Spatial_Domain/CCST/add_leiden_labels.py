import scanpy as sc
import os

# List of all your datasets
datasets = [
    ("BC_TENX13", r"D:\My_Project_ST\DataSets\10X_Visium\BC_TENX13\TENX13.h5ad"),
    ("BC_TENX23", r"D:\My_Project_ST\DataSets\10X_Visium\BC_TENX23\TENX23.h5ad"),
    ("BC_TENX39", r"D:\My_Project_ST\DataSets\10X_Visium\BC_TENX39\TENX39.h5ad"),
    ("Lung_TENX62", r"D:\My_Project_ST\DataSets\10X_Visium\Lung_TENX62\TENX62.h5ad"),
    ("Lung_TENX72", r"D:\My_Project_ST\DataSets\10X_Visium\Lung_TENX72\TENX72.h5ad")
]

def add_leiden_labels(h5ad_path, resolution=0.5):
    print(f"\n=== Processing {os.path.basename(h5ad_path)} ===")

    # Load the .h5ad file
    adata = sc.read_h5ad(h5ad_path)

    # Preprocess
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)

    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)

    # Add as 'ground_truth' for your pipeline
    adata.obs['ground_truth'] = adata.obs['leiden']

    # Save as new labeled file
    output_path = h5ad_path.replace(".h5ad", "_with_labels.h5ad")
    adata.write_h5ad(output_path)

    print(f" Saved labeled data to: {output_path}")
    print(f"   Number of clusters found: {len(adata.obs['ground_truth'].unique())}")

# Run for all datasets
for name, path in datasets:
    add_leiden_labels(path)

print("\n All datasets processed. Now run the data generator.")