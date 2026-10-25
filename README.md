# Evaluation-Clustering-Methods-ST
Evaluation clustering methods for spatial domain identification and SVG detection in spatial transcriptomics


##Overview
In this benchmark study, we evaluate 27 spatial clustering methods for spatial domain identification and investigate how clustering selection shapes downstream spatially variable gene (SVG) discovery. We assemble 39 real spatial transcriptomics datasets spanning six sequencing- and imaging-based platforms, including 17 10X Visium, 4 ST, 3 Slide-seq2, 4 SNARE-seq2, 8 Xenium, and 3 STARmap samples, alongside 12 SRTsim-simulated datasets with known ground-truth SVGs. We categorize these 27 algorithms into baseline, core spatial, and modern spatial groups according to their compatibility with diverse spatial transcriptomics data types. We benchmark each method’s spatial domain segmentation performance across four key criteria: segmentation accuracy, replicate stability, cross-platform generality, and computational scalability, and produce an overall performance ranking. To quantify downstream effects, we use global consensus SVG sets as references and compute the sensitivity and specificity of SVGs recovered from each clustering output. We also characterize clustering-associated variation in SVG spatial patterns, spatial autocorrelation, technical artifact rates, gene overlap, and biological functions of the top SVGs. Using simulated data, we compare SVGs derived from clustering results against ground-truth SVGs and contrast SVG-detection performance between clustering-based pipelines and standalone global SVG detectors. We use AUPR, sensitivity–specificity plots, precision–recall curves, and UpSet plots to quantify detection accuracy and shared SVG composition across approaches.


## Required software and packages

| Package | Version | Notes | Type |
| ---- | ---- | ---- | ---- |
| **Python packages** |  |  |  |
| python | 3.14 | Base interpreter | Interpreter |
| VS Code | 1.141.0 | Code editor | IDE |
| numpy | 2.4.4 | Numerical array operations | Pip-installed |
| pandas | 2.3.3 | Dataframe, saving csv results | Pip-installed |
| torch | 2.12.0+cpu | Model training, tensor operations | Pip-installed |
| scikit-learn | 1.8.0 | KMeans, clustering metrics (NMI, AMI, ARI, silhouette_score) | Pip-installed |
| scanpy | 1.10.3 | sc.AnnData, sc.pp.neighbors, sc.tl.louvain | Pip-installed |
| psutil | 7.2.2 | Memory usage monitoring | Pip-installed |
| torch_geometric | 2.7.0 | PyG, for GATv2Conv and DeepGraphInfomax graph convolution | Pip-installed |
| anndata | 0.12.10 | AnnData object (explicit import) | Pip-installed |
| scipy | 1.17.1 | sparse matrix, spatial KDTree, Laplacian eigen decomposition | Pip-installed |
| matplotlib | 3.10.8 | plotting, plt.switch_backend('Agg') | Pip-installed |
| leidenalg | 0.10.2 | Leiden community detection backend | Pip-installed |
| louvain | 0.16 | Louvain community detection backend | Pip-installed |
| squidpy | 1.8.3 | Spatial transcriptomics analysis | Pip-installed |
| tqdm | 4.67.3 | Progress bar for training loops | Pip-installed |
| seaborn | 0.13.2 | Statistical visualization | Pip-installed |
| SpaceFlow | 1.0.4 | Spatial graph-based deep learning for spatial transcriptomics domain detection | Pip-installed |
| STAGATE_pyG | 1.0.0 | Spatial graph attention auto-encoder for spatial domain identification | Local source |
| GraphST | 1.1.1 | Spatial graph clustering for spatial transcriptomics | Local source |
| IRIS | 3.16.0 | Spatial clustering framework for spatial transcriptomics | Local source |
| SCGDL | NA | Deep graph infomax spatial clustering | Local source |
| SEDR | NA | Spatial embedded deep representation clustering | Local source |
| SpaGCN | 1.2.7 | Spatial graph convolutional network for spatial transcriptomics clustering | Local source |
| SpaNCMG | NA | Spatial neighbourhood contrastive autoencoder for spatial domain clustering | Local source |
| SpaRSE | 0.19.2 | Spatial representation autoencoder for spatial domain clustering | Local source |
| Spatial_MGCN | NA | Multi-view graph convolutional network with ZINB loss for spatial domain segmentation | Local source |
| STAlign | 1.0.1 | Simple PCA-guided autoencoder for spatial embedding + Leiden clustering | Local source |
| STAligner | NA | Autoencoder with LayerNorm, weight decay regularization | Local source |
| stCluster | NA | External spatial clustering pipeline, imported and wrapped in benchmark runner | Local source |
| STFormer | 1.0 | Transformer-based spatial clustering, includes smooth interpolated domain visualization | Local source |
| STMGCN | NA | Multi-view graph convolution, deep clustering with target distribution & KL loss | Local source |
| stRGAT | NA | Multi-head graph attention network, spatial radius graph, MSE reconstruction loss | Local source |
| **R packages** |  |  |  |
| R | 4.5.2 | Base interpreter | Interpreter |
| RStudio | 2026.01.0+392 | IDE only, not required to run script | IDE |
| SpaTalk | 1.0 | Spatial cell communication & spatial clustering | GitHub |
| Seurat | 5.5.1 | Preprocessing, PCA, SCTransform, clustering | CRAN |
| anndata | 0.8.0 | Read h5ad AnnData files | CRAN |
| aricode | 1.0.3 | ARI, NMI, AMI metrics | CRAN |
| tictoc | 1.2.1 | Runtime measurement | CRAN |
| Matrix | 1.7.5 | Sparse matrix operations | CRAN |
| dplyr | 1.2.1 | Data frame manipulation | CRAN |
| ggplot2 | 4.0.3 | Visualization | CRAN |
| cluster | 2.1.8.2 | Silhouette score calculation | CRAN |
| mclust | 6.1.3 | Model-based clustering helper | CRAN |
| clusteval | 0.2.1 | Clustering evaluation | GitHub |
| fpc | 2.2.14 | Clustering utilities | CRAN |
| jsonlite | 2.0.0 | JSON IO | CRAN |
| iSpatial | 1.0.0 | Spatial transcriptomics spatial clustering method | GitHub |
| SpatialExperiment | 1.20.0 | S4 container for spatial omics data | Bioconductor |
| scran | 1.38.1 | Preprocessing & feature selection for single-cell/SpatialExperiment | Bioconductor |
| scater | 1.38.1 | QC, PCA and visualization for SpatialExperiment | Bioconductor |
| bluster | 1.20.0 | Clustering wrappers for Bioconductor objects | Bioconductor |
| ps | 1.9.3 | System process & memory monitoring | CRAN |
| stringr | 1.6.0 | String manipulation | CRAN |
| readr | 2.2.0 | Fast tsv/csv reading (used via write_csv/tibble) | CRAN |
