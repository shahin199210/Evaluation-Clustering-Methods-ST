import sys
sys.path.append(r"D:\My_Project_ST")
import pandas as pd
import utils_for_all as usa

BASE_DATA_PATH = r"D:\My_Project_ST\DataSets"

ALL_DATASETS = [
    '151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676',
    'BC_TENX13','BC_TENX23','BC_TENX39','Lung_TENX62','Lung_TENX72',
    "Patient1_Biopsy1_rep1_chip1_d1_stdata","Patient3_Biopsy2_rep2_Chip3_C2_stdata",
    "Patient3_Biopsy4_rep2_Chip3_E2_stdata","BC_SPA130",
    'Kedney_HBM232.MBNR.586','Kedney_HBM478.VWXX.362','Kedney_HBM965.PSNC.855',
    'LungL_HBM477.TMMN.763','LungL_HBM582.RKVK.974','LungR_HBM285.MWNX.593','LungR_HBM753.NSKJ.788',
    "STARmap_AD_13m_disease","BOOST_HMI_starmap","Tutorial_STARmap",
    'BC_NCBI784','Lung_NCBI856','Lung_NCBI858','Lung_TENX141','Lung_NCBI882',
    'Colorectal_TENX139', 'Ovarian_TENX142', 'Skin_TENX158'
]

summary_list = []
for ds in ALL_DATASETS:
    print(f"Loading dataset: {ds}")
    adata, n_cluster = usa.load_dataset(ds, base_data_path=BASE_DATA_PATH)
    n_spots = adata.n_obs
    n_genes = adata.n_vars
    summary_list.append({
        "Dataset_Name": ds,
        "No_of_spots_cells": n_spots,
        "No_of_genes": n_genes,
        "No_of_clusters": n_cluster
    })

df_summary = pd.DataFrame(summary_list)
df_summary.to_csv(r"D:\My_Project_ST\Auto_Dataset_Summary.csv", index=False)
print(df_summary)