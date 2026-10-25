import os
import sys
# Fix windows console unicode error: MUST PUT AT VERY TOP
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
# ---------- SET GLOBAL FONT TO TIMES NEW ROMAN ----------
plt.rcParams['font.family']      = 'serif'
plt.rcParams['font.serif']       = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'   # for math symbols, if any

# ===================== CONFIG =====================
INPUT_FOLDER = Path(r"D:\My_Project_ST\benchmarking of ST platforms\SVGs_Clustering_Method\All_Scores_Cluster_Methods")
OUTPUT_FIG = Path(r"D:\My_Project_ST\benchmarking of ST platforms\Top_SVGs_Validation\Figure_C")
OUTPUT_FIG.mkdir(exist_ok=True)

# your 7 datasets
dataset_list = [
    "10X_Visium_Lung",
    "DLPFC_151507",
    "Slide-seq2",
    "SNARE-seq2",
    "ST",
    "STARmap",
    "Xenium_BC"
]

# ---- Legend display names (only affects legend text) ----
DISPLAY_LABELS = {
    "10X_Visium_Lung" : "10X_Visium",
    "DLPFC_151507"    : "DLPFC",
    "Xenium_BC"       : "Xenium",
}
# top‑k values for curve
k_values = [1, 5, 10, 20, 50, 100]

# ============== ADJUSTABLE FIGURE PARAMETERS ==============
FIG_WIDTH = 8
FIG_HEIGHT = 6
TITLE_TEXT = "SVG list agreement increases with the number"
XLABEL_TEXT = "Top‑k SVG genes per method"
YLABEL_TEXT = "Jaccard similarity between methods"
TITLE_FONTSIZE = 18   # <-- changed to 16
# ==========================================================

# full list of all observed adjusted p‑value column names
p_col_candidates = [
    "fdr_BH_p_adjust",
    "min_pvals_adj",
    "fdr_bh_padj",
    "fdr_bh_adjust_p",
    "fdr_BH_adjust_p",
    "p_adj_FDR",
    "FDR_BH_padj"
]

# Jaccard function
def jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union

# Step 1: load all csv files, store top‑100 gene list for each (dataset, method)
# dict key: (dataset, method), value: list of top‑100 gene names
top_gene_store = dict()
all_csv = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]

for filename in all_csv:
    stem = os.path.splitext(filename)[0]
    # parse filename method_dataset
    parts = stem.split("_", maxsplit=1)
    if len(parts) !=2:
        continue
    method, ds = parts
    if ds not in dataset_list:
        continue

    df = pd.read_csv(INPUT_FOLDER / filename)

    # find first existing p‑adjust column from candidate list
    sort_col = None
    for candidate in p_col_candidates:
        if candidate in df.columns:
            sort_col = candidate
            break

    if sort_col is None:
        print(f"Skip {filename}: no valid p-value column. Columns present: {list(df.columns)}")
        continue
    if "gene" not in df.columns:
        print(f"Skip {filename}: missing gene column")
        continue

    # sort ascending (small p-value first), take top‑100
    df_sorted = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    top100_genes = df_sorted["gene"].head(100).astype(str).tolist()
    top_gene_store[(ds, method)] = top100_genes

print(f"Loaded SVG lists for {len(top_gene_store)} (dataset,method) pairs")

# Step2 compute mean pairwise jaccard for each dataset at each k
results_records = []

for ds in dataset_list:
    # collect all available methods for this dataset
    ds_method_dict = {}
    for (d, m), genelist in top_gene_store.items():
        if d == ds:
            ds_method_dict[m] = genelist
    method_names = list(ds_method_dict.keys())
    n_methods = len(method_names)
    if n_methods <2:
        print(f"Dataset {ds}: only {n_methods} methods available, skip")
        continue

    # iterate k
    for k in k_values:
        # build set for each method top‑k
        gene_sets = {}
        for m in method_names:
            gl = ds_method_dict[m][:k]
            gene_sets[m] = set(gl)
        # all pairwise combinations
        jaccard_vals = []
        for m1, m2 in combinations(method_names, 2):
            j = jaccard(gene_sets[m1], gene_sets[m2])
            jaccard_vals.append(j)
        mean_j = np.mean(jaccard_vals)
        results_records.append({
            "dataset": ds,
            "k": k,
            "mean_pairwise_jaccard": mean_j
        })

df_jaccard = pd.DataFrame(results_records)
df_jaccard.to_csv(OUTPUT_FIG / "FigC_jaccard_topk_results.csv", index=False, encoding="utf‑8")
print("Jaccard table saved: FigC_jaccard_topk_results.csv")

# Step3 Plot Main Figure C
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# one line per dataset
for ds in dataset_list:
    sub = df_jaccard[df_jaccard["dataset"] == ds]


    plt.plot(sub["k"], sub["mean_pairwise_jaccard"], marker="o",
                 label=DISPLAY_LABELS.get(ds, ds))   # <-- use display label

plt.xlabel(XLABEL_TEXT, fontsize=16, fontweight="bold")                               # <-- X label size 14
plt.ylabel(YLABEL_TEXT, fontsize=16, fontweight="bold")                               # <-- Y label size 14
plt.title(TITLE_TEXT, fontsize=TITLE_FONTSIZE, fontweight="bold")  # <-- title 16 + bold
plt.xticks(k_values, fontsize=16)
plt.yticks(fontsize=16) 
plt.legend(title="Dataset",
           title_fontsize=15,                                      # <-- legend title 15
           fontsize=14,                                            # <-- legend text 13
           bbox_to_anchor=(1.04,1), loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIG / "FigC_jaccard_curve_New.png", dpi=300, bbox_inches="tight")
#plt.savefig(OUTPUT_FIG / "FigC_jaccard_curve.pdf", bbox_inches="tight")
plt.close()
print("✅ Main Fig‑C saved.")