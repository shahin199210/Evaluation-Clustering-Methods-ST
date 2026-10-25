import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
# ============================================================================
#  GLOBAL FONT SETTINGS
# ============================================================================
matplotlib.rcParams['font.family']        = 'Times New Roman'
matplotlib.rcParams['font.serif']         = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype']       = 42
matplotlib.rcParams['ps.fonttype']        = 42
# ============================================================================
#  CONFIGURATION
# ============================================================================
BASE_DIR     = Path(r"D:\My_Project_ST\benchmarking of ST platforms")
SIM_DATA_ROOT   = BASE_DIR / "Simulation_data"
GLOBAL_RESULTS   = SIM_DATA_ROOT / "Global_SVGs_detection" / "All_Scores"
CLUSTER_RESULTS  = SIM_DATA_ROOT / "SVG_Detection_Clustering Methods" / "All_scores"
TRUE_SVGS_FOLDER = SIM_DATA_ROOT / "SRTsim_12_datasets" / "True_SVGs"/"Original"
OUTPUT_DIR = SIM_DATA_ROOT / "Evaluation_Results/accuracy_plot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FDR_THRESHOLD  = 0.05
# -------------------------- DUAL LOGFC CUTOFF --------------------------
LOGFC_THRESHOLD_CLUSTER = 1.0
LOGFC_THRESHOLD_GLOBAL  = 0.05
GLOBAL_METHODS = ["moran", "DESpace", "nnsvg", "SPARKX"]
# --- Priority order for gene columns (first match wins) ---
GENE_COLUMNS = ["gene_symbol", "gene_name", "gene"]
# --- Adjusted p-value column candidates (lowercased) ---
ADJ_PVAL_COLUMNS = [
    "min_pvals_adj", "fdr_bh_padj", "fdr_bh_adjust_p",
    "p_adj_fdr", "padj", "p_adj", "adj_p_value",
    "qvalue", "fdr", "q_val",
    "fdr_bh",          # ← matches "fdr_bh" column
    "bh",
]
# ---- Plot size (edit freely) ----
FIG_WIDTH  = 22
FIG_HEIGHT = 6
# ============================================================================
#  HELPERS
# ============================================================================
def read_result_file(filepath):
    if filepath.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def find_gene_column(df):
    """Return the best gene column name based on priority:
        gene_symbol > gene_name > gene > any column containing 'gene'."""
    for col in GENE_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        if 'gene' in col:
            return col
    raise ValueError(f"No gene column found. Columns: {df.columns}")
def find_padj_column(df):
    """Return the adjusted p-value column."""
    for col in df.columns:
        if col in ADJ_PVAL_COLUMNS:
            return col
    for col in df.columns:
        if col in ('p_value', 'pval', 'p_val', 'pvalue', 'p'):
            continue
        if (('adj' in col or 'adjust' in col or 'fdr' in col
             or 'qvalue' in col or 'bh' in col)
            and ('p' in col or 'fdr' in col or 'bh' in col)):
            return col
    raise ValueError(f"No adjusted p-value column found. Columns: {df.columns}")
def find_maxlogfc_column(df):
    """Find max_logfc column (lowercase after read)."""
    for col in df.columns:
        if col == "max_logfc":
            return col
    for col in df.columns:
        if 'max_logfc' in col:
            return col
    return None
def get_gene_and_padj_columns(df):
    """Convenience wrapper — returns (gene_col, padj_col)."""
    return find_gene_column(df), find_padj_column(df)
def load_ground_truth_svgs(sim_id):
    """Return set of true SVG gene symbols (upper-cased) for a replicate."""
    fp = None
    for ext in ['.txt', '.csv', '.tsv']:
        candidate = TRUE_SVGS_FOLDER / f"{sim_id}_true_svg_genes_symbol{ext}"
        if candidate.exists():
            fp = candidate
            break
    if fp is None:
        for ext in ['.txt', '.csv']:
            candidate = TRUE_SVGS_FOLDER / f"{sim_id}_true_svg_genes{ext}"
            if candidate.exists():
                fp = candidate
                break
    if fp is None:
        raise FileNotFoundError(f"No ground truth file for {sim_id}")
    if fp.suffix in ('.csv', '.tsv'):
        df = pd.read_csv(fp, sep=None, engine='python')
        genes = df.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
    else:
        with open(fp) as f:
            genes = [line.strip().upper() for line in f if line.strip()]
    return set(g for g in genes if g and g != 'NAN')
def compute_pr_auc(y_true, y_score):
    """Area under the Precision–Recall curve via trapezoidal integration."""
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    order = np.argsort(-y_score, kind='mergesort')
    y_true = y_true[order].astype(float)
    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall    = tp / y_true.sum()
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    return float(trapz(precision, recall))
# Modified: pass is_global flag to select group-specific logFC cutoff
def compute_aurp(df, true_svgs, is_global, verbose=False, filepath=None):
    """
    AURP for one method × one replicate.
    Filter genes: FDR<0.05 AND group-specific max_logFC cutoff
    Clustering: max_logFC>1.0; Global: max_logFC>0.05
    """
    padj_col = find_padj_column(df)
    logfc_col = find_maxlogfc_column(df)
    df = df.copy()
    df[padj_col] = pd.to_numeric(df[padj_col], errors='coerce')
    if logfc_col is not None:
        df[logfc_col] = pd.to_numeric(df[logfc_col], errors='coerce')
    df = df.dropna(subset=[padj_col])

    # Assign logFC threshold based on method group
    if is_global:
        logfc_thresh = LOGFC_THRESHOLD_GLOBAL
    else:
        logfc_thresh = LOGFC_THRESHOLD_CLUSTER
    # Apply dual filter
    if logfc_col is not None:
        mask = (df[padj_col] < FDR_THRESHOLD) & (df[logfc_col] > logfc_thresh)
        df_filtered = df[mask].copy()
    else:
        # Print the EXACT filename so you can locate which dataset lacks the column
        fname = Path(filepath).name if filepath is not None else "UNKNOWN"
        print(f"⚠ max_logfc column MISSING in file: {fname}  -> using only FDR filter")
        df_filtered = df[df[padj_col] < FDR_THRESHOLD].copy()
    if df_filtered.empty:
        return np.nan
    # Candidate gene columns (priority first, then any remaining 'gene*' column)
    candidates = []
    for col in GENE_COLUMNS:
        if col in df_filtered.columns and col not in candidates:
            candidates.append(col)
    for col in df_filtered.columns:
        if 'gene' in col and col != padj_col and col not in candidates:
            candidates.append(col)
    if not candidates:
        raise ValueError(f"No gene column found. Columns: {df_filtered.columns}")
    best = None  # (overlap, y_true, y_score, gene_col)
    for gcol in candidates:
        sub = df_filtered[[gcol, padj_col]].copy()
        sub[gcol] = sub[gcol].astype(str).str.strip().str.upper()
        sub = (sub.sort_values(padj_col)
                .drop_duplicates(subset=[gcol], keep='first'))
        y_true  = sub[gcol].isin(true_svgs).astype(int).values
        overlap = int(y_true.sum())
        if best is None or overlap > best[0]:
            y_score = -np.log10(sub[padj_col].clip(lower=1e-300).values)
            best = (overlap, y_true, y_score, gcol)
    overlap, y_true, y_score, chosen = best
    if verbose:
        print(f"      chosen gene_col='{chosen}', overlap={overlap}/{len(y_true)}")
    if overlap == 0 or overlap == len(y_true):
        return np.nan
    return compute_pr_auc(y_true, y_score)
# ============================================================================
#  COLLECT AURP PER METHOD × SIM
# ============================================================================
def collect_aurp():
    all_files = (list(GLOBAL_RESULTS.glob("sim_*_*.*")) +
                 list(CLUSTER_RESULTS.glob("sim_*_*_SVG_AllGenes.*")))
    sim_ids = set()
    for f in all_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            sim_ids.add(f"{parts[0]}_{parts[1]}")
    sim_ids = sorted(sim_ids)
    print(f"Detected simulation replicates: {sim_ids}")
    true_svgs = {}
    for sid in sim_ids:
        try:
            true_svgs[sid] = load_ground_truth_svgs(sid)
            print(f"  {sid}: {len(true_svgs[sid])} true SVGs")
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
    cluster_files = list(CLUSTER_RESULTS.glob("sim_*_*_SVG_AllGenes.*"))
    global_files  = list(GLOBAL_RESULTS.glob("sim_*_*.*"))
    method_files = {}
    for f in cluster_files:
        stem = f.stem
        parts = stem.split('_')
        if len(parts) < 4:
            continue
        sim_id = f"{parts[0]}_{parts[1]}"
        idx = stem.find('SVG_AllGenes')
        method = stem[len(sim_id) + 1: idx - 1] if idx != -1 else '_'.join(parts[2:-3])
        method_files[(sim_id, method, False)] = f
    global_lower = [m.lower() for m in GLOBAL_METHODS]
    for f in global_files:
        stem = f.stem
        parts = stem.split('_')
        if len(parts) < 3:
            continue
        sim_id = f"{parts[0]}_{parts[1]}"
        method = '_'.join(parts[2:])
        if method.lower() in global_lower:
            method_files[(sim_id, method, True)] = f
    records = []
    for (sim_id, method, is_global), filepath in sorted(method_files.items()):
        if sim_id not in sim_ids or sim_id not in true_svgs:
            continue
        try:
            df = read_result_file(filepath)
            # Pass is_global to compute_aurp to pick correct logFC cutoff
            aurp = compute_aurp(df, true_svgs[sim_id], is_global=is_global, verbose=False, filepath=filepath)
            records.append({
                'sim': sim_id,
                'method': method,
                'is_global': is_global,
                'AURP': aurp,
            })
            print(f"  ✓ {sim_id} | {method:20s} AURP={aurp:.4f}")
        except Exception as e:
            print(f"  ✗ Skipping {sim_id} | {method}: {e}")
    return pd.DataFrame(records)
# ============================================================================
#  VIOLIN PLOT: AURP PER METHOD
# ============================================================================
def plot_aurp_violinplot(df):
    if df.empty or df.dropna(subset=['AURP']).empty:
        print("⚠ No AURP values collected — nothing to plot.")
        return
    df = df.dropna(subset=['AURP']).copy()
    CLUSTER_COLOR = '#2E86C1'   # blue
    GLOBAL_COLOR  = '#E74C3C'   # red
    med = (df.groupby(['method', 'is_global'])['AURP']
            .median()
            .reset_index())
    clust = (med[~med['is_global']]
             .sort_values('AURP', ascending=False)
             .reset_index(drop=True))
    glob  = (med[med['is_global']]
             .sort_values('AURP', ascending=False)
             .reset_index(drop=True))
    ordered = pd.concat([clust, glob], ignore_index=True)
    methods       = ordered['method'].tolist()
    is_global_map = dict(zip(ordered['method'], ordered['is_global']))
    data_list     = [df.loc[df['method'] == m, 'AURP'].values for m in methods]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    positions = np.arange(len(methods))

    # Assign color for each violin
    violin_colors = []
    for m in methods:
        if is_global_map[m]:
            violin_colors.append(GLOBAL_COLOR)
        else:
            violin_colors.append(CLUSTER_COLOR)

    # Draw violin plot
    vp = ax.violinplot(data_list, positions=positions, widths=0.7,
                       showmedians=True, showextrema=True)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.85)
    # Style median, extrema lines
    vp['cmedians'].set(color='black', linewidth=1.8)
    vp['cmaxes'].set(color='black', linewidth=1.0)
    vp['cmins'].set(color='black', linewidth=1.0)

    # Vertical dashed divider between clustering / global groups
    pos_clust  = [p for p, m in zip(positions, methods) if not is_global_map[m]]
    pos_glob   = [p for p, m in zip(positions, methods) if is_global_map[m]]
    if pos_clust and pos_glob:
        ax.axvline((max(pos_clust) + min(pos_glob)) / 2.0,
                   color='gray', linestyle='--', linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=45, ha='right',
                       fontsize=16, fontfamily='Times New Roman')
    ax.set_xlabel('Method', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('AURP score', fontsize=18, fontweight='bold', fontfamily='Times New Roman')
    ax.tick_params(axis='y', labelsize=16)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily('Times New Roman')
    ax.set_title(
        'AURP score per method across 12 simulation replicates',
        fontsize=20, fontweight='bold', fontfamily='Times New Roman', pad=15
    )
    legend_elements = [
        Patch(facecolor=CLUSTER_COLOR, edgecolor='black', label='Clustering-based (27)'),
        Patch(facecolor=GLOBAL_COLOR,  edgecolor='black', label='Global (4)'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                    prop={'family': 'Times New Roman', 'size': 15, 'weight': 'bold'})
    leg.get_frame().set_edgecolor('black')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(left = -0.5, right = len(methods)-0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    # Output filenames for dual logFC cutoff
    out_png = OUTPUT_DIR / "AURP_violinplot_FDR0.05_dual_logfc.png"
    out_csv = OUTPUT_DIR / "AURP_per_method_per_sim_FDR0.05_dual_logfc.csv"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Violin plot saved : {out_png}")
    print(f"✓ AURP table saved: {out_csv}")
    print("\n--- Median AURP ranking (descending, FDR<0.05, dual logFC cutoff) ---")
    for _, r in ordered.iterrows():
        grp = 'Global' if is_global_map[r['method']] else 'Clustering'
        print(f"  [{grp:10s}]  {r['method']:22s}  median AURP={r['AURP']:.4f}")
# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    df = collect_aurp()
    print(f"\nCollected {len(df)} method × replicate AURP values "
          f"covering {df['method'].nunique()} methods.")
    df.to_csv(OUTPUT_DIR / "AURP_per_method_per_sim_raw_FDR0.05_dual_logfc.csv", index=False)
    plot_aurp_violinplot(df)
    print("\nDone.")