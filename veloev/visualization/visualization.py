import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


def plot_task(
    benchmark_info: Dict[str, Any], 
    plot_type: str = 'directional',
    base_dir: Union[str, Path] = './',
    save_path: str = None,
    color_palette: Optional[Dict[str, str]] = None
):
    """
    Generates a seamless, compact summary figure for a specific task.
    Final Adjustments:
    - Rank Box Size reduced to 0.45.
    - Vertical header lines extended correctly to the second row level.
    - No scaling on values (Raw display).
    """
    
    # --- 0. Configuration ---
    if color_palette is None:
        color_palette = {
            'bar1': "#5b7e91",
            'bar2': "#e4ab9b",
            'rank_low': "#bf794e",
            'rank_high': "#e6b422"
        }
    
    TYPE_CONFIG = {
        'directional': {
            'title': 'Directional consistency',
            'metrics': ['cbdir', 'icvcoh'], 
            'labels': ['CBDir', 'ICVCoh'],
            'valid_ds_types': ['directional', 'directional_temporal']
        },
        'temporal': {
            'title': 'Temporal precision',
            'metrics': ['cto', 'tsc'], 
            'labels': ['CTO', 'TSC'],
            'valid_ds_types': ['temporal', 'directional_temporal']
        },
        'negative_control': {
            'title': 'Negative control robustness',
            'metrics': ['sts', 'nte'], 
            'labels': ['STS', 'NTE'],
            'valid_ds_types': ['negative_control']
        },
        'simulation': {
            'title': 'Simulation',
            'metrics': ['dcor', 'pr'], 
            'labels': ['dCor', 'Pearson'],
            'valid_ds_types': ['simulation']
        },
        'seq_depth_directional': {
            'title': 'Sequence depth stability (directional)',
            'metrics': ['s_cbdir', 's_icvcoh'], 
            'labels': [r'$S_{CBDir}$', r'$S_{ICVCoh}$'],
            'valid_ds_types': ['seq_depth_directional', 'seq_depth_directional_temporal']
        },
        'seq_depth_temporal': {
            'title': 'Sequence depth stability (temporal)',
            'metrics': ['s_cto', 's_tsc'], 
            'labels': [r'$S_{CTO}$', r'$S_{TSC}$'],
            'valid_ds_types': ['seq_depth_temporal', 'seq_depth_directional_temporal']
        }
    }

    if plot_type not in TYPE_CONFIG:
        raise ValueError(f"Unknown plot_type: '{plot_type}'")
        
    cfg = TYPE_CONFIG[plot_type]
    metrics = cfg['metrics']
    metric_labels = cfg['labels']
    type_title = cfg['title']
    valid_types = cfg['valid_ds_types']
    
    BIPOLAR_METRICS = ['cbdir', 'tsc', 'pr']

    # --- 2. Load Data ---
    names_list = benchmark_info.get('datasets_name')
    types_list = benchmark_info.get('tasks')
    methods = benchmark_info['methods']
    
    dataset_paths = []
    for name, dtype in zip(names_list, types_list):
        if dtype in valid_types:
            dataset_paths.append(Path(base_dir) / name / 'evaluation')

    num_datasets = len(dataset_paths)
    if num_datasets == 0:
        print(f"No datasets found for plot_type: {plot_type}.")
        return

    # Storage (Store RAW values)
    storage = {m: {meth: [] for meth in methods} for m in metrics}

    for metric in metrics:
        for ds_path in dataset_paths:
            file_path = ds_path / f"{metric}_df.csv"
            if not file_path.exists():
                continue
            
            df = pd.read_csv(file_path)
            df = df[df['Method'].isin(methods)]
            
            for _, row in df.iterrows():
                meth = row['Method']
                fold_scores = row.iloc[1:].dropna().astype(float).values
                
                if len(fold_scores) > 0:
                    ds_mean = np.mean(fold_scores)
                    storage[metric][meth].append(ds_mean)

    # --- 3. Compute Stats ---
    method_stats = []
    for meth in methods:
        row_data = {'Method': meth}
        for metric in metrics:
            data = np.array(storage[metric][meth])
            if len(data) == 0:
                mean, std = 0.0, 0.0
            else:
                mean = np.mean(data)
                std = np.std(data)
            
            row_data[f'{metric}_mean'] = mean
            row_data[f'{metric}_std'] = std
            
        method_stats.append(row_data)

    df_stats = pd.DataFrame(method_stats)

    if 'methods_map' in benchmark_info:
        df_stats['Method'] = df_stats['Method'].replace(benchmark_info['methods_map'])

    # Ranks & Sort
    for metric in metrics:
        df_stats[f'{metric}_rank'] = df_stats[f'{metric}_mean'].rank(ascending=False, method='min').astype(int)

    rank_cols = [f'{m}_rank' for m in metrics]
    df_stats['overall_rank'] = df_stats[rank_cols].mean(axis=1).rank(ascending=True, method='min').astype(int)

    df_stats = df_stats.sort_values(by='overall_rank', ascending=True).reset_index(drop=True)
    sorted_methods = df_stats['Method'].tolist()

    # --- 4. Visualization ---
    nrows = len(sorted_methods)
    
    # Layout
    width_ratios = [1.5, 1.5, 0.6, 1.5, 0.6, 0.7]
    fig_width = sum(width_ratios) * 1.5
    
    fig, axes = plt.subplots(nrows=nrows, ncols=6, 
                             figsize=(fig_width, 0.6 * nrows), 
                             gridspec_kw={'width_ratios': width_ratios, 
                                          'wspace': 0.0, 'hspace': 0})
    
    if nrows == 1: axes = np.array([axes])

    rank_cmap = sns.blend_palette([color_palette['rank_high'], color_palette['rank_low']], as_cmap=True)
    
    # [ADJUSTMENT] Smaller Box
    BOX_SIZE = 0.45 
    OFFSET = (1 - BOX_SIZE) / 2

    # --- Draw Rows ---
    for i, row in df_stats.iterrows():
        method = row['Method'] 
        
        # --- Col 0: Method Name ---
        ax_m = axes[i, 0]
        ax_m.text(0.5, 0.5, method, va='center', ha='center', fontsize=10, color='black', transform=ax_m.transAxes)
        ax_m.set_axis_off()
        ax_m.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_m.transAxes, clip_on=False)
        ax_m.plot([1, 1], [0, 1], color='lightgray', lw=0.5, transform=ax_m.transAxes, clip_on=False) 

        # --- Metric Columns ---
        for idx, metric in enumerate(metrics):
            col_bar = 1 + (idx * 2)
            col_rank = 2 + (idx * 2)
            
            raw_mean = row[f'{metric}_mean']
            raw_std = row[f'{metric}_std']
            rank = int(row[f'{metric}_rank'])

            # -- Calculate Visual Bar Length --
            if metric in BIPOLAR_METRICS:
                vis_mean = (raw_mean + 1) / 2
                vis_mean = np.clip(vis_mean, 0, 1)
                vis_std = raw_std / 2
            else:
                vis_mean = raw_mean
                vis_std = raw_std

            # -- Bar Plot --
            current_bar_color = color_palette['bar1'] if idx == 0 else color_palette['bar2']
            ax_bar = axes[i, col_bar]
            
            if num_datasets > 1:
                xerr_val = vis_std
                capsize_val = 3
            else:
                xerr_val = None
                capsize_val = 0

            ax_bar.barh(0, vis_mean, xerr=xerr_val, color=current_bar_color, height=0.6, 
                        capsize=capsize_val, edgecolor='none', error_kw={'elinewidth': 1, 'ecolor': 'black'})
            
            ax_bar.set_xlim(0, 1.25); ax_bar.set_ylim(-0.5, 0.5); ax_bar.axis('off')
            
            # Text Display (RAW Value)
            text_x_pos = min(vis_mean + (vis_std if num_datasets > 1 else 0) + 0.05, 1.15)
            ax_bar.text(text_x_pos, 0, f"{raw_mean:.2f}", va='center', fontsize=9)
            
            ax_bar.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_bar.transAxes, clip_on=False)

            # -- Rank Heatmap --
            ax_rank = axes[i, col_rank]
            ax_rank.set_xlim(0, 1); ax_rank.set_ylim(0, 1)

            norm_val = (rank - 1) / (len(sorted_methods) - 1 + 1e-9)
            rect = plt.Rectangle((OFFSET, OFFSET), BOX_SIZE, BOX_SIZE, color=rank_cmap(norm_val), ec='white', lw=1)
            ax_rank.add_patch(rect)
            ax_rank.text(0.5, 0.5, str(rank), va='center', ha='center', fontsize=10)
            ax_rank.set_axis_off()
            ax_rank.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_rank.transAxes, clip_on=False)
            
            # Separators
            if idx == 0:
                # Dashed separator between M1 Rank and M2 Bar
                ax_rank.plot([1, 1], [0, 1], color='gray', lw=0.5, ls='--', transform=ax_rank.transAxes, clip_on=False)
            elif idx == len(metrics) - 1:
                # Solid separator before Overall
                ax_rank.plot([1, 1], [0, 1], color='lightgray', lw=0.5, transform=ax_rank.transAxes, clip_on=False)

        # --- Col 5: Overall Rank ---
        ax_ovr = axes[i, 5]
        ax_ovr.set_xlim(0, 1); ax_ovr.set_ylim(0, 1)

        ov_val = int(row['overall_rank'])
        norm_ov = (ov_val - 1) / (len(sorted_methods) - 1 + 1e-9)
        rect = plt.Rectangle((OFFSET, OFFSET), BOX_SIZE, BOX_SIZE, color=rank_cmap(norm_ov), ec='white', lw=1)
        ax_ovr.add_patch(rect)
        ax_ovr.text(0.5, 0.5, str(ov_val), va='center', ha='center', fontsize=12)
        ax_ovr.set_axis_off()
        ax_ovr.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_ovr.transAxes, clip_on=False)

    # --- HEADERS ---
    ROW1_Y = 1.7   
    ROW2_Y = 1.15  
    LINE_Y = 1.45  
    METHOD_Y = (ROW1_Y + ROW2_Y) / 2
    
    if nrows > 0:
        # 1. Method
        ax_m0 = axes[0, 0]
        ax_m0.text(0.5, METHOD_Y, "Method", ha='center', va='center', transform=ax_m0.transAxes, fontsize=11)
        
        # 2. Type Title (Centered)
        ax_bar2 = axes[0, 3]
        ax_bar2.text(0.0, ROW1_Y, type_title, ha='center', va='center', transform=ax_bar2.transAxes, fontsize=11, color='black')

        # 3. Metric Headers
        axes[0, 1].text(0.5, ROW2_Y, metric_labels[0], ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=10)
        axes[0, 2].text(0.5, ROW2_Y, "Rank", ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=10)
        
        axes[0, 3].text(0.5, ROW2_Y, metric_labels[1], ha='center', va='center', transform=axes[0, 3].transAxes, fontsize=10)
        axes[0, 4].text(0.5, ROW2_Y, "Rank", ha='center', va='center', transform=axes[0, 4].transAxes, fontsize=10)

        # 4. Overall Header
        ax_ov = axes[0, 5]
        ax_ov.text(0.5, ROW1_Y, "Overall", ha='center', va='center', transform=ax_ov.transAxes, fontsize=11)
        ax_ov.text(0.5, ROW2_Y, "Rank", ha='center', va='center', transform=ax_ov.transAxes, fontsize=10)

        # --- Lines ---
        for col_idx in range(1, 6):
            # Horizontal line under Type Title / Overall
            axes[0, col_idx].plot([0, 1], [LINE_Y, LINE_Y], color='gray', lw=0.5, transform=axes[0, col_idx].transAxes, clip_on=False)

        # [HEADER VERTICALS]
        # Dashed line between Metric 1 and Metric 2 in Header (Right of Col 2)
        # Extending from bottom of data row (1.0) up to Line Y (1.45)
        axes[0, 2].plot([1, 1], [1, LINE_Y], color='gray', lw=0.5, ls='--', transform=axes[0, 2].transAxes, clip_on=False)
        
        # [ADDED] Solid line between Metric 2 and Overall in Header (Right of Col 4)
        axes[0, 4].plot([1, 1], [1, LINE_Y], color='lightgray', lw=0.5, transform=axes[0, 4].transAxes, clip_on=False)

    # --- Borders ---
    for ax in axes[0, :]:
        ax.plot([0, 1], [1, 1], color='black', lw=1, transform=ax.transAxes, clip_on=False)

    for ax in axes[-1, :]:
        ax.plot([0, 1], [0, 0], color='black', lw=1, transform=ax.transAxes, clip_on=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_overall(
    benchmark_info: Dict[str, Any], 
    include_types: List[str],
    base_dir: Union[str, Path] = './',
    save_path: str = None,
    color_palette: Optional[Dict[str, str]] = None,
    type_colors: Optional[Dict[str, str]] = None
):
    """
    Generates a seamless table-like summary figure.
    Updates:
    - Displays RAW values in text.
    - Bars are visually scaled to [0,1] for uniformity.
    - Includes Type Titles, dashed separators, and compact layout.
    """
    
    # --- 0. Configuration ---
    DEFAULT_TYPE_COLORS = {
        'directional': '#ae7c58', 
        'temporal': '#6b6f59', 
        'negative_control': '#6c848d',
        'seq_depth_directional': '#95859c',
        'simulation': '#a86965',
        'seq_depth_temporal': "#839b5c"
    }
    
    if type_colors:
        DEFAULT_TYPE_COLORS.update(type_colors)
    
    if color_palette is None:
        color_palette = {
            'rank_low': "#ecd7a3", 
            'rank_high': "#e68b40"
        }
    
    TYPE_CONFIG = {
        'directional': {
            'title': 'Directional Consistency',
            'metrics': ['cbdir', 'icvcoh'], 
            'labels': ['CBDir', 'ICVCoh'],
            'valid_ds_types': ['directional', 'directional_temporal']
        },
        'temporal': {
            'title': 'Temporal Precision',
            'metrics': ['cto', 'tsc'], 
            'labels': ['CTO', 'TSC'],
            'valid_ds_types': ['temporal', 'directional_temporal']
        },
        'negative_control': {
            'title': 'Negative Control Robustness',
            'metrics': ['sts', 'nte'], 
            'labels': ['STS', 'NTE'],
            'valid_ds_types': ['negative_control']
        },
        'simulation': {
            'title': 'Simulation',
            'metrics': ['dcor', 'pr'], 
            'labels': ['dCor', 'Pearson'],
            'valid_ds_types': ['simulation']
        },
        'seq_depth_directional': {
            'title': 'Sequence depth stability (directional)',
            'metrics': ['s_cbdir', 's_icvcoh'], 
            'labels': [r'$S_{CBDir}$', r'$S_{ICVCoh}$'],
            'valid_ds_types': ['seq_depth_directional', 'seq_depth_directional_temporal']
        },
        'seq_depth_temporal': {
            'title': 'Sequence depth stability (temporal)',
            'metrics': ['s_cto', 's_tsc'], 
            'labels': [r'$S_{CTO}$', r'$S_{TSC}$'],
            'valid_ds_types': ['seq_depth_temporal', 'seq_depth_directional_temporal']
        }
    }

    # Metrics that need visual scaling [-1, 1] -> [0, 1]
    BIPOLAR_METRICS = ['cbdir', 'tsc', 'pr']

    # Validate Inputs
    for t in include_types:
        if t not in TYPE_CONFIG:
            raise ValueError(f"Unknown type: {t}")

    # --- 1. Load & Aggregate Data ---
    methods = benchmark_info['methods']
    names_list = benchmark_info.get('datasets_name')
    types_list = benchmark_info.get('tasks')
    
    method_data = {m: {} for m in methods}
    type_ds_counts = {}

    for dtype in include_types:
        cfg = TYPE_CONFIG[dtype]
        metrics = cfg['metrics']
        
        ds_paths = []
        for name, dst in zip(names_list, types_list):
            if dst in cfg['valid_ds_types']:
                ds_paths.append(Path(base_dir) / name / 'evaluation')
        
        type_ds_counts[dtype] = len(ds_paths)
        if len(ds_paths) == 0:
            print(f"Warning: No datasets found for {dtype}")
            continue

        temp_storage = {m: {meth: [] for meth in methods} for m in metrics}
        
        for metric in metrics:
            for p in ds_paths:
                fpath = p / f"{metric}_df.csv"
                if not fpath.exists(): continue
                df = pd.read_csv(fpath)
                df = df[df['Method'].isin(methods)]
                for _, row in df.iterrows():
                    meth = row['Method']
                    vals = row.iloc[1:].dropna().astype(float).values
                    if len(vals) > 0:
                        val = np.mean(vals)
                        # [CHANGE] Store RAW value
                        temp_storage[metric][meth].append(val)

        type_scores_for_ranking = {meth: [] for meth in methods}

        for meth in methods:
            stats = {}
            for i, metric in enumerate(metrics):
                data = np.array(temp_storage[metric][meth])
                if len(data) == 0:
                    mean, std = 0.0, 0.0
                else:
                    mean = np.mean(data)
                    std = np.std(data)
                
                stats[f'm{i+1}_mean'] = mean
                stats[f'm{i+1}_std'] = std
                
                # For ranking, higher raw value is better
                type_scores_for_ranking[meth].append(mean)
            
            method_data[meth][dtype] = stats

        avg_scores = [np.mean(type_scores_for_ranking[m]) for m in methods]
        ranks = pd.Series(avg_scores).rank(ascending=False, method='min').astype(int)
        
        for meth, r in zip(methods, ranks):
            method_data[meth][dtype]['rank'] = r

    # --- 2. Build DataFrame ---
    rows = []
    for meth in methods:
        row = {'Method': meth}
        rank_list = []
        for dtype in include_types:
            if dtype in method_data[meth]:
                d = method_data[meth][dtype]
                row[f'{dtype}_m1_mean'] = d.get('m1_mean', 0)
                row[f'{dtype}_m1_std'] = d.get('m1_std', 0)
                row[f'{dtype}_m2_mean'] = d.get('m2_mean', 0)
                row[f'{dtype}_m2_std'] = d.get('m2_std', 0)
                row[f'{dtype}_rank'] = d.get('rank', len(methods))
                rank_list.append(row[f'{dtype}_rank'])
            else:
                rank_list.append(len(methods)) 

        row['avg_type_rank'] = np.mean(rank_list) if rank_list else len(methods)
        rows.append(row)

    df = pd.DataFrame(rows)
    if 'methods_map' in benchmark_info:
        df['Method'] = df['Method'].replace(benchmark_info['methods_map'])

    df['final_rank'] = df['avg_type_rank'].rank(ascending=True, method='min').astype(int)
    df = df.sort_values(by='final_rank', ascending=True).reset_index(drop=True)
    sorted_methods = df['Method'].tolist()

    # --- 3. Visualization ---
    n_types = len(include_types)
    nrows = len(sorted_methods)
    
    # Layout Ratios
    W_METHOD = 1.5
    W_BAR1 = 1.5
    W_BAR2 = 1.5
    W_RANK = 0.6
    W_OVERALL = 0.7
    
    width_ratios = [W_METHOD] + [W_BAR1, W_BAR2, W_RANK] * n_types + [W_OVERALL]
    total_cols = 1 + 3 * n_types + 1
    
    fig_width = 2 + (3.6 * n_types) + 1
    
    fig, axes = plt.subplots(nrows=nrows, ncols=total_cols, 
                             figsize=(fig_width, 0.6 * nrows), 
                             gridspec_kw={'width_ratios': width_ratios, 
                                          'wspace': 0.0, 'hspace': 0})
    
    if nrows == 1: axes = np.array([axes])
    
    rank_cmap = sns.blend_palette([color_palette['rank_high'], color_palette['rank_low']], as_cmap=True)
    
    BOX_SIZE = 0.45
    OFFSET = (1 - BOX_SIZE) / 2

    # --- Draw Data Rows ---
    for i, row in df.iterrows():
        method = row['Method']
        
        # Col 0: Method Name
        ax_m = axes[i, 0]
        ax_m.text(0.5, 0.5, method, va='center', ha='center', fontsize=11, transform=ax_m.transAxes)
        ax_m.set_axis_off()
        ax_m.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_m.transAxes, clip_on=False)
        ax_m.plot([1, 1], [0, 1], color='lightgray', lw=0.5, transform=ax_m.transAxes, clip_on=False) 

        # Type Groups
        for t_idx, dtype in enumerate(include_types):
            base_col = 1 + (t_idx * 3)
            current_color = DEFAULT_TYPE_COLORS.get(dtype, '#76b5c5')
            metrics_list = TYPE_CONFIG[dtype]['metrics']
            
            # Retrieve Raw Data
            m1_raw = row[f'{dtype}_m1_mean']
            m1_std_raw = row[f'{dtype}_m1_std']
            m2_raw = row[f'{dtype}_m2_mean']
            m2_std_raw = row[f'{dtype}_m2_std']
            rank = int(row[f'{dtype}_rank'])
            
            has_error_bar = type_ds_counts[dtype] > 1

            # --- Process Metric 1 (Bar 1) ---
            metric_name_1 = metrics_list[0]
            if metric_name_1 in BIPOLAR_METRICS:
                vis_m1 = (m1_raw + 1) / 2
                vis_m1 = np.clip(vis_m1, 0, 1)
                vis_std1 = m1_std_raw / 2
            else:
                vis_m1 = m1_raw
                vis_std1 = m1_std_raw

            ax_b1 = axes[i, base_col]
            xerr1 = vis_std1 if has_error_bar else None
            cap1 = 3 if has_error_bar else 0
            
            ax_b1.barh(0, vis_m1, xerr=xerr1, color=current_color, height=0.6, 
                       capsize=cap1, edgecolor='none', error_kw={'elinewidth': 1, 'ecolor': 'black'})
            ax_b1.set_xlim(0, 1.25); ax_b1.set_ylim(-0.5, 0.5); ax_b1.axis('off')
            
            # Text: Display RAW value
            text_x1 = min(vis_m1 + (vis_std1 if has_error_bar else 0) + 0.05, 1.15)
            ax_b1.text(text_x1, 0, f"{m1_raw:.2f}", va='center', fontsize=9)
            ax_b1.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_b1.transAxes, clip_on=False)

            # --- Process Metric 2 (Bar 2) ---
            metric_name_2 = metrics_list[1]
            if metric_name_2 in BIPOLAR_METRICS:
                vis_m2 = (m2_raw + 1) / 2
                vis_m2 = np.clip(vis_m2, 0, 1)
                vis_std2 = m2_std_raw / 2
            else:
                vis_m2 = m2_raw
                vis_std2 = m2_std_raw

            ax_b2 = axes[i, base_col + 1]
            xerr2 = vis_std2 if has_error_bar else None
            cap2 = 3 if has_error_bar else 0
            
            ax_b2.barh(0, vis_m2, xerr=xerr2, color=current_color, height=0.6, 
                       capsize=cap2, edgecolor='none', error_kw={'elinewidth': 1, 'ecolor': 'black'})
            ax_b2.set_xlim(0, 1.25); ax_b2.set_ylim(-0.5, 0.5); ax_b2.axis('off')
            
            # Text: Display RAW value
            text_x2 = min(vis_m2 + (vis_std2 if has_error_bar else 0) + 0.05, 1.15)
            ax_b2.text(text_x2, 0, f"{m2_raw:.2f}", va='center', fontsize=9)
            ax_b2.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_b2.transAxes, clip_on=False)
            
            # --- Rank ---
            ax_r = axes[i, base_col + 2]
            ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1)
            norm_val = (rank - 1) / (len(sorted_methods) - 1 + 1e-9)
            rect = plt.Rectangle((OFFSET, OFFSET), BOX_SIZE, BOX_SIZE, color=rank_cmap(norm_val), ec='white', lw=1)
            ax_r.add_patch(rect)
            ax_r.text(0.5, 0.5, str(rank), va='center', ha='center', fontsize=10)
            ax_r.set_axis_off()
            ax_r.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_r.transAxes, clip_on=False)
            
            # Separator
            ax_r.plot([1, 1], [0, 1], color='gray', lw=0.5, ls='--', transform=ax_r.transAxes, clip_on=False)

        # --- Final Overall Rank ---
        ax_final = axes[i, -1]
        ax_final.set_xlim(0, 1); ax_final.set_ylim(0, 1)
        final_r = int(row['final_rank'])
        norm_f = (final_r - 1) / (len(sorted_methods) - 1 + 1e-9)
        rect = plt.Rectangle((OFFSET, OFFSET), BOX_SIZE, BOX_SIZE, color=rank_cmap(norm_f), ec='white', lw=1)
        ax_final.add_patch(rect)
        ax_final.text(0.5, 0.5, str(final_r), va='center', ha='center', fontsize=11)
        ax_final.set_axis_off()
        ax_final.plot([0, 1], [0, 0], color='lightgray', lw=0.5, transform=ax_final.transAxes, clip_on=False)

    # --- 4. Custom Headers ---
    ROW1_Y = 1.7   
    ROW2_Y = 1.15  
    LINE_Y = 1.45  
    METHOD_Y = (ROW1_Y + ROW2_Y) / 2
    
    # 1. Method Header
    ax_m0 = axes[0, 0]
    ax_m0.text(0.5, METHOD_Y, "Method", ha='center', va='center', transform=ax_m0.transAxes, fontsize=11)
    
    # 2. Type Headers
    for t_idx, dtype in enumerate(include_types):
        base_col = 1 + (t_idx * 3)
        labels = TYPE_CONFIG[dtype]['labels']
        
        ax_b1 = axes[0, base_col]
        ax_b2 = axes[0, base_col+1]
        ax_r  = axes[0, base_col+2]
        
        # Row 2 Titles
        ax_b1.text(0.5, ROW2_Y, labels[0], ha='center', va='center', transform=ax_b1.transAxes, fontsize=10)
        ax_b2.text(0.5, ROW2_Y, labels[1], ha='center', va='center', transform=ax_b2.transAxes, fontsize=10)
        ax_r.text(0.5, ROW2_Y, "Rank", ha='center', va='center', transform=ax_r.transAxes, fontsize=10)
        
        # Row 1 Title (Centered across Bar1+Bar2+Rank)
        display_name = TYPE_CONFIG[dtype]['title']
        
        total_group_width = W_BAR1 + W_BAR2 + W_RANK
        mid_point_abs = total_group_width / 2
        rel_pos_abs = mid_point_abs - W_BAR1
        header_x_pos = rel_pos_abs / W_BAR2
        
        ax_b2.text(header_x_pos, ROW1_Y, display_name, ha='center', va='center', transform=ax_b2.transAxes, fontsize=11, color='black')
        
        # Separators
        ax_b1.plot([0, 1], [LINE_Y, LINE_Y], color='gray', lw=0.5, transform=ax_b1.transAxes, clip_on=False)
        ax_b2.plot([0, 1], [LINE_Y, LINE_Y], color='gray', lw=0.5, transform=ax_b2.transAxes, clip_on=False)
        ax_r.plot([0, 1], [LINE_Y, LINE_Y], color='gray', lw=0.5, transform=ax_r.transAxes, clip_on=False)
        
        # Vertical Dashed Header Line
        ax_r.plot([1, 1], [1, 2.2], color='gray', lw=0.5, ls='--', transform=ax_r.transAxes, clip_on=False)

    # 3. Overall Header
    ax_fin = axes[0, -1]
    ax_fin.text(0.5, ROW1_Y, "Overall", ha='center', va='center', transform=ax_fin.transAxes, fontsize=11)
    ax_fin.text(0.5, ROW2_Y, "Rank", ha='center', va='center', transform=ax_fin.transAxes, fontsize=10)
    ax_fin.plot([0, 1], [LINE_Y, LINE_Y], color='gray', lw=0.5, transform=ax_fin.transAxes, clip_on=False)

    # --- Borders ---
    for ax in axes[0, :]:
        ax.plot([0, 1], [1, 1], color='black', lw=1, transform=ax.transAxes, clip_on=False)

    for ax in axes[-1, :]:
        ax.plot([0, 1], [0, 0], color='black', lw=1, transform=ax.transAxes, clip_on=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()