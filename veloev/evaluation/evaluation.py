import gc
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.sparse import issparse, csr_matrix
from scipy.stats import pearsonr
from tqdm import tqdm


def run_benchmark_evaluation(benchmark_info: dict, base_dir: str = './'):
    """
    Runs the evaluation metrics for all datasets defined in benchmark_info.
    
    Updates:
    - Supports 'seq_depth_*' dataset types.
    - Automatically calculates stability score for seq_depth datasets after 
      base metrics (cbdir/tsc) are computed.
    """
    
    # 1. Define Metric Mapping
    TYPE_METRIC_MAP = {
        # Standard Types
        'directional': ['cbdir', 'icvcoh'],
        'directional_temporal': ['cbdir', 'icvcoh', 'cto', 'tsc'],
        'temporal': ['cto', 'tsc'],
        'negative_control': ['sts', 'nte'],
        'simulation': ['dcor', 'pr'],
        
        # Seq Depth Types (Stability Analysis)
        'seq_depth_directional': ['cbdir','icvcoh'],
        'seq_depth_temporal': ['cto','tsc'],
        'seq_depth_directional_temporal': ['cbdir','icvcoh','cto','tsc'],
    }

    # 2. Extract Data
    names = benchmark_info.get('datasets_name')
    types = benchmark_info.get('tasks')
    
    if names is None:
        raise KeyError("Could not find 'datasets_name' in benchmark_info")
        
    num_datasets = len(names)

    # 3. Pre-calculate Total Tasks for Progress Bar
    tasks = []
    for i in range(num_datasets):
        ds_type = types[i]
        ds_name = names[i]
        metrics = TYPE_METRIC_MAP.get(ds_type, [])
        for metric in metrics:
            tasks.append((i, ds_name, metric, ds_type))

    if not tasks:
        print("No metrics to process. Check 'datasets_type' definitions.")
        return

    print(f"🚀 Starting evaluation: {len(tasks)} metric calculations across {num_datasets} datasets.")

    # 4. Run Execution Loop
    pbar = tqdm(tasks, unit="metric")
    
    for i, ds_name, metric, ds_type in pbar:
        pbar.set_description(f"[{ds_name}] {metric}")
        
        # Construct Paths
        result_path = Path(base_dir) / ds_name
        
        try:
            # --- A. Calculate Base Metric ---
            single_metric(
                metric=metric,
                result_path=result_path,
                methods=benchmark_info.get('methods'),
                k_fold=benchmark_info['k_fold'][i],
                cell_type_transitions=benchmark_info['cell_type_transitions'][i],
                time_transitions=benchmark_info['time_transitions'][i]
            )
            
            # --- B. Calculate Stability Score (Seq Depth Only) ---
            # Logic: If it is a seq_depth dataset, we calculate stability 
            # immediately after obtaining the raw metric scores.
            if 'seq_depth' in ds_type and metric in ['cbdir', 'icvcoh', 'cto', 'tsc']:
                pbar.set_description(f"[{ds_name}] Stability ({metric})")
                calculate_stability_score(
                    metric=metric,
                    result_path=result_path,
                    methods=benchmark_info.get('methods')
                )
                
        except Exception as e:
            tqdm.write(f"❌ Error calculating {metric} for {ds_name}: {e}")

    print("\n✅ Evaluation completed.")


def keep_type(result_dict, nodes, target):
    return nodes[result_dict["cell_label"][nodes].values == target]

def single_metric(
    metric: str = 'cbdir',
    result_path: Union[str, Path] = './',
    methods: List[str] = ['velocyto', 'unitvelo_uni'],
    k_fold: int = 3,
    cell_type_transitions: Optional[List[Tuple[str, str]]] = None,
    time_transitions: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Wrapper function to calculate specific evaluation metrics for RNA velocity methods.

    This function acts as a dispatcher, calling the appropriate calculation function
    based on the `metric` argument.

    Args:
        metric (str, optional): The metric to calculate. Options:
            - 'cbdir': Cross-Boundary Directionality (requires `cell_type_transitions`).
            - 'icvcoh': In-Cluster Velocity Coherence.
            - 'cto': Cross-Boundary Time Order (requires `time_transitions`).
            - 'tsc': Temporal Spearman Correlation.
            - 'dcor': Distance Correlation.
            - 'pearson': Pearson Correlation.
            - 'nte': Normalized Entropy.
            - 'sts': Self-Transition Score.
            Defaults to 'cbdir'.
        post_path (Union[str, Path], optional): Path to directory with processed pickle files. 
            Defaults to './postprocess/'.
        evl_path (Union[str, Path], optional): Path to directory where results will be saved. 
            Defaults to './evaluation/'.
        methods (List[str], optional): List of method names to evaluate. 
            Defaults to ['velocyto', 'unitvelo_uni'].
        k_fold (int, optional): Number of cross-validation folds. Defaults to 3.
        cell_type_transitions (Optional[List[Tuple[str, str]]], optional): List of valid 
            lineage transitions (e.g., [('Stem', 'Progenitor')]). Required for 'cbdir'. 
            Defaults to None.
        time_transitions (Optional[List[Tuple[str, str]]], optional): List of transitions 
            ordered by time for 'cto' evaluation. Defaults to None.
        gt_method (str, optional): The prefix of the ground truth file (e.g. 'velocyto') 
            used for 'dcor'. Defaults to 'velocyto'.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated scores for the requested metric.
    
    Raises:
        ValueError: If required arguments for a specific metric are missing.
        NotImplementedError: If the metric name is not recognized.
    """
    
    result_path = Path(result_path)
    post_path = result_path / 'postprocess'
    evl_path = result_path / 'evaluation'
    evl_path.mkdir(parents=True, exist_ok=True)

    # Dispatch Logic
    if metric == 'cbdir':
        if cell_type_transitions is None:
            raise ValueError("Argument 'cell_type_transitions' is required for CBDir metric.")
        
        return calculate_cbdir(
            cell_type_transtions=cell_type_transitions,
            post_path=post_path,
            evl_path=evl_path,
            methods=methods,
            k_fold=k_fold
        )

    elif metric == 'icvcoh':
        return calculate_icvcoh(
            post_path=post_path,
            evl_path=evl_path,
            methods=methods,
            k_fold=k_fold
        )

    elif metric == 'cto':
        if time_transitions is None:
            raise ValueError("Argument 'time_transitions' is required for CTO metric.")
            
        return calculate_cto(
            time_transitions=time_transitions,
            methods=methods,
            post_path=post_path,
            evl_path=evl_path,
            k_fold=k_fold
        )

    elif metric == 'tsc':
            
        return calculate_tsc(
            methods=methods,
            post_path=post_path,
            evl_path=evl_path,
            k_fold=k_fold
        )

    elif metric == 'sts':
        return calculate_sts(
            methods=methods,
            post_path=post_path,
            evl_path=evl_path,
            k_fold=k_fold
        )
    elif metric == 'nte':
        return calculate_nte(
            methods=methods,
            post_path=post_path,
            evl_path=evl_path,
            k_fold=k_fold
        )
    elif metric == 'dcor':
        return calculate_dcor(
            result_path=result_path,
            methods=methods,
            post_path=post_path,
            evl_path=evl_path,
            k_fold=k_fold
        )

    elif metric == 'pr':
            
        return calculate_pr(
            methods=methods,
            post_path=post_path,
            evl_path=evl_path,
            k_fold=k_fold,
        )
    
    else:
        raise NotImplementedError(f"Metric '{metric}' is not currently supported.")

def calculate_cbdir(
    cell_type_transtions: List[Tuple[str, str]],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    methods: List[str],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Cross-Boundary Directionality (CBDir) score.

    CBDir measures how well the velocity vectors of cells at the boundary of a source 
    cluster point towards the target cluster in a known lineage trajectory.

    Args:
        cell_type_transtions (List[Tuple[str, str]]): A list of tuples defining valid transitions 
            between clusters (e.g., [('Stem', 'Progenitor')]).
        post_path (Union[str, Path]): Directory containing the post-processed pickle files.
        evl_path (Union[str, Path]): Directory where the result CSV will be saved.
        methods (List[str]): List of velocity methods to evaluate.
        k_fold (int): Number of folds used in cross-validation.
        save (bool, optional): Whether to save the resulting DataFrame to a CSV file. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame where rows represent methods and columns represent 
        scores for each fold.
    """
    
    cbdir_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute CBDir for a single method and fold."""
        file_path = post_path / f"{method}_{fold}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
        
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        x_emb = result_dict["exp_emb"]          # Expression embedding (e.g., UMAP)
        v_emb = result_dict["velocity_emb"]     # Velocity embedding
        cell_labels = np.array(result_dict["cell_label"]) # Cluster labels
        neighbor_indices = result_dict["neighbor_indices"]

        scores = {}

        # Iterate over each ground truth transition (Source Cluster -> Target Cluster)
        for u, v in cell_type_transtions:
            # Find indices of cells in the source cluster 'u'
            sel_indices = np.where(cell_labels == str(u))[0]
            
            if len(sel_indices) == 0:
                continue
            
            nbs = neighbor_indices[sel_indices] 
            x_points = x_emb[sel_indices]
            x_velocities = v_emb[sel_indices]

            type_score = []

            # Iterate through cells in source cluster
            for i in range(len(sel_indices)):
                current_nbs = nbs[i]
                
                # Filter neighbors: keep only those belonging to target cluster 'v'
                nb_labels = cell_labels[current_nbs]
                target_mask = (nb_labels == str(v))
                valid_nodes = current_nbs[target_mask]

                if len(valid_nodes) == 0:
                    continue

                # Calculate direction vector from current cell to valid neighbors
                position_dif = x_emb[valid_nodes] - x_points[i]
                
                # Reshape current cell's velocity vector
                curr_vel = x_velocities[i].reshape(1, -1)

                # Compute Cosine Similarity between velocity and neighbor direction
                dir_scores = cosine_similarity(position_dif, curr_vel).flatten()
                
                # Average score for this specific cell
                type_score.append(np.nanmean(dir_scores))

            # Average score for the specific transition (u -> v)
            if type_score:
                scores[(u, v)] = np.nanmean(type_score)
            else:
                scores[(u, v)] = np.nan

        # Compute overall mean score across all valid transitions for this fold
        if scores:
            mean_score = np.nanmean(list(scores.values()))
        else:
            mean_score = np.nan

        # Memory Cleanup
        del result_dict
        gc.collect()
        
        return mean_score

    # --- Main Execution Loop ---
    for method in methods:
        fold_cbdirs = []
        if k_fold == 0:
            score = _process_single_fold(method, 'full')
            fold_cbdirs.append(score)
        else:
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_cbdirs.append(score)

        # print(f"====== Finished calculating CBDir for method {method}. =======")
        cbdir_records[method] = fold_cbdirs
    
    # Create DataFrame and format output
    cbdir_df = pd.DataFrame(cbdir_records).T.reset_index()
    cbdir_df.rename(columns={'index': 'Method'}, inplace=True)

    if save:
        cbdir_df.to_csv(evl_path / "cbdir_df.csv", index=False)
        
    return cbdir_df

def calculate_icvcoh(
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    methods: List[str],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the In-Cluster Velocity Coherence (ICVCoh) score.

    ICVCoh evaluates the local smoothness of the velocity field. It measures the 
    average cosine similarity between a cell's velocity vector and the velocity 
    vectors of its neighbors *within the same cluster*. A higher score indicates 
    more consistent velocity directions within cell types.

    Args:
        post_path (Union[str, Path]): Directory containing the post-processed pickle files 
            (must contain 'velocity_emb', 'cell_label', 'neighbor_indices').
        evl_path (Union[str, Path]): Directory where the result CSV will be saved.
        methods (List[str]): List of velocity methods to evaluate (e.g., ['scvelo', 'deepvelo']).
        k_fold (int): Number of folds used in cross-validation. Set to 0 for full data.
        save (bool, optional): Whether to save the resulting DataFrame to a CSV file. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the ICVCoh scores for each method and fold.
    """
    
    # Ensure paths are Path objects
    
    icvcoh_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute ICVCoh for a single method and fold."""
        
        file_path = post_path / f"{method}_{fold}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
        
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        # Extract necessary data
        cell_labels = np.array(result_dict["cell_label"]) # Shape: (n_cells,)
        v_emb = result_dict["velocity_emb"]     # Shape: (n_cells, n_dims)
        neighbor_indices = result_dict["neighbor_indices"] # Shape: (n_cells, k_neighbors)
        
        # Handle NaNs in velocity embedding
        v_emb = np.nan_to_num(v_emb, nan=0.0)

        clusters = np.unique(cell_labels)
        cluster_scores = []

        # Iterate over each cell type (cluster)
        for cat in clusters:
            # Find indices of all cells in this cluster
            sel_indices = np.where(cell_labels == str(cat))[0]
            
            if len(sel_indices) == 0:
                continue

            # Get data for these cells
            cat_neighbors = neighbor_indices[sel_indices]
            cat_velocities = v_emb[sel_indices]
            
            # List to store cosine similarity scores for cells in this cluster
            cat_cell_scores = []

            # Iterate over each cell in the current cluster
            for i in range(len(sel_indices)):
                # Get neighbors for the current cell
                current_nbs = cat_neighbors[i]
                
                # Filter neighbors: keep only those in the SAME cluster (cat)
                # (This replaces the undefined 'keep_type' function)
                nb_labels = cell_labels[current_nbs]
                same_cluster_mask = (nb_labels == str(cat))
                valid_nodes = current_nbs[same_cluster_mask]

                if len(valid_nodes) == 0:
                    continue
                
                # Current cell's velocity
                curr_vel = cat_velocities[i].reshape(1, -1)
                
                # Neighbors' velocities
                neighbor_vels = v_emb[valid_nodes]

                # Calculate Cosine Similarity
                # Result shape: (1, n_valid_neighbors)
                sims = cosine_similarity(curr_vel, neighbor_vels)
                
                # Average similarity for this cell
                cat_cell_scores.append(np.mean(sims))

            # Average score for the cluster
            if cat_cell_scores:
                cluster_scores.append(np.mean(cat_cell_scores))

        # Calculate the final score for this fold (average across all clusters)
        if cluster_scores:
            mean_score = np.mean(cluster_scores)
        else:
            mean_score = np.nan

        # Cleanup
        del result_dict
        gc.collect()
        
        return mean_score

    # --- Main Execution Loop ---
    for method in methods:
        fold_icvcohs = []
        if k_fold == 0:
            score = _process_single_fold(method, 'full')
            fold_icvcohs.append(score)
        else:
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_icvcohs.append(score)
        # print(f"====== Finished calculating ICVCoh for method {method}. =======")
        icvcoh_records[method] = fold_icvcohs

    # Create DataFrame
    icvcoh_df = pd.DataFrame(icvcoh_records).T.reset_index()
    icvcoh_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        icvcoh_df.to_csv(evl_path / "icvcoh_df.csv", index=False)
        
    return icvcoh_df

def calculate_cto(
    time_transitions: List[Tuple[str, str]],
    methods: List[str],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Cross-Boundary Time Order (CTO) score.

    CTO evaluates the temporal correctness of an inferred trajectory. For every
    ground truth transition (Source -> Target), it calculates the proportion of 
    cell pairs (one from Source, one from Target) where the Source cell has a 
    lower inferred pseudotime than the Target cell.

    Args:
        cell_type_transtions (List[Tuple[str, str]]): List of valid transitions 
            (e.g., [('Stage1', 'Stage2')]).
        methods (List[str]): List of methods to evaluate.
        post_path (Union[str, Path]): Directory containing processed pickle files.
        evl_path (Union[str, Path]): Directory to save evaluation results.
        k_fold (int): Number of cross-validation folds.
        save (bool, optional): Whether to save results to CSV. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing CTO scores.
    """
    
    cto_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute CTO for a single method and fold."""
        
        # Adjust path structure if needed. Assuming standard file naming:
        file_path = post_path / f"{method}_{fold}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
            
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)
        
        # 1. Get Inferred Time (Method Time or Pseudotime)
        if result_dict.get("method_time") is not None:
            method_time = result_dict["method_time"].astype(float).values
        elif result_dict.get("pseudo_time") is not None:
            method_time = result_dict["pseudo_time"].astype(float).values
        else:
            print(f"Warning: No time data found for {method}. Returning NaN.")
            return np.nan

        # Handle NaNs in inferred time
        method_time = np.nan_to_num(method_time, nan=0.0)

        # 2. Get Ground Truth Labels
        if "time_label" not in result_dict or result_dict["time_label"] is None:
             raise ValueError(f"Key 'time_label' missing in pickle data.")
             
        cell_labels = np.array(result_dict["time_label"].values)
        
        assert len(method_time) == len(cell_labels), "Length mismatch between time and labels."

        scores = {}

        # 3. Calculate Score for each Transition
        for u, v in time_transitions:
            # Masking to find cells belonging to source (u) and target (v)
            # Ensure u and v are cast to the same type as cell_labels if needed (usually str)
            mask_u = (cell_labels == str(u))
            mask_v = (cell_labels == str(v))
            
            # If labels are floats in data but strings in ground_truth, try casting:
            if not np.any(mask_u) and not isinstance(u, str):
                 mask_u = (cell_labels == u)
                 mask_v = (cell_labels == v)

            A_time = method_time[mask_u]
            B_time = method_time[mask_v]

            # Skip if either group is empty
            if len(A_time) == 0 or len(B_time) == 0:
                continue
            
            better_count = 0
            # To save memory, we can loop over the smaller group or use broadcasting if size permits.
            # Here we use broadcasting as usually cluster sizes are < 10k cells.
            if len(A_time) * len(B_time) < 1e8: # < 100 million pairs, safe for RAM
                 better_count = np.sum(B_time[:, None] > A_time[None, :])
                 aggregate_score = better_count / (len(A_time) * len(B_time))
            else:
                # Fallback to loop for massive groups to save RAM
                temp_scores = []
                for t_a in A_time:
                    temp_scores.append(np.mean(B_time > t_a))
                aggregate_score = np.mean(temp_scores)

            scores[(u, v)] = aggregate_score

        # 4. Final Aggregation
        if scores:
            fold_score = np.nanmean(list(scores.values()))
        else:
            fold_score = np.nan

        # Cleanup
        del result_dict
        gc.collect()
        
        return fold_score

    # --- Main Loop ---
    for method in methods:
        fold_ctos = []
        if k_fold == 0:
            score = _process_single_fold(method, 'full')
            fold_ctos.append(score)
        else:
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_ctos.append(score)
        # print(f"====== Finished calculating CTO for method {method}. =======")
        cto_records[method] = fold_ctos

    # Save Results
    cto_df = pd.DataFrame(cto_records).T.reset_index()
    cto_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        cto_df.to_csv(evl_path / "cto_df.csv", index=False)
        
    return cto_df

# def calculate_CTO(cell_type_transtions: List[Tuple[str, str]],
#                 methods: List[str],
#                 post_path: Union[str, Path],
#                 evl_path: Union[str, Path],
#                 k_fold: int,
#                 time_key: str,
#                 save=True) -> None:
#     """Calculate CTO scores for all methods and folds."""

#     cto_records = {}

#     for method in methods:
#         fold_ctos = []
#         for fold in range(k_fold):
#             print(f"=======Calculating CTO for method {method}, fold {fold}...=======")
#             if not (post_path / "processed"/ "post_process" / f"{method}_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
#             with open(post_path / "processed" / "post_process" / f"{method}_{fold}.pkl", "rb") as file:
#                 result_dict = pickle.load(file)
            
#             scores = {}
#             if result_dict["method_time"] is not None:
#                 method_time = result_dict["method_time"].astype(float).values
#             else:
#                 method_time = result_dict["pseudo_time"].astype(float).values
            
#             method_time = np.nan_to_num(method_time, nan=0.0)
            

#             cell_label = result_dict["time_label"].astype(float).values
#             assert len(method_time) == len(cell_label), "Length of method_time must match length of cell_label."

#             for u, v in cell_type_transtions:
#                 A_cell_scores = []
#                 A_cell_indices = np.flatnonzero(cell_label == u)
#                 B_cell_indices = np.flatnonzero(cell_label == v)
#                 B_time = method_time[B_cell_indices]

#                 for cell_idx in A_cell_indices:
#                     cell_time = method_time[cell_idx]
#                     cell_score = np.mean(B_time > cell_time)
#                     A_cell_scores.append(cell_score)
                
#                 if len(A_cell_scores) == 0:
#                     aggragate_score = 0
#                 else:
#                     aggragate_score = np.mean(A_cell_scores)

#                 scores[(u,v)] = aggragate_score

#             fold_ctos.append(np.nanmean(list(scores.values())))
#             del result_dict
#             gc.collect()
#             print(f"======Finished calculating CTO for method {method}, fold {fold}.=======")
#         cto_records[method] = fold_ctos

#     cto_df = pd.DataFrame(cto_records).T.reset_index()
#     cto_df.rename(columns={cto_df.columns[0]: 'Method'}, inplace=True)
#     if save:
#         cto_df.to_csv(evl_path / "cto_df.csv", index=False)
    
def calculate_tsc(
    methods: List[str],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Temporal Spearman Correlation (TSC) score.

    TSC measures the correlation between the inferred latent time (or pseudotime) 
    and the ground truth continuous time labels. A score closer to 1 (or -1) 
    indicates strong agreement with the true temporal progression.

    Args:
        methods (List[str]): List of velocity methods to evaluate.
        post_path (Union[str, Path]): Directory containing the processed pickle files.
        evl_path (Union[str, Path]): Directory where the result CSV will be saved.
        k_fold (int): Number of cross-validation folds. Set to 0 for full data.
        save (bool, optional): Whether to save the resulting DataFrame to CSV. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the Spearman correlation scores.
    """
    
    spearman_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute TSC for a single method and fold."""
        
        file_path = post_path / f"{method}_{fold}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
            
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        # 1. Retrieve Inferred Time
        # Prioritize 'method_time' (latent time), fallback to 'pseudo_time'
        if result_dict.get("method_time") is not None:
            mt_series = result_dict["method_time"]
        elif result_dict.get("pseudo_time") is not None:
            mt_series = result_dict["pseudo_time"]
        else:
            print(f"Warning: Neither method_time nor pseudo_time found for {method}. Returning NaN.")
            return np.nan

        # 2. Retrieve Ground Truth Time
        if result_dict.get("time_label") is None:
             raise ValueError(f"Key 'time_label' missing in pickle data.")
        tl_series = result_dict["time_label"]

        # 3. Align and Clean Data
        # Ensure both series have the same index and drop NaNs
        # (Assuming they are Pandas Series; if numpy arrays, converting to Series first is safer)
        if not isinstance(mt_series, pd.Series):
            mt_series = pd.Series(mt_series)
        if not isinstance(tl_series, pd.Series):
            tl_series = pd.Series(tl_series)

        # Create a temporary DataFrame to handle alignment and NaNs cleanly
        tmp = pd.concat([mt_series, tl_series], axis=1, keys=["method_time", "time_label"]).dropna()
        
        if tmp.empty:
            print(f"Warning: No valid overlapping data points for {method}. Returning NaN.")
            return np.nan

        method_time = tmp["method_time"].astype(float).values
        time_label  = tmp["time_label"].astype(float).values

        # 4. Calculate Spearman Correlation
        # We only care about the correlation coefficient (index 0)
        corr, _ = spearmanr(method_time, time_label)
        
        # Cleanup
        del result_dict
        gc.collect()
        
        return corr

    # --- Main Execution Loop ---
    for method in methods:
        fold_spearmans = []
        
        if k_fold == 0:
            # Handle the 'full' dataset case if k_fold is 0
            score = _process_single_fold(method, 'full')
            fold_spearmans.append(score)
        else:
            # Handle standard k-fold cross-validation
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_spearmans.append(score)
        # print(f"====== Finished calculating TSC for method {method}. =======")
        spearman_records[method] = fold_spearmans
    
    # Format Output
    tsc_df = pd.DataFrame(spearman_records).T.reset_index()
    tsc_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        tsc_df.to_csv(evl_path / "tsc_df.csv", index=False)
        
    return tsc_df


# def calculate_TSC(
#     methods: List[str],
#     post_path: Union[str, Path],
#     evl_path: Union[str, Path],
#     k_fold: int,
#     time_key: str,  # This key in adata.obs should correspond to the labels in cell_type_transtions
#     save: bool = True
# ) -> pd.DataFrame:
    
#     """Calculate temporal Spearman correlation for all methods and folds."""

#     spearman_records = {}
#     for method in methods:
#         fold_spearmans = []
#         for fold in range(k_fold):
#             print(f"=======Calculating Spearman for method {method}, fold {fold}...=======")
#             if not (post_path /  f"{method}_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
#             with open(post_path / f"{method}_{fold}.pkl", "rb") as file:

#                 result_dict = pickle.load(file)

#             if result_dict["method_time"] is not None:
#                 mt_series = result_dict["method_time"]
#             elif result_dict["pseudo_time"] is not None:
#                 mt_series = result_dict["pseudo_time"]
#             else:
#                 raise ValueError(f"Neither method_time nor pseudo_time found for method {method}, fold {fold}.")

#             tl_series = result_dict["time_label"]

#             tmp = pd.concat([mt_series, tl_series], axis=1, keys=["method_time", "time_label"]).dropna()
#             method_time = tmp["method_time"].astype(float).values
#             time_label  = tmp["time_label"].astype(float).values
#             corr, _ = spearmanr(method_time, time_label)
#             fold_spearmans.append(corr)
#             del result_dict
#             gc.collect()
#             print(f"======Finished calculating Spearman for method {method}, fold {fold}.=======")
#         spearman_records[method] = fold_spearmans
    
#     tsc_df = pd.DataFrame(spearman_records).T.reset_index()
#     tsc_df.rename(columns={tsc_df.columns[0]: 'Method'}, inplace=True)
#     if save:
#         tsc_df.to_csv(evl_path / "tsc_df.csv", index=False)


def calculate_sts(
    methods: List[str],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Self-Transition Score (STS).

    STS measures the stability of cell states by quantifying the probability of 
    cells transitioning to themselves (or remaining in the same state) in the 
    inferred transition matrix. A higher score typically indicates higher stability 
    or robustness against noise (depending on the experimental context).

    Args:
        methods (List[str]): List of velocity methods to evaluate.
        post_path (Union[str, Path]): Directory containing the processed pickle files.
        evl_path (Union[str, Path]): Directory where the result CSV will be saved.
        k_fold (int): Number of cross-validation folds. Set to 0 for full data.
        save (bool, optional): Whether to save the resulting DataFrame to CSV. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the mean self-transition scores.
    """
    
    sts_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute STS for a single method and fold."""
        
        file_path = post_path / f"{method}_{fold}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
            
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        # Check for self-transition data
        # Note: Depending on your post_processing, this key might vary. 
        # Ensure 'self_trans' was saved correctly in your previous steps.
        if result_dict.get("self_trans") is None:
             # Try fallback to computing from transition matrix if available
             if result_dict.get("trans_mat") is not None:
                 print(f"Info: 'self_trans' not pre-calculated. Extracting diagonal from transition matrix.")
                 # Diagonal elements represent self-transitions
                 self_trans_values = result_dict["trans_mat"].diagonal()
             else:
                 print(f"Warning: Self-transition data not found for {method}. Returning NaN.")
                 return np.nan
        else:
            self_trans_values = result_dict["self_trans"]

        # Handle NaNs in the data array
        self_trans_values = np.nan_to_num(self_trans_values, nan=0.0)

        # Calculate mean score
        mean_score = np.mean(self_trans_values)
        
        # Cleanup
        del result_dict
        gc.collect()
        
        return mean_score

    # --- Main Execution Loop ---
    for method in methods:
        fold_scores = []
        
        if k_fold == 0:
            # Handle 'full' dataset processing
            score = _process_single_fold(method, 'full')
            fold_scores.append(score)
        else:
            # Handle cross-validation folds
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_scores.append(score)
        # print(f"====== Finished calculating STS for method {method}. =======")
        sts_records[method] = fold_scores
    
    # Format Output
    sts_df = pd.DataFrame(sts_records).T.reset_index()
    sts_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        sts_df.to_csv(evl_path / "sts_df.csv", index=False)
        
    return sts_df


# def calculate_STS(
#     methods: List[str],
#     post_path: Union[str, Path],
#     evl_path: Union[str, Path],
#     k_fold: int,
#     save: bool = True
# ) -> pd.DataFrame:
#     """Calculate self-transition scores for all methods and folds."""
#     sts_records = {}
#     for method in methods:
#         fold_self_transitions = []
#         for fold in range(k_fold):
#             print(f"=======Calculating self-transition for method {method}, fold {fold}...=======")
#             if not (post_path / f"{method}_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
#             with open(post_path  / f"{method}_{fold}.pkl", "rb") as file:
#                 result_dict = pickle.load(file)
        
#             if result_dict["self_trans"] is None:
#                 raise ValueError(f"Self-transition data not found for method {method}, fold {fold}.")
#             mean_score = np.mean(result_dict["self_trans"])
#             fold_self_transitions.append(mean_score)
#             del result_dict
#             gc.collect()
#             print(f"======Finished calculating self-transition for method {method}, fold {fold}.=======")
#         sts_records[method] = fold_self_transitions
    
#     sts_df = pd.DataFrame(sts_records).T.reset_index()
#     sts_df.rename(columns={sts_df.columns[0]: 'Method'}, inplace=True)
#     if save:
#         sts_df.to_csv(evl_path / "sts_df.csv", index=False)

# def normalized_entropy_sparse(T):
#     if issparse(T):
#         T = T.tocsr()
#     else:
#         T = csr_matrix(T)
    
#     n, m = T.shape
#     entropies = []
    
#     for i in range(n):
#         row_data = T.data[T.indptr[i]:T.indptr[i+1]]
#         if len(row_data) == 0:
#             continue
        
#         k_i = len(row_data)
        
#         row_sum = np.sum(row_data)
#         if row_sum <= 0:
#             continue
        
#         prob_dist = row_data / row_sum
        
#         entropy = -np.sum(prob_dist * np.log(prob_dist))
        
#         uniform_entropy = np.log(k_i) if k_i > 1 else 0
        
#         if uniform_entropy > 0:
#             entropies.append(entropy / uniform_entropy)
    
#     if len(entropies) == 0:
#         return 1.0
    
#     return np.mean(entropies)


def normalized_entropy_sparse(T: Union[np.ndarray, csr_matrix]) -> float:
    """
    Calculates the mean normalized entropy of a transition matrix efficiently.
    
    This function measures the uncertainty of cell transitions. High entropy means 
    a cell has equal probability of transitioning to many neighbors (high uncertainty), 
    while low entropy means transitions are deterministic (low uncertainty).
    
    Args:
        T: The transition matrix (sparse or dense).
        
    Returns:
        float: The mean normalized entropy.
    """
    # 1. Ensure Sparse CSR Format
    if not issparse(T):
        T = csr_matrix(T)
    else:
        T = T.tocsr() # Ensure it's CSR for fast row slicing

    # 2. Vectorized Calculation (Much faster than looping rows)
    # T.data contains the non-zero probabilities.
    # We need to normalize rows to sum to 1 first (just in case they aren't).
    
    # Calculate row sums
    row_sums = np.array(T.sum(axis=1)).flatten()
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0 
    
    # Normalize the matrix (this is usually cheap for sparse matrices)
    # Note: If T is already row-normalized, this is redundant but safe.
    # For speed, if you are sure T is normalized, you can skip this block.
    # Here, we do manual broadcasting for sparse division
    rows, cols = T.nonzero()
    data = T.data
    # data / row_sums[rows] gives probability p_ij
    probs = data / row_sums[rows]
    
    # 3. Calculate Entropy: -Sum(p * log(p))
    # Filter out zeros for log calculation (log(0) is undefined)
    # In sparse structure, data is usually non-zero, but let's be safe
    valid_mask = probs > 0
    probs = probs[valid_mask]
    
    # Calculate -p * log(p) for all non-zero elements
    entropies = -probs * np.log(probs)
    
    # Sum entropies back to their respective rows
    # We use np.add.at for fast unbuffered summation based on row indices
    n_samples = T.shape[0]
    row_entropies = np.zeros(n_samples)
    np.add.at(row_entropies, rows[valid_mask], entropies)
    
    # 4. Calculate Normalization Factor (Uniform Entropy: log(k))
    # k is the number of non-zero transitions per row (degree)
    # T.getnnz(axis=1) gives the number of non-zero elements per row
    k_per_row = T.getnnz(axis=1)
    
    # log(k), avoid log(0) or log(1) which is 0
    uniform_entropies = np.zeros(n_samples)
    mask_k = k_per_row > 1
    uniform_entropies[mask_k] = np.log(k_per_row[mask_k])
    
    # 5. Normalized Entropy: H / H_uniform
    # Only consider rows where uniform_entropy > 0
    valid_rows = uniform_entropies > 0
    
    if not np.any(valid_rows):
        return 1.0 # Fallback if no valid transitions
    
    normalized_scores = row_entropies[valid_rows] / uniform_entropies[valid_rows]
    
    return np.mean(normalized_scores)


def calculate_nte(    
    methods: List[str],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Normalized Shannon Entropy (NTE) for trajectory inference methods.

    NTE evaluates the "peakedness" of the transition probability distribution. 
    Lower values indicate that the method predicts distinct, confident trajectories 
    (sparse transition matrix). Higher values indicate "blurry" or uncertain predictions 
    (uniform transition probabilities).

    Args:
        methods (List[str]): List of velocity methods to evaluate.
        post_path (Union[str, Path]): Directory containing the processed pickle files.
        evl_path (Union[str, Path]): Directory where the result CSV will be saved.
        k_fold (int): Number of cross-validation folds. Set to 0 for full data.
        save (bool, optional): Whether to save the resulting DataFrame to CSV. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the mean normalized entropy scores.
    """
    
    nte_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute NTE for a single method and fold."""
        
        file_path = post_path / f"{method}_{fold}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
            
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)
        
        # Check for Transition Matrix
        if result_dict.get("trans_mat") is None:
            print(f"Warning: Transition matrix not found for {method}. Returning NaN.")
            return np.nan

        trans_mat = result_dict["trans_mat"]

        # Compute Entropy using the optimized function
        score = normalized_entropy_sparse(trans_mat)
        
        # Cleanup
        del result_dict
        gc.collect()
        
        # print(f"====== Finished calculating normalized entropy for method {method}, fold {fold}. =======")
        return score

    # --- Main Execution Loop ---
    for method in methods:
        fold_scores = []
        
        if k_fold == 0:
            score = _process_single_fold(method, 'full')
            fold_scores.append(score)
        else:
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_scores.append(score)
        
        nte_records[method] = fold_scores
    
    # Format Output
    nte_df = pd.DataFrame(nte_records).T.reset_index()
    nte_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        nte_df.to_csv(evl_path / "nte_df.csv", index=False)
        
    return nte_df


# def calculate_nte(    
#     methods: List[str],
#     post_path: Union[str, Path],
#     evl_path: Union[str, Path],
#     k_fold: int,
#     save: bool = True
# ) -> pd.DataFrame:
#     """Calculate normalized entropy for all methods and folds."""
#     nte_records = {}
#     for method in methods:
#         fold_normalized_entropies = []
#         for fold in range(k_fold):
#             print(f"=======Calculating normalized entropy for method {method}, fold {fold}...=======")
#             if not (post_path / f"{method}_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
#             with open(post_path / f"{method}_{fold}.pkl", "rb") as file:

#                 result_dict = pickle.load(file)
            
#             if result_dict["trans_mat"] is None:
#                 raise ValueError(f"Transition matrix not found for method {method}, fold {fold}.")
#             fold_normalized_entropies.append(normalized_entropy_sparse(result_dict["trans_mat"]))
#             del result_dict
#             gc.collect()
#             print(f"======Finished calculating normalized entropy for method {method}, fold {fold}.=======")
#         nte_records[method] = fold_normalized_entropies
    
#     nte_df = pd.DataFrame(nte_records).T.reset_index()
#     nte_df.rename(columns={nte_df.columns[0]: 'Method'}, inplace=True)
#     if save:
#         nte_df.to_csv(evl_path / "normalized_entropy_df.csv", index=False)

def pairwise_distances_np(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """
    Squared Euclidean pair-wise distance matrix.

    Parameters
    ----------
    x, y : np.ndarray, shape (n_samples, n_features)
        If `y` is None, distances are computed within `x`;
        otherwise between `x` and `y`.

    Returns
    -------
    dist : np.ndarray, shape (n_samples_x, n_samples_y)
        Squared distance matrix.
    """
    x_norm = np.sum(x ** 2, axis=1, keepdims=True)          # (n_x, 1)
    if y is not None:
        y_norm = np.sum(y ** 2, axis=1, keepdims=True).T     # (1, n_y)
    else:
        y = x
        y_norm = x_norm.T                                    # (1, n_x)

    dist = x_norm + y_norm - 2.0 * np.dot(x, y.T)
    # to avoid negative distances due to numerical errors
    np.maximum(dist, 0, out=dist)
    return dist


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Distance correlation between two data matrices (NumPy implementation).

    Parameters
    ----------
    X, Y : np.ndarray, shape (n_samples, n_features_X / n_features_Y)

    Returns
    -------
    dcor : float
        Distance correlation in the range [0, 1].
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must have the same number of samples."
            f"X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]}"
        )

    n = X.shape[0]
    DX = pairwise_distances_np(X)          # (n, n)
    DY = pairwise_distances_np(Y)          # (n, n)

    J = np.eye(n) - np.ones((n, n)) / n    # Double-centering matrix

    RX = J @ DX @ J
    RY = J @ DY @ J

    covXY = np.sum(RX * RY) / (n * n)
    covX  = np.sum(RX * RX) / (n * n)
    covY  = np.sum(RY * RY) / (n * n)

    # avoid zero division
    if covX == 0 or covY == 0:
        return 0.0

    return covXY / np.sqrt(covX * covY)

# def calculate_dcor(
#     methods: List[str],
#     post_path: Union[str, Path],
#     evl_path: Union[str, Path],
#     k_fold: int,
#     save: bool = True) -> pd.DataFrame:
#     """Calculate distance correlation for all methods and folds."""

#     dcor_records = {}
#     for method in methods:
#         fold_distance_correlations = []
#         for fold in range(k_fold):
#             print(f"=======Calculating distance correlation for method {method}, fold {fold}...=======")
#             if not (post_path / f"velocyto_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"you need to run velocyto first to get the ground truth velocity matrix.")
#             if not (post_path / f"{method}_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
#             with open(post_path / f"velocyto_{fold}.pkl", "rb") as file:
#                 result_dict = pickle.load(file)
#             true_v = result_dict["true_velocity_mat"]
#             if true_v is None:
#                 raise ValueError(f"velocity ground truth not found for fold {fold}.")
#             if issparse(true_v):
#                 true_v = true_v.toarray().astype(np.float32)
#             del result_dict
#             gc.collect()
#             with open(post_path / f"{method}_{fold}.pkl", "rb") as file:
#                 result_dict = pickle.load(file)
#             if result_dict["velocity_mat"] is None:
#                 raise ValueError(f"Velocity matrix not found for method {method}, fold {fold}.")
#             v_mat = result_dict["velocity_mat"]
#             if issparse(v_mat):
#                 v_mat = v_mat.toarray().astype(np.float32)
#             if not (v_mat.shape[0] == true_v.shape[0]):
#                 raise ValueError(f"cell number mismatch for estimate_velocity and true_velocity for {method}, fold {fold}")
#             v_mat = np.nan_to_num(v_mat, nan=0.0)
#             true_v = np.nan_to_num(true_v, nan=0.0)
#             distance_corr = distance_correlation(v_mat, true_v)
#             fold_distance_correlations.append(distance_corr)
#             del result_dict
#             gc.collect()
#             print(f"======Finished calculating distance correlation for method {method}, fold {fold}.=======")
#         dcor_records[method] = fold_distance_correlations
    
#     dcor_df = pd.DataFrame(dcor_records).T.reset_index()
#     dcor_df.rename(columns={dcor_df.columns[0]: 'Method'}, inplace=True)
#     if save:
#         dcor_df.to_csv(evl_path / "dcor_df.csv", index=False)

# def pairwise_distances_np(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
#     """
#     Computes Squared Euclidean pair-wise distance matrix.
    
#     Returns:
#         np.ndarray: Matrix of squared Euclidean distances.
#     """
#     x_norm = np.sum(x ** 2, axis=1, keepdims=True)
#     if y is not None:
#         y_norm = np.sum(y ** 2, axis=1, keepdims=True).T
#     else:
#         y = x
#         y_norm = x_norm.T

#     # Squared Euclidean Distance: |x|^2 + |y|^2 - 2<x, y>
#     dist = x_norm + y_norm - 2.0 * np.dot(x, y.T)
    
#     # Numerical stability: clamp negative values to 0
#     np.maximum(dist, 0, out=dist)
#     return dist


# def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
#     """
#     Calculates the Distance Correlation (dCor) between two matrices.
    
#     Reference: Szekely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
    
#     Args:
#         X, Y: Input matrices (n_samples, n_features).
        
#     Returns:
#         float: Distance correlation statistic in [0, 1].
#     """
#     if X.shape[0] != Y.shape[0]:
#         raise ValueError(f"X and Y must have same samples. Got {X.shape[0]}, {Y.shape[0]}")

#     n = X.shape[0]
    
#     # 1. Compute Pairwise Distances
#     # NOTE: dCor requires Euclidean Distance, not Squared Euclidean.
#     DX_sq = pairwise_distances_np(X)
#     DY_sq = pairwise_distances_np(Y)
    
#     DX = np.sqrt(DX_sq)
#     DY = np.sqrt(DY_sq)
    
#     # Cleanup squared matrices to save memory
#     del DX_sq, DY_sq
#     gc.collect()

#     # 2. Double Centering
#     # Formula: A_ij = d_ij - d_i. - d_.j + d_..
#     # Algebraic implementation is faster and uses less memory than matrix mult (J @ D @ J)
    
#     # Means
#     mu_X = np.mean(DX)
#     mu_Y = np.mean(DY)
#     mu_X_row = np.mean(DX, axis=1, keepdims=True)
#     mu_X_col = np.mean(DX, axis=0, keepdims=True)
#     mu_Y_row = np.mean(DY, axis=1, keepdims=True)
#     mu_Y_col = np.mean(DY, axis=0, keepdims=True)

#     # Centered matrices
#     RX = DX - mu_X_row - mu_X_col + mu_X
#     RY = DY - mu_Y_row - mu_Y_col + mu_Y
    
#     # Cleanup raw distance matrices
#     del DX, DY
#     gc.collect()

#     # 3. Compute Covariances (dCov^2 and dVar^2)
#     # The statistic is based on the average product of the centered matrices
#     dCov2_XY = np.mean(RX * RY)
#     dVar2_X  = np.mean(RX * RX)
#     dVar2_Y  = np.mean(RY * RY)

#     # 4. Compute Distance Correlation
#     if dVar2_X <= 0 or dVar2_Y <= 0:
#         return 0.0

#     # R^2 = dCov^2 / sqrt(dVar^2 * dVar^2)
#     dCor_sq = dCov2_XY / np.sqrt(dVar2_X * dVar2_Y)
    
#     # Return R (Correlation), ensure valid range
#     return np.sqrt(np.clip(dCor_sq, 0.0, 1.0))


def calculate_dcor(
    result_path: Union[str, Path],
    methods: List[str],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculate Distance Correlation (dCor) between estimated velocity and ground truth.
    
    Args:
        methods: List of method names.
        post_path: Path to processed pickle files.
        evl_path: Path to save evaluation results.
        k_fold: Number of folds (0 for full data).
        save: Save results to CSV.
    """
    
    dcor_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute dCor for a single method and fold."""
        
        # 1. Load Ground Truth
        
        gt_file_path = result_path / f"ground_truth_velocity_{fold}.pkl"
        if not gt_file_path.exists():
            raise FileNotFoundError(f"Ground truth data for fold {fold} not found at {gt_file_path}.")
    
        with open(gt_file_path, "rb") as file:
            true_v = pickle.load(file)

        if issparse(true_v):
            true_v = true_v.toarray().astype(np.float32)
        true_v = np.nan_to_num(true_v, nan=0.0)
        
        gc.collect()

        # 2. Load Method Estimation
        method_file_path = post_path / f"{method}_{fold}.pkl"
        if not method_file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
            
        with open(method_file_path, "rb") as file:
            method_dict = pickle.load(file)
            
        v_mat = method_dict.get("velocity_mat")
        if v_mat is None:
            raise ValueError(f"Velocity matrix not found for method {method}, fold {fold}.")

        if issparse(v_mat):
            v_mat = v_mat.toarray().astype(np.float32)
        v_mat = np.nan_to_num(v_mat, nan=0.0)

        # 3. Check Consistency
        if v_mat.shape[0] != true_v.shape[0]:
            raise ValueError(
                f"Cell number mismatch: Method {v_mat.shape[0]} vs GT {true_v.shape[0]} "
                f"for {method}, fold {fold}."
            )

        # 4. Compute Distance Correlation
        score = distance_correlation(v_mat, true_v)
        
        # Cleanup
        del method_dict
        del v_mat
        del true_v
        gc.collect()
        
        return score

    # --- Main Execution Loop ---
    for method in methods:
        fold_scores = []
        
        if k_fold == 0:
            score = _process_single_fold(method, 'full')
            fold_scores.append(score)
        else:
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_scores.append(score)
        # print(f"====== Finished calculating dCor for method {method}. =======")
        dcor_records[method] = fold_scores
    
    # Save Results
    dcor_df = pd.DataFrame(dcor_records).T.reset_index()
    dcor_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        dcor_df.to_csv(evl_path / "dcor_df.csv", index=False)
        
    return dcor_df



def calculate_pr(
    methods: List[str],
    post_path: Union[str, Path],
    evl_path: Union[str, Path],
    k_fold: int,
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Pearson correlation between inferred time and ground truth time.

    Pearson correlation measures the linear correlation between the inferred 
    latent time (or pseudotime) and the true continuous time labels.

    Args:
        methods (List[str]): List of velocity methods to evaluate.
        post_path (Union[str, Path]): Directory containing processed pickle files.
        evl_path (Union[str, Path]): Directory where the result CSV will be saved.
        k_fold (int): Number of cross-validation folds. Set to 0 for full data.
        save (bool, optional): Whether to save the resulting DataFrame to CSV. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the Pearson correlation scores.
    """
    
    pr_records = {}

    def _process_single_fold(method: str, fold: Union[int, str]) -> float:
        """Internal helper to compute Pearson correlation for a single method/fold."""
        
        file_path = post_path / f"{method}_{fold}.pkl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data for {method} fold {fold} not found at {file_path}")
            
        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)
        
        # 1. Retrieve Inferred Time
        if result_dict.get("method_time") is not None:
            mt_series = result_dict["method_time"]
        elif result_dict.get("pseudo_time") is not None:
            mt_series = result_dict["pseudo_time"]
        else:
            print(f"Warning: Neither method_time nor pseudo_time found for {method}. Returning NaN.")
            return np.nan

        # 2. Retrieve Ground Truth Time
        if result_dict.get("time_label") is None:
             raise ValueError(f"Key 'time_label' missing in pickle data.")
        tl_series = result_dict["time_label"]

        # 3. Clean and Align Data
        # Ensure inputs are Series for safe concatenation
        if not isinstance(mt_series, pd.Series):
            mt_series = pd.Series(mt_series)
        if not isinstance(tl_series, pd.Series):
            tl_series = pd.Series(tl_series)

        # Align indices and drop NaNs
        tmp = pd.concat([mt_series, tl_series], axis=1, keys=["method_time", "time_label"]).dropna()
        
        if tmp.empty:
            print(f"Warning: No valid overlapping data points for {method}. Returning NaN.")
            return np.nan

        method_time = tmp["method_time"].astype(float).values
        time_label  = tmp["time_label"].astype(float).values

        # 4. Calculate Pearson Correlation
        # pearsonr returns (statistic, p-value), we only need statistic [0]
        corr, _ = pearsonr(method_time, time_label)
        
        # Cleanup
        del result_dict
        gc.collect()
        
        return corr

    # --- Main Execution Loop ---
    for method in methods:
        fold_pearsons = []
        
        if k_fold == 0:
            score = _process_single_fold(method, 'full')
            fold_pearsons.append(score)
        else:
            for fold in range(k_fold):
                score = _process_single_fold(method, fold)
                fold_pearsons.append(score)

        # print(f"====== Finished calculating Pearson correlation for method {method}. =======")
        pr_records[method] = fold_pearsons
    
    # Format and Save
    pr_df = pd.DataFrame(pr_records).T.reset_index()
    pr_df.rename(columns={'index': 'Method'}, inplace=True)
    
    if save:
        evl_path.mkdir(parents=True, exist_ok=True)
        pr_df.to_csv(evl_path / "pr_df.csv", index=False)
        
    return pr_df


# def calculate_pearson(
#         ground_truth_velocity: np.ndarray,
#         methods: List[str],
#         post_path: Union[str, Path],
#         evl_path: Union[str, Path],
#         k_fold: int,
#         save: bool = True
#     ) -> pd.DataFrame:
#     """Calculate Pearson correlation for all methods and folds."""
    
#     pr_records = {}
#     for method in methods:
#         fold_pearsons = []
#         for fold in range(k_fold):
#             print(f"=======Calculating Pearson for method {method}, fold {fold}...=======")
#             if not (post_path / f"{method}_{fold}.pkl").exists():
#                 raise FileNotFoundError(f"Processed data for {method} fold {fold} not found.")
#             with open(post_path / f"{method}_{fold}.pkl", "rb") as file:
#                 result_dict = pickle.load(file)
            
#             if result_dict["method_time"] is not None:
#                 mt_series = result_dict["method_time"]
#             elif result_dict["pseudo_time"] is not None:
#                 mt_series = result_dict["pseudo_time"]
#             else:
#                 raise ValueError(f"Neither method_time nor pseudo_time found for method {method}, fold {fold}.")

#             tl_series = result_dict["time_label"]

#             tmp = pd.concat([mt_series, tl_series], axis=1, keys=["method_time", "time_label"]).dropna()
#             method_time = tmp["method_time"].astype(float).values
#             time_label  = tmp["time_label"].astype(float).values
#             corr, _ = pearsonr(method_time, time_label)
#             fold_pearsons.append(corr)
#             del result_dict
#             gc.collect()
#             print(f"======Finished calculating Pearson for method {method}, fold {fold}.=======")
#         pr_records[method] = fold_pearsons
    
#     pr_df = pd.DataFrame(pr_records).T.reset_index()
#     pr_df.rename(columns={pr_df.columns[0]: 'Method'}, inplace=True)
#     if save:

#         pr_df.to_csv(evl_path / "pr_df.csv", index=False)


def calculate_stability_score(
    metric: str,
    result_path: Union[str, Path],
    methods: List[str],
    save: bool = True
) -> pd.DataFrame:
    """
    Calculates the Stability Score (SS) for a given metric across cross-validation folds.

    The Stability Score is a composite metric that rewards methods for having 
    high average performance and low variance (standard deviation) across folds.
    It combines min-max scaled mean performance and inverted scaled standard deviation.

    Formula: SS = 0.5 * ( (Mean_Scaled)^2 + (1 - Std_Scaled)^2 )

    Args:
        metric (str): The name of the metric to evaluate (e.g., 'cbdir', 'cto'). 
            Expects a file named '{metric}_df.csv' in `evl_path`.
        methods (List[str]): List of methods to include in the calculation.
        evl_path (Union[str, Path]): Directory containing the evaluation CSV files.
        save (bool, optional): Whether to save the resulting SS scores to a CSV. 
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the Method names and their calculated Stability Scores.
    """
    
    result_path = Path(result_path)
    evl_path = result_path / 'evaluation'
    evl_path.mkdir(parents=True, exist_ok=True)
    file_path = evl_path / f"{metric}_df.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file for metric '{metric}' not found at {file_path}")

    # 1. Load Data
    score_df = pd.read_csv(file_path)

    # 2. Filter DataFrame to include only requested methods
    # This ensures we respect the 'methods' input order and content
    score_df = score_df[score_df["Method"].isin(methods)].copy()
    
    # Check if we have all requested methods
    found_methods = set(score_df["Method"].unique())
    missing_methods = set(methods) - found_methods
    if missing_methods:
        print(f"Warning: The following methods were not found in {file_path.name}: {missing_methods}")

    if score_df.empty:
        raise ValueError("No matching methods found in the evaluation file.")

    # 3. Calculate Mean and Std across folds (assuming folds are columns 1 onwards)
    # iloc[:, 1:] selects all columns after 'Method'
    numeric_df = score_df.iloc[:, 1:]
    means = numeric_df.mean(axis=1).values
    stds = numeric_df.std(axis=1).values

    # 4. Min-Max Scaling
    # Handle edge case where max == min (e.g., only 1 method or identical scores)
    ptp_mean = np.ptp(means) # Peak-to-peak (max - min)
    ptp_std = np.ptp(stds)
    
    # Avoid division by zero if all values are identical
    denom_mean = ptp_mean if ptp_mean > 1e-9 else 1.0
    denom_std = ptp_std if ptp_std > 1e-9 else 1.0

    mean_scaled = (means - np.min(means)) / denom_mean
    std_scaled = (stds - np.min(stds)) / denom_std

    # 5. Calculate Stability Score (SS)
    # Higher Mean is better; Lower Std is better (hence 1 - std_scaled)
    ss = 0.5 * ((mean_scaled)**2 + (1 - std_scaled)**2)

    # 6. Create Result DataFrame
    ss_df = pd.DataFrame({
        "Method": score_df["Method"].values,
        "SS": ss
    })
    
    # Sort by SS descending (optional, but usually helpful)
    ss_df = ss_df.sort_values(by="SS", ascending=False)

    if save:
        ss_df.to_csv(evl_path / f"s_{metric}_df.csv", index=False)
        
    return ss_df



# def calculate_stability_score(
#         metric: str,
#         result_path: Union[str, Path],
#         methods: List[str],
#         save: bool = True
#     ) -> pd.DataFrame:
#     """"Calculate directional stability scores for all methods"""
    
#     result_path = Path(result_path)
#     evl_path = result_path / 'evaluation'
#     evl_path.mkdir(parents=True, exist_ok=True)
#     score_df = pd.read_csv(evl_path / f"{metric}_df.csv")

#     # check whether the methods match
#     if not set(score_df["Method"]) == set(methods):
#         raise ValueError("Methods in score DataFrame do not match the provided methods. " \
#         "Please re-calculate the scores.")
#     methods = score_df["Method"].values
#     mean = score_df.iloc[:, 1:].mean(axis=1).values
#     std = score_df.iloc[:, 1:].std(axis=1).values

#     # min-max scale the mean and std
#     mean_scaled = (mean - np.min(mean)) / (np.max(mean) - np.min(mean) + 1e-8)
#     std_scaled = (std - np.min(std)) / (np.max(std) - np.min(std) + 1e-8)

#     # calculate SS scores
#     ss = 0.5 * ((mean_scaled)**2 + (1 - std_scaled)**2)

#     ss_df = pd.DataFrame({"Method": methods, "SS": ss})
#     if save:
#         ss_df.to_csv(evl_path / f"s_{metric}.csv", index=False)

# def calculate_stability_score(self, save=True) -> None:
#     """"Calculate directional stability scores for all methods"""
#     if not (self.flag_d or self.flag_t):
#         raise ValueError(f"stability score calculation is not enabled for this dataset type: {self.dataset_type}.")
#     if self.methods is None or len(self.methods) == 0:
#         raise ValueError("No methods provided for stability score calculation.")

#     # check whether the previous score output exists
#     if self.flag_d:
#         if not (self.file_path / "evaluation" / "cbdir_df.csv").exists():
#             raise FileNotFoundError("CBDir scores not found. Please calculate it first.")
#     else:
#         if not (self.file_path / "evaluation" / "spearman_df.csv").exists():
#             raise FileNotFoundError("spearman scores not found. Please calculate it first.")
    
#     # load CBDir or spearman scores
#     if self.flag_d:
#         score_df = pd.read_csv(self.file_path / "evaluation" / "cbdir_df.csv")
#     else:
#         score_df = pd.read_csv(self.file_path / "evaluation" / "spearman_df.csv")

#     # check whether the methods match
#     if not set(score_df["Method"]) == set(self.methods):
#         raise ValueError("Methods in score DataFrame do not match the provided methods. " \
#         "Please re-calculate the scores.")
#     methods = score_df["Method"].values
#     mean = score_df.iloc[:, 1:].mean(axis=1).values
#     std = score_df.iloc[:, 1:].std(axis=1).values

#     # min-max scale the mean and std
#     mean_scaled = (mean - np.min(mean)) / (np.max(mean) - np.min(mean) + 1e-8)
#     std_scaled = (std - np.min(std)) / (np.max(std) - np.min(std) + 1e-8)

#     # calculate SS scores
#     ss = 0.5 * ((mean_scaled)**2 + (1 - std_scaled)**2)

#     ss_df = pd.DataFrame({"Method": methods, "SS": ss})
#     if save:
#         output_path = self.file_path / "evaluation"
#         if self.flag_d:
#             ss_df.to_csv(output_path / "ssd_df.csv", index=False)
#         if self.flag_t:
#             ss_df.to_csv(output_path / "sst_df.csv", index=False)