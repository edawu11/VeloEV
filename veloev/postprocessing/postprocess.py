import gc
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import scanpy as sc
import scvelo as scv
from scipy.sparse import issparse
from . import utils
from tqdm import tqdm
import pandas as pd


scv.settings.verbosity = 0


def check_save_summarize_info(
    info_dict: Dict[str, List[Any]], 
    save: bool = True, 
    file_path: Union[str, Path] = "benchmark_info.pkl"
) -> pd.DataFrame:
    """
    Validates, saves, and summarizes the benchmark configuration.
    Updated to support 'seq_depth' variations of dataset types.
    """
    
    print("\n" + "="*50)
    print("🔹 STEP 1: VALIDATION CHECK")
    print("="*50)

    # --- 1. Mandatory Keys Check ---
    required_keys = ['methods', 'datasets_name', 'tasks', 'k_fold']
    missing_keys = [key for key in required_keys if key not in info_dict]
    
    if missing_keys:
        raise ValueError(f"Error: The following required keys are missing: {missing_keys}")

    # --- 1.1 Optional Methods Map Check ---
    if 'methods_map' in info_dict:
        methods = info_dict['methods']
        m_map = info_dict['methods_map']
        
        if not isinstance(m_map, dict):
            raise ValueError("Error: 'methods_map' must be a dictionary.")
            
        missing_in_map = [m for m in methods if m not in m_map]
        if missing_in_map:
            raise ValueError(f"Error: The following methods are missing from 'methods_map': {missing_in_map}")
        
        print("✅ Optional Check: 'methods_map' correctly covers all methods.")

    # --- 2. Structure Check ---
    dataset_keys = [
        'datasets_name', 'tasks', 'k_fold', 
        'cluster_key', 'time_key', 
        'cell_type_transitions', 'time_transitions'
    ]
    
    present_ds_keys = [k for k in dataset_keys if k in info_dict]
    baseline_key = 'datasets_name'
    num_datasets = len(info_dict[baseline_key])

    for key in present_ds_keys:
        current_len = len(info_dict[key])
        if current_len != num_datasets:
            raise ValueError(
                f"Structure Error: Key '{key}' has length {current_len}, "
                f"but expected {num_datasets} (based on '{baseline_key}')."
            )

    # --- 3. Logic Check (UPDATED) ---
    names = info_dict['datasets_name']
    types = info_dict['tasks']
    
    # Defaults to None lists if optional keys are missing
    cluster_keys = info_dict.get('cluster_key', [None] * num_datasets)
    time_keys = info_dict.get('time_key', [None] * num_datasets)
    ct_trans_list = info_dict.get('cell_type_transitions', [None] * num_datasets)
    time_trans_list = info_dict.get('time_transitions', [None] * num_datasets)

    for i in range(num_datasets):
        ds_name = names[i]
        ds_type = types[i]
        
        c_key = cluster_keys[i]
        t_key = time_keys[i]
        ct_trans = ct_trans_list[i]
        t_trans = time_trans_list[i]
        
        err_prefix = f"Config Error ({ds_name}, Task: {ds_type})"

        # --- Updated Validation Logic ---
        
        # 1. Directional Types (Standard + Seq Depth)
        if ds_type in ['directional', 'seq_depth_directional']:
            if c_key is None: raise ValueError(f"{err_prefix}: Missing 'cluster_key'.")
            if ct_trans is None: raise ValueError(f"{err_prefix}: Missing 'cell_type_transitions'.")

        # 2. Temporal Types (Standard + Seq Depth)
        elif ds_type in ['temporal', 'seq_depth_temporal']:
            if t_key is None: raise ValueError(f"{err_prefix}: Missing 'time_key'.")
            if t_trans is None: raise ValueError(f"{err_prefix}: Missing 'time_transitions'.")

        # 3. Hybrid Types (Standard + Seq Depth)
        elif ds_type in ['directional_temporal', 'seq_depth_directional_temporal']:
            if c_key is None: raise ValueError(f"{err_prefix}: Missing 'cluster_key'.")
            if ct_trans is None: raise ValueError(f"{err_prefix}: Missing 'cell_type_transitions'.")
            if t_key is None: raise ValueError(f"{err_prefix}: Missing 'time_key'.")
            if t_trans is None: raise ValueError(f"{err_prefix}: Missing 'time_transitions'.")
            
        # Optional: Warn if unknown type, or strictly enforce valid types
        # else:
        #     print(f"Warning: Unknown dataset type '{ds_type}' for {ds_name}. No specific checks applied.")

    print("✅ Validation Successful: Configuration is valid.")

    # --- 4. Save Logic ---
    if save:
        file_path = Path(file_path)
        if file_path.parent != Path('.'):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        try:
            with open(file_path, "wb") as f:
                pickle.dump(info_dict, f)
            print(f"💾 File Saved: {file_path}")
        except Exception as e:
            print(f"❌ Save Failed: {e}")
            raise e

    # --- 5. Summary Generation ---
    print("\n" + "="*50)
    print("🔹 STEP 2: SUMMARY REPORT")
    print("="*50)
    
    methods = info_dict['methods']
    
    if 'methods_map' in info_dict:
        m_map = info_dict['methods_map']
        display_methods = [f"{m} ({m_map[m]})" for m in methods]
    else:
        display_methods = methods

    print(f"• Total Methods:  {len(methods)}")
    print(f"  └─ {', '.join(display_methods)}")
    print(f"• Total Datasets: {num_datasets}")

    type_map = {}
    for name, dtype in zip(names, types):
        if dtype not in type_map:
            type_map[dtype] = []
        type_map[dtype].append(name)

    table_data = []
    for dtype, names_list in type_map.items():
        table_data.append({
            "Dataset Task": dtype,
            "Dataset Names": ", ".join(names_list),
            "Count": len(names_list)
        })

    summary_df = pd.DataFrame(table_data)
    summary_df = summary_df.sort_values(by="Count", ascending=False).reset_index(drop=True)

    print("\n📋 Dataset Summary:")
    # print(summary_df) # Uncomment if you want to print the df directly
    
    return summary_df
       


def run_postprocessing(benchmark_info: dict, base_dir: str = './', n_jobs: int = 8):
    """
    Runs the post-processing step for all datasets defined in benchmark_info.

    Args:
        benchmark_info (dict): The benchmark configuration dictionary.
        base_dir (str): The base directory where dataset results are stored. Defaults to './'.
        n_jobs (int): Number of cores to use for processing. Defaults to 20.
    """
    
    datasets = benchmark_info.get('datasets_name', [])
    num_datasets = len(datasets)
    
    if num_datasets == 0:
        print("No datasets found in benchmark_info.")
        return

    print(f"🚀 Starting post-processing for {num_datasets} datasets...")
    
    # Initialize tqdm progress bar
    pbar = tqdm(range(num_datasets), unit="dataset")
    
    for i in pbar:
        ds_name = datasets[i]
        
        # Update progress bar description
        pbar.set_description(f"Processing {ds_name}")
        
        # Construct the result path (e.g., ./01_bone_marrow/)
        result_path = Path(base_dir) / ds_name
        
        try:
            postprocess(
                methods=benchmark_info['methods'],
                task=benchmark_info['tasks'][i],
                cluster_key=benchmark_info['cluster_key'][i],
                time_key=benchmark_info['time_key'][i],
                k_fold=benchmark_info['k_fold'][i],
                basis='umap',
                result_path=result_path,
                n_jobs=n_jobs
            )
            # tqdm.write(f"✅ Successfully processed {ds_name}.")
        except Exception as e:
            # Use tqdm.write so the error message doesn't break the progress bar layout
            tqdm.write(f"❌ Error processing {ds_name}: {str(e)}")
            
    print("\n✅ Post-processing completed.")


     

def postprocess(
    methods: List[str],
    task: str,
    k_fold: int,
    cluster_key: Optional[str] = None,
    time_key: Optional[str] = None,
    basis: str = 'umap',
    result_path: Union[str, Path] = './',
    n_jobs: int = 8,
    scale: float = 30.0
) -> None:
    """
    Post-processes for RNA velocity methods.

    This function iterates through specified methods and cross-validation folds,
    loading the resulting AnnData objects. It computes velocity graphs, embeddings,
    pseudotime, and transition matrices based on the task (e.g., specific
    flags for directional consistency or temporal precision). The processed results
    are saved as pickle files for downstream evaluation.

    Args:
        methods (List[str]): A list of method names to process (e.g., ['scvelo', 'unitvelo']).
        task (str): evaluation task. Determines which metrics 
            are computed. Options include:
            - 'directional': Computes velocity graph and embeddings.
            - 'temporal': Computes pseudotime.
            - 'negative_control': Computes transition matrices.
            - 'directional_temporal': Computes both Directional and Temporal metrics.
            - 'simulation': Handles ground truth velocity comparison.
            - 'seq_depth_directional': Directional metrics with sequencing depth variation.
            - 'seq_depth_temporal': Temporal metrics with sequencing depth variation.
            - 'seq_depth_directional_temporal': Both Directional and Temporal metrics with sequencing depth variation.
        k_fold (int): The number of cross-validation folds. Set to 0 for full data processing.
        cluster_key (Optional[str], optional): Key in `adata.obs` storing cell type/cluster labels.
            Defaults to None.
        time_key (Optional[str], optional): Key in `adata.obs` storing ground truth time labels.
            Defaults to None.
        basis (str, optional): The embedding basis to use for visualization (e.g., 'umap', 'pca'). 
            Defaults to 'umap'.
        result_path (Union[str, Path], optional): Path to the directory containing raw results.
            Defaults to './'.
        n_jobs (int, optional): Number of parallel jobs for velocity graph computation. 
            Defaults to 8.
        scale (float, optional): Scale factor for transition matrix computation. 
            Defaults to 30.0.

    Raises:
        ValueError: If neither `cluster_key` nor `time_key` is provided.
        ValueError: If the provided keys do not exist in `adata.obs`.

    Returns:
        None: Results are saved to disk in the '{result_path}/postprocess' directory.
    """
    
    # Ensure result_path is a Path object
    result_path = Path(result_path)
    save_dir = result_path / "postprocess"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate velocity keys (e.g., 'scvelo_velocity')
    vkey_list = [f"{method}_velocity" for method in methods]

    # Set processing flags based on task
    flag_d = task in ['directional', 'directional_temporal','seq_depth_directional','seq_depth_directional_temporal']
    flag_t = task in ['temporal', 'directional_temporal','seq_depth_temporal','seq_depth_directional_temporal']
    flag_n = task == 'negative_control'
    flag_s = task == 'simulation'

    def _extract_and_save(method: str, vkey: str, fold: Union[int, str]):
        """Internal helper to process a single method and fold."""
        file_name = f"adata_run_{method}_{fold}.h5ad"
        file_path = result_path / file_name

        if not file_path.exists():
            warnings.warn(f"File {file_path} not found. Skipping.")
            return

        adata = sc.read_h5ad(file_path)

        # 1. Preprocessing for specific methods
        if method in ['dynamo_m2', 'velvetvae']:
            adata.layers['spliced'] = adata.layers['total'].copy()
            adata.layers['unspliced'] = adata.layers['new'].copy()
            scv.pp.moments(adata, n_neighbors=30, n_pcs=30)

        # 2. Standardize Velocity Layer (Sparse -> Dense Float32)
        # Note: 'toarray()' can be memory intensive for large datasets.
        if vkey in adata.layers:
            if issparse(adata.layers[vkey]):
                adata.layers[vkey] = adata.layers[vkey].toarray().astype(np.float32)
            adata.layers[vkey] = np.nan_to_num(adata.layers[vkey], nan=0.0).astype(np.float32)
            gc.collect()
        
        # 3. Standardize True Velocity Layer (if simulation)
        if flag_s and "true_velocity" in adata.layers:
            if issparse(adata.layers["true_velocity"]):
                adata.layers["true_velocity"] = adata.layers["true_velocity"].toarray().astype(np.float32)
            adata.layers["true_velocity"] = np.nan_to_num(adata.layers["true_velocity"], nan=0.0).astype(np.float32)
            gc.collect()

        # 4. Compute Velocity Graph
        # When calculating CBDir, unitvelo and cell2fate use the graph settings defined in their 
        # original papers or their own graph formulations; otherwise, scv.tl.velocity_graph 
        # is used to calculate the graph with default settings.
        if method in ["unitvelo_uni", "unitvelo_ind"] and flag_d:
            scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=True, n_jobs=n_jobs, show_progress_bar=False)
        elif method == "cell2fate" and flag_d:
            adata.uns[f'{method}_velocity_graph'] = adata.uns['Velocity_graph']
        else:
            scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=False, n_jobs=n_jobs, show_progress_bar=False)
        gc.collect()

        # 5. Compute Velocity Embedding
        if flag_d:
            scv.tl.velocity_embedding(adata, vkey=vkey, basis=basis)
            
            if "neighbors" in adata.uns:
                if "indices" not in adata.uns['neighbors'] or \
                   adata.uns['neighbors']['indices'].shape[0] != adata.n_obs:
                   if "indices" in adata.uns['neighbors']:
                       del adata.uns['neighbors']['indices']
                   utils.fill_in_neighbors_indices(adata)
            gc.collect()

        # 6. Compute Pseudotime
        if flag_t or flag_s:
            # Ensure layers are dense for pseudotime calculation stability
            for key in adata.layers.keys():
                if issparse(adata.layers[key]):
                    adata.layers[key] = adata.layers[key].toarray().astype(np.float32)
            
            if f"{method}_time" not in adata.obs:
                try:
                    scv.tl.velocity_pseudotime(adata, vkey=vkey)
                except Exception as e:
                    warnings.warn(f"Failed to compute pseudotime for {method}: {e}")

            if "indices" not in adata.uns['neighbors']:
                utils.fill_in_neighbors_indices(adata)
            gc.collect()

        # 7. Compute Transition Matrix (Negative Control)
        trans_mat = None
        if flag_n and f"{vkey}_graph" in adata.uns:
            trans_mat = scv.tl.transition_matrix(
                adata, 
                scale=scale, 
                vgraph=adata.uns[f"{vkey}_graph"],
                vkey=vkey, 
                self_transitions=False
            )
            gc.collect()

        # 8. Determine Cell Labels
        if cluster_key:
            if cluster_key not in adata.obs:
                raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs.")
            cell_label = adata.obs[cluster_key].astype(str)
        elif time_key:
            if time_key not in adata.obs:
                raise ValueError(f"time_key '{time_key}' not found in adata.obs.")
            warnings.warn("No cluster_key provided. Using time_key for cell labels.")
            cell_label = adata.obs[time_key].astype(str)
        else:
            raise ValueError("At least one of cluster_key or time_key must be provided.")

        # 9. Pack and Save Results
        result_dict = {
            "velocity_mat": adata.layers.get(vkey) if (flag_d or flag_s) else None,
            "true_velocity_mat": adata.layers.get("true_velocity") if flag_s else None,
            "velocity_emb": adata.obsm.get(f"{vkey}_{basis}") if flag_d else None,
            "exp_emb": adata.obsm.get(f"X_{basis}") if flag_d else None,
            "velocity_graph": adata.uns.get(f"{vkey}_graph") if flag_d else None,
            "cell_label": cell_label,
            "neighbor_indices": adata.uns.get('neighbors', {}).get('indices') if flag_d else None,
            "method_time": adata.obs.get(f"{method}_time") if (flag_t or flag_s) else None,
            "pseudo_time": adata.obs.get(f"{vkey}_pseudotime") if (flag_t or flag_s) else None,
            "time_label": adata.obs.get(time_key) if (time_key and (flag_t or flag_s)) else None,
            "self_trans": adata.obs.get(f'{vkey}_self_transition') if flag_n else None,
            "trans_mat": trans_mat if flag_n else None
        }

        save_file = save_dir / f"{method}_{fold}.pkl"
        with open(save_file, "wb") as file:
            pickle.dump(result_dict, file)

        # Cleanup
        del result_dict
        del adata
        if trans_mat is not None:
            del trans_mat
        gc.collect()
        
        # print(f"======= Finished processing {method} fold {fold} =======")

    # --- Main Execution Loop ---
    for method, vkey in zip(methods, vkey_list):
        if k_fold == 0:
            _extract_and_save(method, vkey, 'full')
        else:
            for fold in range(k_fold):
                _extract_and_save(method, vkey, fold)
    
    # print("======= Finish postprocessing all methods =======")
