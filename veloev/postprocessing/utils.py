import numpy as np
from sklearn.metrics import pairwise_distances

def _get_indices_distances_from_dense_matrix(D, n_neighbors: int):
    sample_range = np.arange(D.shape[0])[:, None]
    indices = np.argpartition(D, n_neighbors - 1, axis=1)[:, :n_neighbors]
    indices = indices[sample_range, np.argsort(D[sample_range, indices])]
    distances = D[sample_range, indices]
    return indices, distances

def fill_in_neighbors_indices(adata):
    if 'X_pca' not in adata.obsm:
        raise ValueError("adata.obsm must contain 'X_pca' for neighbor indices calculation.")
    X = adata.obsm['X_pca']
    _distances = pairwise_distances(X, metric='euclidean')
    knn_indices, _ = _get_indices_distances_from_dense_matrix(_distances, n_neighbors=30)
    adata.uns["neighbors"]["indices"] = knn_indices