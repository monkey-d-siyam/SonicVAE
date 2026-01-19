from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
import numpy as np

def evaluate_clustering(data, labels, true_labels=None):
    """
    Compute clustering metrics.
    
    Args:
        data: Latent features (numpy array)
        labels: Predicted cluster labels (numpy array)
        true_labels: Ground truth labels (optional)
        
    Returns:
        dict: Dictionary of metric names and values
    """
    metrics = {}
    
    # Internal validation metrics (Ground truth not required)
    if len(set(labels)) > 1: # Silhouette needs at least 2 clusters
        metrics['silhouette'] = float(silhouette_score(data, labels))
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(data, labels))
        metrics['davies_bouldin'] = float(davies_bouldin_score(data, labels))
    else:
        metrics['silhouette'] = -1.0
        metrics['calinski_harabasz'] = 0.0
        metrics['davies_bouldin'] = float('inf')

    # External validation metrics (Ground truth required)
    if true_labels is not None:
        metrics['ari'] = float(adjusted_rand_score(true_labels, labels))
        metrics['nmi'] = float(normalized_mutual_info_score(true_labels, labels))
        metrics['purity'] = float(compute_purity(true_labels, labels))
        
    return metrics

def compute_purity(y_true, y_pred):
    # Compute confusion matrix
    contingency_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
    
    # This is a simplified O(N^2) purity check for matching, 
    # but typical way is: sum(max(intersection)) / N
    
    # Using a faster way with sklearn contingency matrix if available, but manual for now:
    from sklearn.metrics.cluster import contingency_matrix
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)
