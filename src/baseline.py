import argparse
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.dataset import get_dataloader
from src.evaluation import evaluate_clustering
from src.visualization import plot_latent_space
import os

def run_baseline(args):
    # Load Data
    _, X_full, y_full = get_dataloader(synthetic=True)
    
    print(f"Running Baseline: PCA ({args.n_components} comps) + KMeans ({args.n_clusters} clusters)")
    
    # PCA
    pca = PCA(n_components=args.n_components)
    X_pca = pca.fit_transform(X_full)
    
    # Clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(X_pca)
    
    # Evaluation
    metrics = evaluate_clustering(X_pca, pred_labels, y_full)
    print("Baseline Metrics:")
    print(json.dumps(metrics, indent=2))
    
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Visualization
    plot_latent_space(X_pca, labels=pred_labels, method='tsne', save_path='results/baseline_clusters_pred.png')
    plot_latent_space(X_pca, labels=y_full, method='tsne', save_path='results/baseline_clusters_true.png')
    
    print("Baseline done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_components', type=int, default=10, help='PCA components')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    args = parser.parse_args()
    
    run_baseline(args)
