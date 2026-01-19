try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print(f"Warning: Visualization libraries failed to import: {e}")
    plt = None
    sns = None

from sklearn.manifold import TSNE
import numpy as np
import os

try:
    import umap
except ImportError:
    umap = None

def plot_latent_space(latent_vectors, labels=None, method='tsne', save_path='results/latent_space.png'):
    """
    Visualize latent space using t-SNE or UMAP.
    """
    if latent_vectors.shape[0] < 5:
        print("Skipping visualization for small dataset (< 5 samples).")
        return

    if method == 'tsne':
        n_samples = latent_vectors.shape[0]
        perplex = min(30, n_samples - 1) if n_samples > 1 else 1
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplex)
    elif method == 'umap':
        if umap is None:
            print("UMAP not installed, falling back to t-SNE")
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")
        
    embedding = reducer.fit_transform(latent_vectors)
    
    if plt is None:
        print("Skipping plot (matplotlib not available)")
        return

    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(scatter, label='Cluster/Class')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.6)
        
    plt.title(f'Latent Space Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_training_history(losses, save_path='results/loss_history.png'):
    if plt is None:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
