import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import get_dataloader
from src.vae import VAE, ConvVAE, HybridVAE, CVAE, ResNetVAE, loss_function
from src.clustering import Clusterer
from src.evaluation import evaluate_clustering
from src.visualization import plot_latent_space, plot_training_history

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_kl_beta(epoch, total_epochs, cyclique=False):
    # Monotonic annealing from 0 to 1
    if not cyclique:
        return min(1.0, (epoch / (total_epochs * 0.5)))
    else:
        # Cyclic: repeat 0->1 every N epochs
        cycle_len = total_epochs // 4
        return min(1.0, ((epoch % cycle_len) / (cycle_len * 0.5)))

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{args.model_type}")
    
    # Data Loading
    synthetic = args.data_dir is None
    if not synthetic:
        print(f"Loading real data from {args.data_dir}...")
    
    dataloader, X_full, y_full = get_dataloader(
        batch_size=args.batch_size, 
        synthetic=synthetic, 
        mode='resnet' if args.model_type == 'resnet' else args.model_type, 
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    n_classes = len(np.unique(y_full)) if y_full is not None else 0
    
    # Model Selection
    if args.conditional:
        if args.model_type != 'linear':
            raise ValueError("CVAE currently only implemented for linear mode")
        print(f"Initializing CVAE with {n_classes} classes")
        model = CVAE(input_dim=X_full.shape[1], n_classes=n_classes, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    elif args.model_type == 'linear':
        model = VAE(input_dim=X_full.shape[1], hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    elif args.model_type == 'conv':
        model = ConvVAE(input_channels=1, latent_dim=args.latent_dim).to(device)
    elif args.model_type == 'resnet':
        model = ResNetVAE(input_channels=1, latent_dim=args.latent_dim).to(device)
    elif args.model_type == 'hybrid':
        audio_dim = X_full[0].shape[1]
        lyrics_dim = X_full[1].shape[1]
        model = HybridVAE(audio_dim=audio_dim, lyrics_dim=lyrics_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Mixed Precision Scaler
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=args.patience)
    
    # Training Loop
    model.train()
    history = []
    
    print(f"Starting training ({args.model_type}, conditional={args.conditional}), AMP={use_amp}...")
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        current_beta = get_kl_beta(epoch, args.epochs) if args.anneal else args.beta
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            # Handle list/tuple for generic device moving
            if isinstance(data, list) or isinstance(data, tuple):
                data = [d.to(device) for d in data]
                data = tuple(data) # Hybrid needs tuple
            else:
                data = data.to(device)
            
            labels = labels.to(device).long()
                
            optimizer.zero_grad()
            
            # AMP Autocast
            with autocast(enabled=use_amp):
                if args.conditional:
                    recon_batch, mu, logvar = model(data, labels)
                else:
                    recon_batch, mu, logvar = model(data)
                    
                loss = loss_function(recon_batch, data, mu, logvar, beta=current_beta)
            
            if torch.isnan(loss):
                print(f"Loss is NaN at epoch {epoch}, batch {batch_idx}")
                break
                
                
            # Scaled Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # Step-level logging
            if batch_idx % 10 == 0:
                writer.add_scalar('Training/BatchLoss', loss.item(), epoch * len(dataloader) + batch_idx)
            
        avg_loss = epoch_loss / len(dataloader.dataset)
        history.append(avg_loss)
        
        # Epoch logging
        writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Parameters/Beta', current_beta, epoch)
        
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Beta: {current_beta:.3f}')
        
        # Check Early Stopping (using training loss here as proxy if no val set, usually split is better)
        # Ideally we'd have a validation set, but for synthetic logic we'll just check train convergence
        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
            
    # Save Model
    os.makedirs('results', exist_ok=True)
    suffix = f"{args.model_type}_cvae" if args.conditional else args.model_type
    torch.save(model.state_dict(), f'results/vae_model_{suffix}.pth')
    plot_training_history(history, save_path=f'results/loss_history_{suffix}.png')
    
    # Feature Extraction
    model.eval()
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            if isinstance(data, list) or isinstance(data, tuple):
                data = [d.to(device) for d in data]
                data = tuple(data)
            else:
                data = data.to(device)
            
            labels_device = labels.to(device).long()
                
            if args.conditional:
                _, mu, _ = model(data, labels_device)
            else:
                _, mu, _ = model(data)
                
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
            
    latent_data = np.concatenate(all_mu, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    
    print("Performing Clustering...")
    # Clustering
    if len(latent_data) < args.n_clusters:
        print(f"Warning: Not enough samples ({len(latent_data)}) for {args.n_clusters} clusters. Reducing clusters to {len(latent_data)}.")
        n_clusters = len(latent_data)
    else:
        n_clusters = args.n_clusters
        
    clusterer = Clusterer(method=args.clustering, n_clusters=n_clusters)
    pred_labels = clusterer.fit_predict(latent_data)
    
    # Evaluation
    metrics = evaluation_metrics = evaluate_clustering(latent_data, pred_labels, true_labels)
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    
    with open(f'results/metrics_{suffix}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Visualization
    plot_latent_space(latent_data, labels=pred_labels, method='tsne', save_path=f'results/latent_clusters_pred_{suffix}.png')
    plot_latent_space(latent_data, labels=true_labels, method='tsne', save_path=f'results/latent_clusters_true_{suffix}.png')
    
    # CVAE Special Visualization
    if args.conditional and args.model_type == 'linear':
        generate_cvae_samples(model, args.latent_dim, n_classes, device, save_path=f'results/cvae_generation.png')
        
    writer.close()
    print("Done! Results saved to 'results/' directory.")

def generate_cvae_samples(model, latent_dim, n_classes, device, save_path):
    model.eval()
    z = torch.randn(1, latent_dim).to(device)
    z = z.repeat(n_classes, 1)
    y = torch.arange(n_classes).to(device)
    
    with torch.no_grad():
        recon = model.decode(z, y).cpu().numpy()
        
    plt.figure(figsize=(12, 6))
    for i in range(n_classes):
        plt.subplot(n_classes, 1, i+1)
        plt.plot(recon[i])
        plt.title(f'Generated Class {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Music Clustering")
    parser.add_argument('--mode', type=str, default='train', choices=['train'], help='Mode: train')
    parser.add_argument('--model_type', type=str, default='linear', choices=['linear', 'conv', 'resnet', 'hybrid'], help='Model type')
    parser.add_argument('--conditional', action='store_true', help='Use Conditional VAE (CVAE)')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for Beta-VAE')
    parser.add_argument('--anneal', action='store_true', help='Enable KL Annealing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--clustering', type=str, default='kmeans', help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to real audio data directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
