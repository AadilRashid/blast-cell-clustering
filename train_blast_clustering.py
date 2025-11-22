#!/usr/bin/env python3
"""
Unsupervised Morphological Clustering of Blast Cells with Uncertainty Quantification
Novel approach: Discover blast subtypes without labels using deep features + clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BlastDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, str(img_path)

class FeatureExtractor(nn.Module):
    """Deep feature extractor with uncertainty via MC Dropout"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        # Use pretrained ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add dropout for uncertainty
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2048, 512)
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        features = self.fc(features)
        return F.normalize(features, p=2, dim=1)  # L2 normalize
    
    def extract_with_uncertainty(self, x, n_samples=20):
        """Extract features with uncertainty estimation"""
        self.train()  # Enable dropout
        features_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.forward(x)
                features_list.append(features.cpu().numpy())
        
        features_array = np.array(features_list)
        mean_features = features_array.mean(axis=0)
        uncertainty = features_array.std(axis=0).mean(axis=1)  # Average std across dimensions
        
        return mean_features, uncertainty

def extract_all_features(model, dataloader, device):
    """Extract features for all images"""
    model.eval()
    all_features = []
    all_uncertainties = []
    all_paths = []
    
    print("\nExtracting deep features with uncertainty...")
    for images, paths in tqdm(dataloader):
        images = images.to(device)
        features, uncertainties = model.extract_with_uncertainty(images, n_samples=20)
        
        all_features.append(features)
        all_uncertainties.extend(uncertainties)
        all_paths.extend(paths)
    
    all_features = np.vstack(all_features)
    all_uncertainties = np.array(all_uncertainties)
    
    return all_features, all_uncertainties, all_paths

def perform_clustering(features, n_clusters_range=[3, 4, 5, 6, 7, 8]):
    """Try different clustering methods and find optimal k"""
    print("\nPerforming clustering analysis...")
    
    results = {}
    
    # Try different k values for K-means
    best_score = -1
    best_k = None
    best_labels = None
    
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski = calinski_harabasz_score(features, labels)
        
        results[k] = {
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'calinski_harabasz': float(calinski),
            'labels': labels
        }
        
        print(f"K={k}: Silhouette={silhouette:.3f}, DB={davies_bouldin:.3f}, CH={calinski:.1f}")
        
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
            best_labels = labels
    
    print(f"\n✓ Best K={best_k} (Silhouette={best_score:.3f})")
    
    return best_labels, best_k, results

def analyze_clusters(features, labels, uncertainties, paths, n_clusters):
    """Analyze cluster characteristics"""
    print(f"\nAnalyzing {n_clusters} clusters...")
    
    cluster_stats = {}
    
    for i in range(n_clusters):
        mask = labels == i
        cluster_features = features[mask]
        cluster_uncertainties = uncertainties[mask]
        cluster_paths = [p for p, m in zip(paths, mask) if m]
        
        stats = {
            'size': int(mask.sum()),
            'mean_uncertainty': float(cluster_uncertainties.mean()),
            'std_uncertainty': float(cluster_uncertainties.std()),
            'sample_images': [str(p) for p in cluster_paths[:10]]  # First 10 images
        }
        
        cluster_stats[f'cluster_{i}'] = stats
        
        print(f"Cluster {i}: {stats['size']} cells, "
              f"uncertainty={stats['mean_uncertainty']:.4f}±{stats['std_uncertainty']:.4f}")
    
    return cluster_stats

def visualize_clusters(features, labels, uncertainties, save_dir='figures'):
    """Generate visualization figures"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # 1. PCA visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Blast Cell Clusters (PCA)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PCA visualization")
    
    # 2. t-SNE visualization
    print("Computing t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Blast Cell Clusters (t-SNE)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clusters_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved t-SNE visualization")
    
    # 3. Uncertainty distribution per cluster
    n_clusters = len(np.unique(labels))
    plt.figure(figsize=(12, 6))
    for i in range(n_clusters):
        mask = labels == i
        plt.hist(uncertainties[mask], bins=30, alpha=0.5, label=f'Cluster {i}')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution by Cluster')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/uncertainty_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved uncertainty distribution")
    
    # 4. Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='steelblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    plt.title('Cluster Size Distribution')
    for i, (u, c) in enumerate(zip(unique, counts)):
        plt.text(u, c, str(c), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cluster_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster sizes")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Collect all blast images
    train_images = list(Path('data/C-NMC/C-NMC_Leukemia/training_data/fold_0/all').glob('*.bmp'))
    test_images = list(Path('data/C-NMC/C-NMC_Leukemia/testing_data/C-NMC_test_final_phase_data').glob('*.bmp'))
    all_images = train_images + test_images
    
    print(f"\nTotal blast images: {len(all_images)}")
    print(f"  Training: {len(train_images)}")
    print(f"  Testing: {len(test_images)}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = BlastDataset(all_images, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize feature extractor
    print("\nInitializing feature extractor (ResNet50)...")
    model = FeatureExtractor(dropout_rate=0.3).to(device)
    
    # Extract features
    features, uncertainties, paths = extract_all_features(model, dataloader, device)
    print(f"✓ Extracted features: {features.shape}")
    
    # Perform clustering
    labels, best_k, clustering_results = perform_clustering(features, n_clusters_range=[3, 4, 5, 6, 7, 8])
    
    # Analyze clusters
    cluster_stats = analyze_clusters(features, labels, uncertainties, paths, best_k)
    
    # Visualize
    visualize_clusters(features, labels, uncertainties)
    
    # Save results (remove non-serializable data)
    metrics = {k: v for k, v in clustering_results[best_k].items() if k != 'labels'}
    results = {
        'n_images': len(all_images),
        'n_clusters': int(best_k),
        'clustering_metrics': metrics,
        'cluster_statistics': cluster_stats
    }
    
    with open('clustering_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.savez('clustering_data.npz',
             features=features,
             labels=labels,
             uncertainties=uncertainties,
             paths=paths)
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE")
    print("="*60)
    print(f"Discovered {best_k} blast cell subtypes")
    print(f"Results saved to clustering_results.json")
    print(f"Data saved to clustering_data.npz")
    print(f"Figures saved to figures/")
    print("="*60)

if __name__ == '__main__':
    main()
