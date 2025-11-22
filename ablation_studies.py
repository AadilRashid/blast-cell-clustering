#!/usr/bin/env python3
"""
Ablation Studies for Blast Cell Clustering
Tests different hyperparameters and architectures
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json
from tqdm import tqdm
import sys

# Import from main script
sys.path.append('.')
from train_blast_clustering import BlastDataset, FeatureExtractor

class AblationExperiments:
    def __init__(self, data_dir='data/C-NMC/C-NMC_Leukemia'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Collect all images
        train_images = list(Path(data_dir).glob('training_data/fold_0/all/*.bmp'))
        test_images = list(Path(data_dir).glob('testing_data/C-NMC_test_final_phase_data/*.bmp'))
        self.all_images = train_images + test_images
        print(f"Total images: {len(self.all_images)}")
    
    def extract_features(self, model, batch_size=32):
        """Extract features using given model"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = BlastDataset(self.all_images, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        model.eval()
        all_features = []
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                features = model(images)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def evaluate_clustering(self, features, k=3):
        """Evaluate clustering quality"""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        return {
            'silhouette': float(silhouette_score(features, labels)),
            'davies_bouldin': float(davies_bouldin_score(features, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(features, labels))
        }
    
    def ablation_backbone(self):
        """Ablation 1: Different backbone architectures"""
        print("\n" + "="*60)
        print("ABLATION 1: Backbone Architecture")
        print("="*60)
        
        backbones = {
            'ResNet18': models.resnet18(pretrained=True),
            'ResNet50': models.resnet50(pretrained=True),
            'ResNet101': models.resnet101(pretrained=True),
        }
        
        results = {}
        for name, backbone in backbones.items():
            print(f"\nTesting {name}...")
            
            # Modify backbone
            if 'resnet' in name.lower():
                model = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
            
            model = model.to(self.device)
            features = self.extract_features(model)
            metrics = self.evaluate_clustering(features)
            
            results[name] = metrics
            print(f"{name}: Silhouette={metrics['silhouette']:.4f}")
        
        self.results['backbone'] = results
        return results
    
    def ablation_feature_dim(self):
        """Ablation 2: Feature dimensionality"""
        print("\n" + "="*60)
        print("ABLATION 2: Feature Dimensionality")
        print("="*60)
        
        dims = [128, 256, 512, 1024]
        results = {}
        
        for dim in dims:
            print(f"\nTesting dim={dim}...")
            
            model = FeatureExtractor(dropout_rate=0.3)
            model.fc = nn.Linear(2048, dim)
            model = model.to(self.device)
            
            features = self.extract_features(model)
            metrics = self.evaluate_clustering(features)
            
            results[f'dim_{dim}'] = metrics
            print(f"Dim {dim}: Silhouette={metrics['silhouette']:.4f}")
        
        self.results['feature_dim'] = results
        return results
    
    def ablation_dropout(self):
        """Ablation 3: Dropout rate"""
        print("\n" + "="*60)
        print("ABLATION 3: Dropout Rate")
        print("="*60)
        
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = {}
        
        for rate in dropout_rates:
            print(f"\nTesting dropout={rate}...")
            
            model = FeatureExtractor(dropout_rate=rate).to(self.device)
            features = self.extract_features(model)
            metrics = self.evaluate_clustering(features)
            
            results[f'dropout_{rate}'] = metrics
            print(f"Dropout {rate}: Silhouette={metrics['silhouette']:.4f}")
        
        self.results['dropout'] = results
        return results
    
    def ablation_k_values(self):
        """Ablation 4: Number of clusters"""
        print("\n" + "="*60)
        print("ABLATION 4: Number of Clusters (K)")
        print("="*60)
        
        # Use baseline model
        model = FeatureExtractor(dropout_rate=0.3).to(self.device)
        features = self.extract_features(model)
        
        k_values = [2, 3, 4, 5, 6, 7, 8]
        results = {}
        
        for k in k_values:
            print(f"\nTesting K={k}...")
            metrics = self.evaluate_clustering(features, k=k)
            results[f'k_{k}'] = metrics
            print(f"K={k}: Silhouette={metrics['silhouette']:.4f}")
        
        self.results['k_values'] = results
        return results
    
    def ablation_clustering_algorithm(self):
        """Ablation 5: Clustering algorithms"""
        print("\n" + "="*60)
        print("ABLATION 5: Clustering Algorithms")
        print("="*60)
        
        # Use baseline model
        model = FeatureExtractor(dropout_rate=0.3).to(self.device)
        features = self.extract_features(model)
        
        results = {}
        
        # K-means
        print("\nTesting K-means...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(features)
        results['kmeans'] = {
            'silhouette': float(silhouette_score(features, labels_kmeans)),
            'davies_bouldin': float(davies_bouldin_score(features, labels_kmeans)),
            'calinski_harabasz': float(calinski_harabasz_score(features, labels_kmeans))
        }
        print(f"K-means: Silhouette={results['kmeans']['silhouette']:.4f}")
        
        # DBSCAN
        print("\nTesting DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        labels_dbscan = dbscan.fit_predict(features)
        n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        
        if n_clusters > 1:
            results['dbscan'] = {
                'n_clusters': n_clusters,
                'silhouette': float(silhouette_score(features, labels_dbscan)),
                'davies_bouldin': float(davies_bouldin_score(features, labels_dbscan)),
                'calinski_harabasz': float(calinski_harabasz_score(features, labels_dbscan))
            }
            print(f"DBSCAN: {n_clusters} clusters, Silhouette={results['dbscan']['silhouette']:.4f}")
        else:
            results['dbscan'] = {'error': 'Only 1 cluster found'}
            print("DBSCAN: Failed (only 1 cluster)")
        
        self.results['clustering_algorithm'] = results
        return results
    
    def run_all(self):
        """Run all ablation studies"""
        print("\n" + "="*60)
        print("RUNNING ALL ABLATION STUDIES")
        print("="*60)
        
        self.ablation_backbone()
        self.ablation_feature_dim()
        self.ablation_dropout()
        self.ablation_k_values()
        self.ablation_clustering_algorithm()
        
        # Save results
        with open('ablation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*60)
        print("ABLATION STUDIES COMPLETE")
        print("="*60)
        print("Results saved to ablation_results.json")
        
        return self.results

def main():
    ablation = AblationExperiments()
    results = ablation.run_all()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for study, study_results in results.items():
        print(f"\n{study.upper()}:")
        for config, metrics in study_results.items():
            if isinstance(metrics, dict) and 'silhouette' in metrics:
                print(f"  {config}: Silhouette={metrics['silhouette']:.4f}")

if __name__ == '__main__':
    main()
