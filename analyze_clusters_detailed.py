#!/usr/bin/env python3
"""
Detailed cluster analysis and figure generation for manuscript
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import pandas as pd

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

def load_results():
    """Load clustering results"""
    data = np.load('clustering_data.npz', allow_pickle=True)
    with open('clustering_results.json', 'r') as f:
        results = json.load(f)
    
    return {
        'features': data['features'],
        'labels': data['labels'],
        'uncertainties': data['uncertainties'],
        'paths': data['paths'],
        'results': results
    }

def create_summary_table(data):
    """Create detailed summary table"""
    labels = data['labels']
    uncertainties = data['uncertainties']
    n_clusters = len(np.unique(labels))
    
    summary = []
    for i in range(n_clusters):
        mask = labels == i
        cluster_unc = uncertainties[mask]
        
        summary.append({
            'Cluster': i,
            'Size': int(mask.sum()),
            'Percentage': f"{100*mask.sum()/len(labels):.1f}%",
            'Mean Uncertainty': f"{cluster_unc.mean():.6f}",
            'Std Uncertainty': f"{cluster_unc.std():.6f}",
            'Min Uncertainty': f"{cluster_unc.min():.6f}",
            'Max Uncertainty': f"{cluster_unc.max():.6f}"
        })
    
    df = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("CLUSTER SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    df.to_csv('cluster_summary.csv', index=False)
    print("\n✓ Saved to cluster_summary.csv")
    
    return df

def identify_high_uncertainty_cases(data, threshold_std=2.0):
    """Identify cases requiring expert review"""
    uncertainties = data['uncertainties']
    paths = data['paths']
    labels = data['labels']
    
    mean_unc = uncertainties.mean()
    std_unc = uncertainties.std()
    threshold = mean_unc + threshold_std * std_unc
    
    high_unc_mask = uncertainties > threshold
    high_unc_cases = {
        'threshold': float(threshold),
        'n_cases': int(high_unc_mask.sum()),
        'percentage': float(100 * high_unc_mask.sum() / len(uncertainties)),
        'cases': []
    }
    
    for i, (path, unc, label) in enumerate(zip(paths[high_unc_mask], 
                                                 uncertainties[high_unc_mask],
                                                 labels[high_unc_mask])):
        high_unc_cases['cases'].append({
            'image': str(path),
            'uncertainty': float(unc),
            'cluster': int(label)
        })
    
    print(f"\nHigh-uncertainty cases (>{threshold:.6f}):")
    print(f"  Count: {high_unc_cases['n_cases']} ({high_unc_cases['percentage']:.2f}%)")
    
    with open('high_uncertainty_cases.json', 'w') as f:
        json.dump(high_unc_cases, f, indent=2)
    
    print("✓ Saved to high_uncertainty_cases.json")
    
    return high_unc_cases

def plot_elbow_curve():
    """Plot elbow curve for K selection"""
    with open('clustering_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract metrics for different K values
    k_values = [3, 4, 5, 6, 7, 8]
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    
    # Note: This requires re-running clustering with all K values
    # For now, create placeholder
    print("\nNote: Elbow plot requires clustering_results.json with all K values")
    
def visualize_sample_images(data, n_samples=5):
    """Visualize sample images from each cluster"""
    labels = data['labels']
    paths = data['paths']
    n_clusters = len(np.unique(labels))
    
    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(15, 3*n_clusters))
    
    for i in range(n_clusters):
        cluster_paths = [p for p, l in zip(paths, labels) if l == i]
        samples = np.random.choice(cluster_paths, min(n_samples, len(cluster_paths)), replace=False)
        
        for j, img_path in enumerate(samples):
            try:
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'Cluster {i}', fontsize=12, fontweight='bold')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('figures/sample_images_per_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved sample images visualization")

def create_uncertainty_boxplot(data):
    """Create boxplot of uncertainty by cluster"""
    labels = data['labels']
    uncertainties = data['uncertainties']
    n_clusters = len(np.unique(labels))
    
    cluster_data = [uncertainties[labels == i] for i in range(n_clusters)]
    
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(cluster_data, labels=[f'Cluster {i}' for i in range(n_clusters)],
                     patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty Distribution by Cluster')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/uncertainty_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved uncertainty boxplot")

def statistical_tests(data):
    """Perform statistical tests"""
    from scipy.stats import chi2_contingency, kruskal
    
    labels = data['labels']
    uncertainties = data['uncertainties']
    n_clusters = len(np.unique(labels))
    
    # Test 1: Chi-square test for uniform cluster distribution
    observed = [np.sum(labels == i) for i in range(n_clusters)]
    expected = [len(labels) / n_clusters] * n_clusters
    chi2, p_value_chi2 = chi2_contingency([observed, expected])[:2]
    
    print(f"\nChi-square test for uniform distribution:")
    print(f"  Chi2 = {chi2:.4f}, p-value = {p_value_chi2:.4f}")
    if p_value_chi2 > 0.05:
        print("  → Clusters are uniformly distributed (p>0.05)")
    else:
        print("  → Clusters are NOT uniformly distributed (p<0.05)")
    
    # Test 2: Kruskal-Wallis test for uncertainty differences
    cluster_uncertainties = [uncertainties[labels == i] for i in range(n_clusters)]
    h_stat, p_value_kw = kruskal(*cluster_uncertainties)
    
    print(f"\nKruskal-Wallis test for uncertainty differences:")
    print(f"  H = {h_stat:.4f}, p-value = {p_value_kw:.4f}")
    if p_value_kw > 0.05:
        print("  → No significant difference in uncertainty between clusters (p>0.05)")
    else:
        print("  → Significant difference in uncertainty between clusters (p<0.05)")
    
    return {
        'chi2_test': {'statistic': float(chi2), 'p_value': float(p_value_chi2)},
        'kruskal_wallis': {'statistic': float(h_stat), 'p_value': float(p_value_kw)}
    }

def main():
    print("Loading clustering results...")
    data = load_results()
    
    print("\n" + "="*80)
    print("DETAILED CLUSTER ANALYSIS")
    print("="*80)
    
    # Summary table
    summary_df = create_summary_table(data)
    
    # High uncertainty cases
    high_unc = identify_high_uncertainty_cases(data, threshold_std=2.0)
    
    # Statistical tests
    stats = statistical_tests(data)
    
    # Visualizations
    print("\nGenerating additional visualizations...")
    visualize_sample_images(data, n_samples=5)
    create_uncertainty_boxplot(data)
    
    # Save comprehensive results
    comprehensive_results = {
        'summary': summary_df.to_dict('records'),
        'high_uncertainty': high_unc,
        'statistical_tests': stats
    }
    
    with open('comprehensive_analysis.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Files generated:")
    print("  - cluster_summary.csv")
    print("  - high_uncertainty_cases.json")
    print("  - comprehensive_analysis.json")
    print("  - figures/sample_images_per_cluster.png")
    print("  - figures/uncertainty_boxplot.png")
    print("="*80)

if __name__ == '__main__':
    main()
