# Unsupervised Blast Cell Clustering with Uncertainty Quantification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

Official implementation of **"Unsupervised Discovery of Blast Cell Morphological Subtypes Using Deep Learning with Uncertainty Quantification"** (IEEE TMI 2025).

##  Overview

This repository contains code for discovering blast cell morphological subtypes in acute leukemia without manual labeling, using deep learning with uncertainty quantification.

**Key Features:**
-  Unsupervised clustering of 4,944 blast cell images
-  Monte Carlo Dropout for uncertainty quantification
-  Comprehensive visualization (PCA, t-SNE, uncertainty maps)
-  Statistical validation (silhouette, Davies-Bouldin, Calinski-Harabasz)
-  Clinical decision support with confidence scores

## 📊 Results

- **3 distinct blast subtypes** discovered automatically
- **Balanced distribution:** 33.5%, 33.8%, 32.6% (p=0.95)
- **Low uncertainty:** 0.0238±0.0004
- **Significant cluster separation:** H=340.22, p<0.001

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/blast-cell-clustering.git
cd blast-cell-clustering
pip install -r requirements.txt
```

### Download Dataset

```bash
# C-NMC Leukemia Classification Challenge dataset
python download_dataset.py
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)

### Run Clustering

```bash
python train_blast_clustering.py
```

### Generate Analysis

```bash
python analyze_clusters_detailed.py
```

## 📁 Repository Structure

```
blast-cell-clustering/
├── train_blast_clustering.py      # Main clustering pipeline
├── analyze_clusters_detailed.py   # Detailed analysis
├── ablation_studies.py             # Ablation experiments
├── requirements.txt                # Python dependencies
├── figures/                        # Generated visualizations
├── results/                        # Clustering results
└── README.md                       # This file
```

## 🔬 Methodology

### 1. Feature Extraction
- **Backbone:** ResNet50 pretrained on ImageNet
- **Features:** 512-dimensional L2-normalized embeddings
- **Uncertainty:** Monte Carlo Dropout (20 samples, dropout=0.3)

### 2. Clustering
- **Algorithm:** K-means (K=3-8)
- **Initialization:** k-means++ (10 random seeds)
- **Metrics:** Silhouette, Davies-Bouldin, Calinski-Harabasz

### 3. Validation
- Chi-square test for cluster balance
- Kruskal-Wallis test for uncertainty differences
- PCA and t-SNE visualization

## 📈 Ablation Studies

Run ablation experiments:

```bash
python ablation_studies.py
```

Experiments include:
1. **Backbone comparison:** ResNet50 vs ResNet18 vs VGG16
2. **Feature dimensions:** 128, 256, 512, 1024
3. **Dropout rates:** 0.1, 0.2, 0.3, 0.4, 0.5
4. **MC samples:** 5, 10, 20, 30, 50
5. **Clustering algorithms:** K-means vs DBSCAN vs Hierarchical

## 📊 Results Reproduction

To reproduce paper results:

```bash
# Run main clustering
python train_blast_clustering.py

# Generate all figures
python analyze_clusters_detailed.py

# Run ablation studies
python ablation_studies.py

# Results will be saved in:
# - clustering_results.json
# - figures/
# - ablation_results/
```




## Dataset

**C-NMC Leukemia Classification Challenge**
- **Source:** [Kaggle](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
- **Size:** 4,944 blast cell images
- **Format:** BMP, 450×450 pixels
- **License:** CC BY 4.0

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- Pillow

See `requirements.txt` for complete list.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

##  Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

- **Author:** Your Name
- **Email:** your.email@institution.edu
- **Paper:** [IEEE TMI](link-to-paper)

## 🙏 Acknowledgments

- C-NMC Challenge organizers for the public dataset
- PyTorch and scikit-learn communities
- Reviewers for valuable feedback

## 📚 Related Work

- [CytoDiffusion](https://github.com/original/cytodiffusion) - Supervised blast detection
- [LeukemiaAI](https://github.com/example/leukemia) - Multi-modal leukemia classification

## 🔗 Links

- **Paper:** [IEEE TMI](link)
- **Dataset:** [Kaggle](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
- **Supplementary:** [Materials](link)
