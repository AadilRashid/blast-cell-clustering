# Dataset Download Instructions

## C-NMC Leukemia Classification Challenge

### Option 1: Kaggle (Recommended)

1. Install kagglehub:
   ```bash
   pip install kagglehub
   ```

2. Download dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download('andrewmvd/leukemia-classification')
   print(f'Downloaded to: {path}')
   ```

3. Move to project directory:
   ```bash
   mv <downloaded_path> data/C-NMC/
   ```

### Option 2: Manual Download

1. Go to https://www.kaggle.com/datasets/andrewmvd/leukemia-classification
2. Click "Download" (requires Kaggle account)
3. Extract to `data/C-NMC/C-NMC_Leukemia/`

### Dataset Structure

```
data/C-NMC/C-NMC_Leukemia/
├── training_data/
│   └── fold_0/
│       └── all/  (2,358 blast images)
└── testing_data/
    └── C-NMC_test_final_phase_data/  (2,586 blast images)
```

**Total:** 4,944 blast cell images

### Dataset Details

- **Format:** BMP images, 450×450 pixels
- **Staining:** Giemsa-stained peripheral blood smears
- **Source:** C-NMC Challenge 2019
- **License:** CC BY 4.0
- **Citation:** Gupta & Gupta (2019)

### Troubleshooting

**Issue:** Kagglehub authentication error
**Solution:** Set up Kaggle API credentials:
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Place `kaggle.json` in `~/.kaggle/`

**Issue:** Disk space error
**Solution:** Dataset requires ~2GB free space
