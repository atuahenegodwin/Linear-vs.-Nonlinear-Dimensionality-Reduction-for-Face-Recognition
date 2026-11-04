# PCA and Kernel PCA: linear baselines

Code, data arrays and results for the linear / kernel-linear part of the study.

## What is here

```
src/pca_kpca/core.py                  PCA, polynomial and RBF Kernel PCA, tuning, 5 classifiers
scripts/01_prepare_data.py            data/raw.zip -> 46 x 56 arrays (ORL, Yale, Extended Yale B)
scripts/02_run_pca_kpca.py            runs all four protocols (--only to run a subset)
scripts/06_analyze_pca_kpca.py        significance tests + summary tables
data/processed/                       orl.npz, yale.npz, eyb.npz, eyb_exclusions.csv
results/pca_kpca/                     accuracy_long.csv, summary_table.csv, significance.csv
requirements.txt
```

## How to run (from the repo root)

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py       # about 20 s
python scripts/02_run_pca_kpca.py       # about 1-2 min
python scripts/06_analyze_pca_kpca.py   # seconds
```

The run is deterministic: split seeds are fixed, so the numbers below reproduce exactly.

## Protocols

| Protocol | Dataset | Split |
|---|---|---|
| `orl` | ORL, 40 people | 3, 4, 5, 6 training images per person, 10 random splits each |
| `yale` | Yale, 15 people | leave-one-condition-out, 11 folds |
| `eyb_random` | Extended Yale B, 27 people | 10, 20, 30 training images per person, 5 random splits each |
| `eyb_light` | Extended Yale B | train on lighting subset 1, test on subsets 2, 3, 4, 5 |

All images are 46 x 56, scaled to [0, 1], reduced to d = 60. Centring uses training images only.
Kernel settings (3 candidates per kernel) are chosen by 1-NN accuracy on validation folds inside
the training set. Classifiers: 1-NN, soft k-NN, LDA, linear SVM, RBF SVM.

## Statistics

Random-split protocols: corrected resampled t-test (Nadeau and Bengio, 2003).
Fixed-split protocols: exact McNemar test on per-image predictions.
All 120 p-values (2 kernels x 5 classifiers x 12 settings) are Holm-adjusted together, alpha = 0.05.

## Main finding

Kernel PCA does not beat PCA on these data. Of 120 comparisons, 113 show no significant
difference, 1 favours Kernel PCA and 6 favour PCA. The RBF kernel is significantly worse under
strong lighting change (LDA, linear SVM). Polynomial Kernel PCA is indistinguishable from PCA.

## Extended Yale B cleaning

`data/processed/eyb_exclusions.csv` lists the 102 images (5.9%) removed. Rule: correlate each image
with the pixel-wise median of the same shot (same light direction) over all subjects; drop images
below 0.40. Lighting subsets use the light-to-camera angle: <=12, <=25, <=50, <=77, >77 degrees.
