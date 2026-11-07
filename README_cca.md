# CCA (Curvilinear Component Analysis) and the curvature gauge

**Author:** Salma Hasannejad · Team Outliers · CS 7300

A from-scratch implementation of Curvilinear Component Analysis (Demartines and Hérault, 1997),
compared with PCA on ORL, Yale and Extended Yale B, plus the curvature gauge G from the proposal
(Section 4.2). scikit-learn has no CCA (`sklearn.cross_decomposition.CCA` is *Canonical*
Correlation Analysis, a different method).

## What is here

```
src/cca/core.py                 CurvilinearComponentAnalysis class (fit / transform, like PCA)
src/cca/data.py                 reads data/raw.zip, 46 x 56 preprocessing, train/test splits
scripts/03_run_cca.py           CCA vs PCA, d = 60, five classifiers, all protocols
scripts/04_run_gauge.py         curvature gauge G and G+ (PCA vs Isomap residual variance)
examples/cca_scurve_check.py    sanity check on the S-curve and Swiss roll
examples/cca_toy_step_by_step.py   3 photos x 4 pixels, prints every step of training
docs/cca_explained.md           plain-English explanation of the method and results
docs/design_decisions.md        every choice we made, why, the divide-by-zero guards, and the checks
results/cca/                    cca_face_results.csv, cca_face_summary.csv, cca_check.png
results/gauge/                  gauge_per_split.csv, gauge_summary.csv, gauge_vs_cca.csv, rv_curves.png
```

## How to run (from the repo root)

```bash
pip install -r requirements.txt matplotlib
python scripts/03_run_cca.py --quick    # fast check, about 2 min (saved to results/cca/quick/)
python scripts/03_run_cca.py            # full run, about 5-10 min
python scripts/04_run_gauge.py          # a few minutes, uses results/cca/ for the comparison
```

Both scripts read `data/raw.zip` directly (`--zip "path/to/raw.zip"` if it is elsewhere).
Splits use a fixed seed (0), so the numbers reproduce.

## Protocols

| Dataset | Split |
|---|---|
| ORL, 40 people | 5 training images per person, 10 random splits |
| Yale, 15 people | leave-one-condition-out, 11 folds |
| Extended Yale B | 30 training images per person, 5 random splits |
| Extended Yale B lighting | train on subset 1, test on subsets 2, 3, 4, 5 (angle <=12, <=25, <=50, <=77, >77 degrees) |

### Ext. Yale B lighting subsets

Each Ext. Yale B photo is taken with the light at a different angle from the camera. The bigger the
angle, the more the light comes from the side, so the face has **more shadow** and is harder to
recognise. We group the photos into 5 standard subsets by that angle:

| Subset | Light angle from the camera | What the face looks like | Photos (27 people) | Used for |
|---|---|---|---|---|
| 1 | 0–12° | light from the front, almost no shadow | 189 | **training** |
| 2 | 13–25° | small shadows | 270 | test |
| 3 | 26–50° | half the face partly in shadow | 324 | test |
| 4 | 51–77° | strong shadows | 432 | test |
| 5 | more than 77° | light from the side, mostly dark | 513 | test |

The angle is computed from each file name (azimuth A, elevation E):
angle = arccos(cos A · cos E). We train only on the well-lit subset 1 and test on each harder
subset separately, so we can see how accuracy drops as the shadows grow.

Images are 46 x 56, scaled to [0, 1], centred with training images only, reduced to d = 60.
CCA and PCA use exactly the same splits and classifiers (1-NN, soft k-NN, LDA, linear SVM, RBF SVM).

## What is saved in the CSV files

Accuracy is a fraction (0.945 = 94.5%). Time is in seconds.

**`results/cca/cca_face_results.csv`**: one row per split × method × classifier (the raw results)

| Column | Meaning |
|---|---|
| `setting` | ORL, Yale, ExtYaleB random, or ExtYaleB light: test subset 2–5 |
| `split` | which random split or fold (0, 1, 2, …) |
| `method` | PCA or CCA |
| `classifier` | 1-NN, soft k-NN, LDA, Linear SVM, RBF SVM |
| `accuracy` | share of test photos named correctly |
| `seconds` | time to fit the reducer and place the test photos |

**`results/cca/cca_face_summary.csv`**: the averages over splits (the table we present)

| Column | Meaning |
|---|---|
| `mean_acc`, `sd_acc` | mean and standard deviation of accuracy over the splits |
| `n_splits` | how many splits were averaged (10 ORL, 11 Yale, 5 Ext. Yale B, 1 lighting) |
| `mean_seconds` | average fit + transform time |

**`results/gauge/gauge_per_split.csv`**: one row per training set × Isomap k

| Column | Meaning |
|---|---|
| `n_train`, `k` | number of training photos; Isomap neighbours |
| `G`, `d_star` | the gauge and the dimension d where the gap is largest |
| `G_plus`, `d_star_plus` | signed gauge (only gaps where Isomap is better) and its d |
| `RV_PCA_d`, `RV_Isomap_d` | residual variance at d = 2, 5, 10, 20, 40, 60 (lower is better) |

**`results/gauge/gauge_summary.csv`**: G, G+ and RV at d = 60, averaged per setting and k.
**`results/gauge/gauge_vs_cca.csv`**: G and G+ (k = 10) next to the CCA − PCA gains in
percentage points (1-NN, LDA, and best CCA − best PCA).

## How CCA works

CCA looks for points in d dimensions whose pairwise distances match the original pixel
distances, but only for pairs that are close on the map:

E = ½ Σ (Dx_ij − Dy_ij)² · F(Dy_ij, λ),   F = 1 if Dy_ij < λ, else 0

λ shrinks from the largest distance to the 5th percentile over 50 epochs, starting from PCA.
Test images are placed against the frozen training map, so they never influence training.
Why we start from PCA, why λ ends at the 5th percentile, the divide-by-zero guards, and every
check we did are in [`docs/design_decisions.md`](docs/design_decisions.md).

## Main results (mean test accuracy, %)

| Setting | 1-NN PCA | 1-NN CCA | LDA PCA | LDA CCA |
|---|---|---|---|---|
| ORL | 94.6 | 94.8 | 95.6 | 92.8 |
| Yale | 87.3 | 86.1 | 91.5 | 88.5 |
| Ext. Yale B (30/person) | 57.4 | **65.4** | 84.2 | 75.7 |

* CCA helps nearest-neighbour classifiers under strong lighting (+8.0 points with 1-NN on Ext. Yale B),
  because it keeps local distances.
* PCA still gives the best accuracy on every dataset (with LDA or an SVM): ORL 95.6 (LDA),
  Yale 93.9 (linear SVM), Ext. Yale B 87.9 (RBF SVM), against CCA's best of 94.8, 90.9 and 79.8.
* CCA costs more time (on Salma's laptop): about 7 s per split on ORL, 2 s on Yale and 58 s on Ext. Yale B,
  against under 1 s for PCA.
* Lighting test, LDA accuracy as the light angle grows:

  | Test subset | 2 (13–25°) | 3 (26–50°) | 4 (51–77°) | 5 (>77°) |
  |---|---|---|---|---|
  | PCA | 99.6 | 69.4 | 43.8 | 17.2 |
  | CCA | 93.7 | 54.6 | 27.1 | 7.4 |

  Both methods fail as the shadows grow, and CCA falls faster.

## Use CCA in your own code

```python
import sys; sys.path.insert(0, "src")
from cca import CurvilinearComponentAnalysis

cca = CurvilinearComponentAnalysis(n_components=60, random_state=0)
Z_train = cca.fit_transform(X_train)   # training images only
Z_test = cca.transform(X_test)          # test images placed on the frozen map
```

It follows the scikit-learn fit / transform interface, so it can replace PCA in any pipeline.

## Notes

* Ext. Yale B is read from the raw images here. The team's PCA table (`README_pca_kpca.md`) first
  drops 102 badly lit images, so its PCA numbers differ from ours. To compare CCA with PCA, use
  the PCA column in **our** CSV (same images, same splits).
* λ end (5th percentile) and the step size α were not tuned.

## Curvature gauge

RV_m(d) = 1 − R²(D̂_m, D_Y,m(d)),  G = max_d |RV_PCA(d) − RV_Isomap(d)|,
G+ = max(0, max_d [RV_PCA(d) − RV_Isomap(d)]) (only gaps where Isomap is better).
Isomap is used only as a ruler, with k in {5, 10, 15}; disconnected graphs are skipped.

G+ (k = 10): ORL 0.055, Yale 0.094, Ext. Yale B 0, lighting 0. The gauge recommends PCA everywhere.
That matches the best overall accuracy, but not CCA's +8 with 1-NN on Ext. Yale B.

## Reference

P. Demartines and J. Hérault, "Curvilinear Component Analysis: A Self-Organizing Neural Network for
Nonlinear Mapping of Data Sets," *IEEE Transactions on Neural Networks*, 8(1):148–154, 1997.
