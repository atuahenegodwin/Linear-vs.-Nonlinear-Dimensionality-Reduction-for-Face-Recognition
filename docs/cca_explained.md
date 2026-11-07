# CCA in our project: explained step by step

*Team Outliers, CS 7300. Salma's part: Curvilinear Component Analysis (CCA).*
Numbers below are from the full run of `run_cca_faces.py` on 5 Nov 2026.

---

## 1. The pipeline, step by step

| Step | What happens | Where in the code |
|---|---|---|
| 1. Load | Read every image straight from `raw.zip`: ORL 400 images / 40 people, Yale 165 / 15, Ext. Yale B 1,728 / 27 | `load_all()` |
| 2. Preprocess | Grayscale, crop to 46:56 shape, resize to 46 x 56 = **2,576 pixels**, scale to [0, 1] | `to_vector()` |
| 3. Split | Train / test split (same for PCA and CCA) | `all_splits()` |
| 4. Centre | Subtract the mean image of the **training** images only | `main()` |
| 5. Reduce | 2,576 pixels -> **60 features**, with PCA and with CCA, fitted on training images only | `reducers()` |
| 6. Project test | Test images are placed into the same 60-dim space (`transform`) | `cca.transform()` |
| 7. Classify | Five classifiers on the 60 features: 1-NN, soft k-NN, LDA, linear SVM, RBF SVM | `classifiers()` |
| 8. Evaluate | Accuracy on the test images, averaged over splits; time per fit | `main()` |

Splits (protocols):

| Dataset | Protocol | Splits |
|---|---|---|
| ORL | 5 training images per person, rest for testing | 10 random |
| Yale | Leave one condition out (e.g. train on 10, test on "happy") | 11 folds |
| Ext. Yale B random | 30 training images per person (as in the team PCA) | 5 random |
| Ext. Yale B lighting | Train on subset 1 (light within 12 degrees of the camera), test on subsets 2, 3, 4, 5 (harder and harder light) | 1 each |

## 2. Folder skeleton

```
Team Work/
├── Linear-vs.-Nonlinear-...-Recognition/   team repo
│   └── data/raw.zip                         ORL, Yale, Ext. Yale B images
└── CCA/                                     my part
    ├── cca.py               the CCA method (class CurvilinearComponentAnalysis)
    ├── example_cca.py       check on toy data (S-curve, Swiss roll) -> cca_check.png
    ├── run_cca_faces.py     CCA vs PCA on the faces -> results/
    ├── requirements.txt     packages to install
    ├── README.md            how to run
    ├── CCA_explained.md     this file
    └── results/
        ├── cca_face_results.csv   every split x classifier x method
        └── cca_face_summary.csv   mean and SD
```

## 3. What CCA does (the idea)

Each face is a point in a 2,576-dimensional space. Faces of the same person under small
changes lie on a curved surface in that space. PCA flattens the data onto a flat
60-dimensional plane, so a curved surface gets squashed. **CCA instead tries to lay the
points out in 60 dimensions so that the distance between every pair of nearby faces stays
the same as in the original space.** Long distances are allowed to be wrong, which lets CCA
"unroll" a curved surface (like flattening an orange peel by letting it tear far away while
keeping each small piece accurate).

Reference: Demartines and Hérault (1997), IEEE Trans. Neural Networks 8(1), 148-154.

## 4. The formulas and the matrices

**Matrices used**

| Symbol | Size | What it is |
|---|---|---|
| X | n x 2,576 | training images, one per row (n = 200 for ORL) |
| Dx | n x n | distance between every **pair** of images in pixel space: Dx_ij = ‖x_i − x_j‖ |
| Y | n x d | the new 60-dim coordinates we are looking for (d = 60) |
| Dy | n x n | distance between every pair in the new space: Dy_ij = ‖y_i − y_j‖ |

Dx is computed once. Y (and so Dy) is improved step by step.

**What CCA minimises (the stress)**

    E = 1/2 * Σ_{i<j} (Dx_ij − Dy_ij)² * F(Dy_ij, λ)

    F(Dy_ij, λ) = 1 if Dy_ij < λ   (pair is close in the new space: must match)
                = 0 otherwise      (pair is far: ignore its error)

**How it learns (update rule).** Pick one image i, keep it fixed, and move every other
point j along the line through y_i:

    y_j ← y_j + α * F(Dy_ij, λ) * (Dx_ij − Dy_ij) / Dy_ij * (y_j − y_i)

If j is too close to i (Dx_ij > Dy_ij) it is pushed away; if too far, it is pulled in.
This is repeated for every i in random order (one epoch), for 50 epochs.

**Two settings shrink during training**

* α (learning rate): 0.5 → 0.005, so steps get smaller and the map settles.
* λ (neighbourhood radius): starts at the largest distance (every pair counts, so the start
  is a global, PCA-like map) and shrinks to the 5th percentile of all pairwise distances
  (only near neighbours count at the end, so the curved surface can unfold).

**Start.** Y starts from PCA's 60 components (deterministic, converges faster).

**Test images.** CCA has no formula that maps a new image to its coordinates. For each test
image x we keep the training map fixed and find the y that minimises the same stress against
the training points:

    E(y) = 1/2 * Σ_j (‖x − x_j‖ − ‖y − y_j‖)² * F(‖y − y_j‖, λ_end)

starting from the average position of its 5 nearest training images. Test images never
change the training map, so there is no leakage.

## 5. PCA uses eigenvectors. Does CCA?

**No.** This is the main difference.

| | PCA | CCA |
|---|---|---|
| How it is solved | Eigen-decomposition of the covariance matrix C = XᵀX / (n − 1); keep the top d eigenvectors | Iterative optimisation (gradient steps) of the stress E |
| What it preserves | Directions of largest variance (global, linear) | Distances between nearby pairs (local, nonlinear) |
| Components ranked? | Yes, by eigenvalue: component 1 explains the most variance | **No**: the 60 coordinates have no order or meaning on their own |
| Fewer components? | Keep the first k of the same fit (e.g. by 95 % explained variance) | Must **refit** for each d; you cannot cut columns off |
| Exact answer? | Yes, one unique solution | Approximate; depends on random seed (we fix `random_state=0`) |
| Speed | One matrix decomposition | 50 epochs over all n² pairs |

So in PCA "fewer PCs" means dropping the smallest eigenvectors. In CCA d is chosen **before**
fitting. We used **d = 60 for both** so the comparison is fair (same number of features).
To pick d for CCA on its own, refit for several d (e.g. 5, 10, 20, 40, 60) and choose by
validation accuracy, or look for the "elbow" where the stress stops dropping.

## 6. Do we pair anything?

Yes, in two senses:

1. **CCA itself works on pairs of images.** It looks at all n(n−1)/2 pairs (19,900 pairs for
   ORL with n = 200) and tries to keep each nearby pair at the same distance.
2. **PCA and CCA are compared as pairs.** On every split they get exactly the same training
   and test images and the same classifier, so each split gives one paired difference
   (CCA − PCA). That is what the "CCA − PCA" column shows, and it allows a paired test
   (e.g. paired t-test or Wilcoxon over the 10 ORL splits).

## 7. How do we choose k (and the other settings)?

CCA does not use k nearest neighbours like LLE or Isomap. Its "neighbourhood" is the
**radius λ** instead. The settings in our project:

| Setting | Value we used | How chosen |
|---|---|---|
| d (features) | 60 | Same as PCA in the midterm, for a fair comparison |
| λ start | largest pairwise distance | Rule from the paper: start global |
| λ end | 5th percentile of pairwise distances | Default; **to tune**, try 2, 5, 10, 20th percentile and keep the best 1-NN accuracy on a validation part of the training set (never the test set) |
| α | 0.5 → 0.005 | Standard; check that `stress_history_` falls and levels off |
| epochs | 50 | Stress has levelled off by then |
| k in soft k-NN | 5 | Classifier setting, same for PCA and CCA |

Too small a λ end tears the map into pieces; too large keeps it close to PCA.

## 8. Advantages and disadvantages

**Advantages**

* **vs PCA:** nonlinear, so it can unfold curved structure (S-curve: trustworthiness 0.999
  vs 0.931 for PCA). It keeps local distances, so **nearest neighbours stay meaningful**.
* **vs Isomap and LLE:** no neighbour graph, so no "disconnected graph" failure and no k to
  pick; it can tear the data apart where needed instead of forcing one connected sheet.
* **vs Sammon mapping / MDS:** it ignores long distances at the end, which are unreliable on
  a curved surface.

**Disadvantages**

* **Slow:** cost grows with n² (all pairs) × epochs. Up to 30 s vs 0.1 s for PCA.
* **No formula for new images:** test images need their own optimisation (`transform`).
* **Not unique:** depends on the random visiting order and on λ.
* **Coordinates are not ranked**, so they are hard to interpret and d must be fixed in advance.
* **Swiss roll:** with straight-line (Euclidean) distances it tears the roll instead of
  unrolling it cleanly; Isomap's geodesic distances do better there.

## 9. Our results (full run on our laptop, d = 60, mean test accuracy %)

| Setting | Classifier | PCA | CCA | CCA − PCA |
|---|---|---|---|---|
| ORL | 1-NN | 94.6 | 94.8 | +0.3 |
| ORL | RBF SVM | 93.8 | 94.6 | +0.7 |
| ORL | LDA | 95.6 | 92.8 | −2.8 |
| Yale | 1-NN | 87.3 | 86.1 | −1.2 |
| Yale | LDA | 91.5 | 88.5 | −3.0 |
| Yale | RBF SVM | 90.9 | 80.0 | −10.9 |
| Ext. Yale B random | 1-NN | 57.4 | **65.4** | **+8.0** |
| Ext. Yale B random | soft k-NN | 56.5 | **63.3** | **+6.8** |
| Ext. Yale B random | LDA | 84.2 | 75.7 | −8.5 |
| Ext. Yale B random | RBF SVM | 87.9 | 79.8 | −8.1 |
| Lighting, subset 3 | LDA | 69.4 | 54.6 | −14.8 |
| Lighting, subset 4 | LDA | 43.8 | 27.1 | −16.7 |
| Lighting, subset 5 | LDA | 17.2 | 7.4 | −9.7 |

(All 5 classifiers for every setting are in `results/cca_face_summary.csv`.)

**Time to fit + project (seconds, average per split)**

| Setting | PCA | CCA |
|---|---|---|
| ORL (200 train) | 0.58 | 7.39 |
| Yale (150 train) | 0.63 | 2.09 |
| Ext. Yale B random (810 train) | 0.56 | 57.83 |
| Ext. Yale B lighting (subset 1 train) | 0.07–0.28 | 2.6–4.4 |

**What this means (the message for the slides)**

1. **On ORL, CCA ties with PCA** (within 1 point with 1-NN and RBF SVM).
2. **CCA helps nearest-neighbour classifiers under lighting changes:** +8.0 points with 1-NN
   on Ext. Yale B. CCA keeps local distances, and 1-NN depends only on local distances.
3. **With LDA and SVMs, PCA wins** by 2–11 points. These classifiers like PCA's linear,
   variance-ordered axes.
4. **Under strong lighting (subsets 3–5) both fail**, and CCA fails more.
5. **CCA costs 3–100 times more time** than PCA.

**Conclusion:** the extra cost of CCA pays off only in one case (nearest-neighbour on Ext.
Yale B). Otherwise PCA is as good or better and much faster. This supports the project's
main question: nonlinear is not automatically better.

**Note:** our PCA numbers are a little lower than the team's midterm slides (e.g. LDA on
lighting subset 3: 69.4 % vs 81.4 %) because our image cropping is simpler. Compare CCA with
the PCA column of **this** script, which used exactly the same images.
