# Linear vs. Nonlinear Dimensionality Reduction for Face Recognition

**Team Outliers · CS 7300 (Fall 2026)** — Anisha Raju Vachavaya, Salma Hasannejad, Godwin Atuahene, Cadence Litteral

We compare PCA with six nonlinear dimensionality-reduction methods on three face databases under one
identical protocol, for both **recognition** (classification) and **grouping without labels** (clustering), and
we test whether a **curvature gauge**, computed from training images alone, can tell in advance whether a
dataset needs a nonlinear method. The study reproduces and extends Huang and Yin (2009).

## Results in brief

* **Nonlinear reduction rarely beat PCA.** Across 270 paired comparisons (6 nonlinear methods × 9 dataset
  settings × 5 classifiers, Holm-adjusted), a nonlinear method was significantly better 9 times and significantly
  worse 88 times. All 9 wins were on Extended Yale B with nearest-neighbour classifiers (CCA up to +14 points with
  1-NN); with LDA or an SVM, PCA was the best inductive choice on every dataset.
* **ORL and Yale reproduce Huang and Yin (2009):** PCA within 1.6 points of their numbers on average; no nonlinear
  method significantly better than PCA. Inductive LLE and Isomap fall short of the published values because placing
  unseen faces is their weak step; fitting them transductively recovers most of the gap.
* **Clustering is where nonlinear methods helped most**, under heavy lighting: on Extended Yale B, NMI 0.42–0.47
  (LLE) and 0.35–0.46 (autoencoder) against 0.16–0.30 (PCA). Label-free rules for the number of clusters
  (eigengap, stability) picked 2–4 groups instead of 15–40 people.
* **The curvature gauge does not predict the gain.** It reads 0.58 on a Swiss roll and 0.006 on a flat plane, but
  across 14 face dataset settings its Spearman correlation with the real nonlinear gain is −0.14 (p = 0.63); a
  pipeline using it with leave-one-dataset-out cut-offs did no better than always using PCA. On faces Isomap's
  residual variance levels off above PCA's, so the sign-ignored gap mostly measures that floor.
* **Cost:** at these sizes kernel PCA, LLE and Isomap fit as fast as PCA (< 0.25 s); CCA takes up to 3.6 s and the
  autoencoder up to 10.6 s.

Full write-up: the project report, a Claude doc at
https://claude.ai/code/artifact/05d53157-56fc-4328-818c-562f329db07b (exports to Word or PDF).
Replication check: `results/replication_check.md`.

The notebooks in `notebooks/` show every result with the code that produced it.

---

## What is in this folder

```
Outliers_DR_Project/
├── README.md                  this file
├── requirements.txt           package versions used for the reported results
├── run_all.sh                 reproduce everything (download → results → notebooks → check), ~1 h on an M1
├── src/outliers/              the library
│   ├── config.py              paths and every experiment setting (resolution, d, seeds, splits)
│   ├── data.py                loading, Yale GIF→PNG conversion, face cropping, Ext. Yale B cleaning
│   ├── splits.py              train/test splits and validation folds (shared by all methods)
│   ├── reducers.py            the seven reducers behind one interface + their tuning grids
│   ├── cca.py                 curvilinear component analysis (Demartines and Hérault, 1997)
│   ├── isomap.py              Isomap with a robust out-of-sample extension
│   ├── autoencoder.py         2576–128–d–128–2576 tanh autoencoder (Adam)
│   ├── classifiers.py         1-NN, soft k-NN, LDA, linear SVM, RBF SVM
│   ├── clustering.py          k-means, spectral clustering, eigengap and stability for k
│   ├── gauge.py               the curvature gauge and Huang and Yin's NLR
│   ├── stats.py               corrected resampled t-test, McNemar, Holm, Spearman permutation test
│   ├── experiments.py         one function per experiment task (resumable)
│   ├── tasks.py               the full list of tasks
│   ├── analysis.py            tables and tests from the raw results
│   └── plotting.py            figures
├── scripts/                   run in order (00 → 08); each is resumable
├── notebooks/                 01 data · 02 classification · 03 clustering · 04 curvature gauge · 05 cost
├── data/raw/                  downloaded archives and extracted images (not for redistribution)
├── data/processed/            46 × 56 arrays (.npz), Yale PNGs + metadata, Ext. Yale B exclusion list
├── results/raw/               one file per task (accuracies, per-image predictions, tuning, gauge curves)
├── results/tables/            every table as CSV and Markdown
├── results/figures/           every figure (PNG)
└── logs/                      run logs
```

## How to run

```bash
pip install -r requirements.txt
./run_all.sh            # or run the scripts one by one:
python -I scripts/00_download_data.py        # ORL, Yale, Extended Yale B (checksums verified)
python -I scripts/01_prepare_data.py         # crop, resize to 46 x 56, clean, save .npz
python -I scripts/02_run_classification.py --jobs 3
python -I scripts/03_run_dimension_sweep.py --jobs 3
python -I scripts/04_run_clustering.py --jobs 3
python -I scripts/05_run_cost.py             # timings: run alone, one thread
python -I scripts/06_analyze.py              # tables + figures
python -I scripts/07_build_notebooks.py      # executes the notebooks
python -I scripts/08_verify_replication.py   # reruns a sample from scratch and compares
```

Measured on an 8 GB Apple M1 with 3 workers: classification 34 min, dimension sweep 9 min, clustering 5 min,
cost benchmark 2 min, data preparation, analysis and notebooks about 3 min together (about 55 min in total,
plus the downloads).

Every script skips work whose output already exists, so an interrupted run can simply be restarted.
On an 8 GB machine use **3 workers**: with 6, the processes competed for memory and the run was about
four times slower.

## Data

| Dataset | Source | People | Images used | Varies |
|---|---|---|---|---|
| ORL (AT&T) | [AT&T Laboratories Cambridge archive](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) | 40 | 400 | pose, expression, glasses |
| Yale | [Kaggle mirror](https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database) of the Yale Face Database | 15 | 165 | lighting, expression, glasses |
| Extended Yale B | [Kaggle mirror](https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full), frontal pose | 27 | 1,609 | 64 lighting directions |

Preprocessing (identical for all datasets): grayscale → crop to the face → resize to 46 × 56 (2,576 pixels)
→ scale to [0, 1]; mean-centring uses training images only. Details and checks are in notebook 01:

* **Yale**: files have no extension (`subject01.happy`); headers are checked (all GIF), files are converted to PNG
  with a metadata CSV. Faces are cropped from the head's silhouette against the bright wall; the side-lit photos
  reuse the same person's frame from their other photos.
* **Extended Yale B**: the official CroppedYale download is no longer online; the Kaggle copy holds 28 subjects
  (yaleB11–yaleB39) in 9 poses. We use the frontal pose. yaleB16 is excluded (no frontal images; its pose labels
  appear shifted). 119 of 1,728 images (6.9 %) are excluded as failed shots or inconsistent crops, by an explicit
  rule that compares each image with the same shot of the other subjects; see `data/processed/eyb_exclusions.csv`.

Please cite the databases if you reuse them: Samaria and Harter (1994) for ORL; Belhumeur, Hespanha and Kriegman
(1997) for Yale; Georghiades, Belhumeur and Kriegman (2001) and Lee, Ho and Kriegman (2005) for Extended Yale B.

## Methods in one paragraph

Seven reducers at **d = 60** (PCA; kernel PCA, polynomial and RBF; LLE; Isomap; curvilinear component analysis;
an autoencoder), each with at most three settings chosen by 1-NN accuracy on validation folds inside the training
set. Five classifiers on every reduced space (1-NN, soft k-NN, LDA, linear SVM, RBF SVM), their settings chosen on
the same folds. Protocols: ORL with 3–6 training images per person (10 random splits each, as in Huang and Yin);
Yale leave-one-condition-out (11 folds); Extended Yale B with 10/20/30 per person (5 splits) and a lighting
protocol (train on subset 1, test on subsets 2–5). Clustering: k-means and spectral clustering on reduced faces,
scored with NMI/ARI, with *k* both fixed and estimated (eigengap + stability). Significance: corrected resampled
t-test (Nadeau and Bengio, 2003) for random splits, exact McNemar on per-image predictions for Yale and the
lighting protocol, Holm-adjusted. Curvature gauge: max over d of |RV_PCA(d) − RV_Iso(d)| with residual variance
from Tenenbaum et al. (2000), validated on known shapes, against the real gains, with leave-one-dataset-out
cut-offs, and on ORL faces rotated by controlled amounts.

## Deviations from the proposal (and why)

* **Resolution**: images are 46 × 56 (half of ORL's 92 × 112) for every dataset, to keep the experiment grid
  affordable on a laptop; there are still far more pixels (2,576) than training images.
* **Autoencoder**: PyTorch was not available, so the autoencoder is trained with scikit-learn's `MLPRegressor`
  (Adam, tanh, 2576–128–d–128–2576). The design slide's "PyTorch" line should read "scikit-learn".
* **Extended Yale B**: 27 subjects instead of 38 (see Data). The pose images were not used: their crops are
  framed differently from person to person, which would confound a curvature test.
* **Isomap**: our own implementation (`isomap.py`) keeps only positive-eigenvalue directions; scikit-learn's
  raises an error when many dimensions are requested from few faces. It matches scikit-learn exactly where both run.
* **Added checks**: transductive fitting of LLE/Isomap/CCA, the rotated-ORL dose–response test of the gauge, and a
  signed version of the gauge (added after seeing that Isomap's residual variance levels off above PCA's on faces).

## References

* Huang, W. and Yin, H. (2009). Linear and nonlinear dimensionality reduction for face recognition. *ICIP*, 3337–3340.
* Tenenbaum, J. B., de Silva, V. and Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. *Science* 290, 2319–2323.
* Roweis, S. T. and Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. *Science* 290, 2323–2326.
* Schölkopf, B., Smola, A. and Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation* 10, 1299–1319.
* Demartines, P. and Hérault, J. (1997). Curvilinear component analysis. *IEEE Trans. Neural Networks* 8, 148–154.
* de Silva, V. and Tenenbaum, J. B. (2004). Sparse multidimensional scaling using landmark points. Stanford technical report.
* Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning* 52, 239–281.
* Keller, J. M., Gray, M. R. and Givens, J. A. (1985). A fuzzy k-nearest neighbor algorithm. *IEEE Trans. SMC* 15, 580–585.
* Turk, M. and Pentland, A. (1991). Eigenfaces for recognition. *J. Cognitive Neuroscience* 3, 71–86.
