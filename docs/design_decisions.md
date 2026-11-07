# Design decisions and checks (CCA part)

Every choice we made in the CCA code, why we made it, and what we checked.
Written in plain English. Code references point to `src/cca/` and `scripts/`.

---

## 1. What the task is

* **Input:** face photos of many people, each 46 × 56 = 2,576 pixels.
* **Output:** the name (person id) of each test photo.
* **Goal:** recognition. We do **not** rebuild (reconstruct) the photos. CCA turns each photo
  into 60 numbers (a point on a "map"), and a classifier names the photo from those 60 numbers.
* CCA itself is **unsupervised**: it never sees names. Only the classifier uses names.

## 2. Preprocessing (`src/cca/data.py`)

* Grayscale, then crop to the 46:56 shape, resize to 46 × 56, and scale pixels to [0, 1].
* **Centring uses the training photos' mean only**, so the test photos don't influence it.

| Dataset | Photos | People | Train / test per split |
|---|---|---|---|
| ORL | 400 | 40 | 200 / 200 |
| Yale | 165 | 15 | 150 / 15 |
| Extended Yale B | 1,728 (27 people × 64 lights) | 27 | 810 / 918 |
| Ext. Yale B lighting | same | 27 | 189 (subset 1) / 1,539 (subsets 2–5) |

## 3. Splits and leakage (`src/cca/data.py`)

* **Random splits vs folds.** ORL and Ext. Yale B use *random* splits (10 and 5), averaged so one
  lucky or unlucky split doesn't decide the result. Yale uses **leave-one-condition-out** (11 folds):
  each fold holds out one condition (e.g. "glasses") as the test set, so it tests a condition the
  model has never seen.
* **Stratified split (`per_person_split`).** We pick the training photos *per person*, not from
  the whole pile. This **guarantees every person is in the training map**, so 1-NN can always
  name the right person. A random 200 from the whole pile could miss someone.
* **Lighting test:** subsets 1–5 are grouped by the angle between the light and the camera
  (≤12°, 13–25°, 26–50°, 51–77°, >77°). A bigger angle means more shadow. We train on subset 1
  (front light, 189 photos) and test on subsets 2, 3, 4 and 5 (270, 324, 432 and 513 photos) one by one.
* **30 training photos per person on Ext. Yale B**, to match the team's PCA protocol.
* **Fixed seed (0)**, so the splits and numbers reproduce exactly.
* **Paired comparison:** PCA and CCA use exactly the same splits, the same photos and the same
  classifiers, so their accuracies can be compared side by side.
* **No leakage:** in every split, CCA, PCA and the classifiers are built from scratch using
  training photos only. The test photos are never used to fit anything. A photo can be a test
  photo in one split and a training photo in another. That's fine, because each split starts fresh.

## 4. How CCA trains (`src/cca/core.py`, `fit`)

**Stress (error) minimised:**

E = ½ Σ_{i<j} (Dx_ij − Dy_ij)² · F(Dy_ij, λ),  F = 1 if Dy_ij < λ, else 0

* **Dx_ij:** the real distance between photos i and j (pixels). It's computed **once** for every
  pair of training photos (ORL: a 200 × 200 matrix, 19,900 pairs) and never changes.
* **Dy_ij:** the distance between the same two photos on the map (their 60-number points).
* **F:** an on/off switch. A pair counts only if it is closer than λ on the map.

**Update rule:** pick photo i (it stays fixed) and move every other photo j:

y_j ← y_j + α · step_j · (y_j − y_i),  step_j = (Dx_ij − Dy_ij) / max(Dy_ij, 10⁻¹²)

* step > 0: j is too close, so it's pushed away. step < 0: j is too far, so it's pulled in.

**Decisions:**

| Setting | Value | Why |
|---|---|---|
| Start (epoch 0) | **PCA** | Fast, deterministic (same every run) and already a good map, so CCA only fine-tunes. It also makes the comparison fair: CCA = PCA + extra nudging, so any change comes from the nudging. |
| d | 60 | Same as the team's PCA (midterm slides). We choose it; CCA fills in the best 60 numbers. |
| Epochs | 50 | One epoch = every photo is "i" once. The stress levels off before 50. |
| λ start | largest pairwise distance | At first every pair counts, so we get a global, PCA-like layout. |
| λ end | **5th percentile** of pairwise distances | At the end only the closest ~5% of pairs count, so CCA can unfold curves. Too small tears the map into pieces; too large keeps it like PCA. This is a standard rule of thumb and **was not tuned**; it could be tuned on validation data. |
| λ schedule | λ(t) = λ_start · (λ_end/λ_start)^(t/49) | Shrinks smoothly from global to local. |
| α (step size) | 0.5 → 0.005, same schedule | Big moves early to find the shape, tiny moves late to fine-tune (like parking a car). |
| Visiting order | shuffled each epoch (seed 0) | So no photo always goes first. |

## 5. Divide-by-zero and other safety guards

Each guard does nothing on normal data. Each one is only there so an edge case can't crash or
break the run.

| Where | Guard | Why |
|---|---|---|
| `fit`: step | `/ np.maximum(dy, 1e-12)` | Two identical photos sit on the same map spot, so Dy = 0 and the division would crash. |
| `fit`: self-pair | `active[i] = False` | Photo i is never compared with itself (distance 0) and does not move. |
| λ end | percentile of the **upper triangle** of Dx | Self-distances (the zeros on the diagonal) are left out, so λ end isn't pulled down to 0. |
| `transform`: map distance | `dy = np.maximum(dy, 1e-12)` | A test photo can land exactly on a training point. |
| `transform`: start weights | `1 / np.maximum(dist, 1e-12)` | A test photo identical to a training photo would give 1/0. |
| `transform`: empty neighbourhood | if no training point is within λ, use all of them | Otherwise that test photo would get no move at all. |

## 6. How test photos are placed (`transform`), with no leakage

1. **The training map is frozen.** The training points never move again.
2. **Start:** a distance-weighted average of the map spots of the test photo's **5 nearest** training photos.
3. **100 small steps:** move the test point so that its map distances to nearby training
   points (within λ_end) match its real pixel distances. Each step is the average move over
   those neighbours, so it is damped and stable.
4. It works like landmarks (or GPS): the training photos are landmarks with known positions.

## 7. Classifiers (`scripts/03_run_cca.py`)

* **1-NN:** j* = argmin_j ‖z − z_j‖, then copy that photo's name. Why k = 1: few photos per
  person (ORL has 5), 2-NN can tie 1–1, it uses exactly the local distances CCA tries to keep,
  and it's the standard in face recognition.
* **Ties:** exact ties are practically impossible with 60 real-valued numbers. If one happens,
  scikit-learn picks the training photo that comes first in order.
* **LDA:** maximises J(w) = wᵀS_Bw / wᵀS_Ww (far apart between people, close within a person).
  It is known to pair well with PCA (Fisherfaces, Belhumeur et al. 1997).
* We also run soft 5-NN, linear SVM and RBF SVM. All the results are in the CSV.

## 8. Curvature gauge (`scripts/04_run_gauge.py`)

* RV = 1 − R² between reference distances and map distances; **lower is better**.
* G = max_d |RV_PCA − RV_Isomap|; **G+** counts only gaps where Isomap is better. We added G+
  because the plain G is also large when Isomap is *worse*, which is not curvature.
* Isomap is used **only as a ruler**, not as a reduction method in our comparison.
* k ∈ {5, 10, 15}; a k whose neighbour graph is disconnected is skipped (geodesic distances are
  undefined).
* Only training photos are used, with the same training sets as the CCA run.

## 9. Checks we did

* **S-curve:** CCA unrolls it (trustworthiness 0.999 vs 0.931 for PCA).
* **Swiss roll:** trustworthiness 0.994 vs 0.888, but the roll tears into pieces. That is a
  known limit of CCA with straight-line distances.
* **`transform` of training photos** lands within about 1% of their fitted spots.
* **Toy example** (`examples/cca_toy_step_by_step.py`): 3 photos × 4 pixels, printing every matrix
  and move.
* **Image counts** checked in the terminal (Ext. Yale B: 1,728 = 27 × 64).
* **Reproducible:** fixed seeds. The official numbers are from Salma's own laptop run (tiny
  floating-point differences between computers are normal).
* All scripts were re-run from the repo root after the reorganisation.

## 10. Known limits

* λ end (5th percentile) and α were not tuned.
* Time and memory grow with n² (all pairs); Ext. Yale B is the slowest (about 1 minute per split on a laptop).
* On Ext. Yale B we read the raw images. The team's PCA table drops 102 badly lit images
  (`data/processed/eyb_exclusions.csv`), so its PCA numbers differ from the PCA column here.
  Compare CCA with the PCA column **in our own CSV** (same images, same splits).
