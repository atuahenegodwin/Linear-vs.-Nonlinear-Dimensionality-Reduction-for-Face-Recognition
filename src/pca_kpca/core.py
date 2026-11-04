"""
PCA and Kernel PCA (polynomial, RBF) for face recognition.

Protocol follows the project README:
  * images 46 x 56 (2576 pixels), scaled to [0, 1]
  * reduce to d = 60 dimensions; centring uses TRAINING images only
  * kernel settings (at most 3) chosen by 1-NN accuracy on validation folds
    inside the training set
  * five classifiers on every reduced space:
      1-NN, soft k-NN, LDA, linear SVM, RBF SVM
  * ORL: 3-6 training images per person, 10 random splits each

Library module (PCA / Kernel PCA reducers, tuning, classifiers). Used by
scripts/02_run_pca_kpca.py. Quick ORL-only check from the repo root:
    python src/pca_kpca/core.py --data data/processed/orl.npz --n-train 5 --repeats 3
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

D = 60  # target dimensionality


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_npz(path, x_key=None, y_key=None):
    """Load images + labels from an .npz file.

    Keys are auto-detected (X/images/data/faces and y/labels/target/ids).
    Pass x_key / y_key if your file uses other names.
    """
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    x_key = x_key or next((k for k in keys if k.lower() in
                           ("x", "images", "data", "faces", "x_all", "imgs")), None)
    y_key = y_key or next((k for k in keys if k.lower() in
                           ("y", "labels", "target", "ids", "subjects", "y_all", "person")), None)
    if x_key is None or y_key is None:
        raise KeyError(f"Could not detect keys in {path}. Found: {keys}. "
                       f"Use --x-key / --y-key.")
    X = z[x_key].astype(np.float64)
    y = z[y_key].astype(int).ravel()
    X = X.reshape(len(X), -1)           # flatten images -> (n, 2576)
    if X.max() > 1.5:                   # scale 0-255 to [0, 1]
        X = X / 255.0
    return X, y


def inspect(path):
    z = np.load(path, allow_pickle=True)
    for k in z.keys():
        print(f"{k:15s} shape={z[k].shape} dtype={z[k].dtype}")


def split_per_person(y, n_train, rng):
    """n_train random images per person for training, the rest for testing."""
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        tr.extend(idx[:n_train])
        te.extend(idx[n_train:])
    return np.array(tr), np.array(te)


# --------------------------------------------------------------------------
# Reducers
# --------------------------------------------------------------------------
def median_sqdist(X):
    """Median squared pairwise distance, used to scale the RBF width."""
    sub = X[np.random.RandomState(0).permutation(len(X))[:500]]
    d2 = euclidean_distances(sub, squared=True)
    return np.median(d2[np.triu_indices_from(d2, k=1)])


def make_reducer(name, params, d=D):
    d = d
    if name == "pca":
        return PCA(n_components=d, svd_solver="full")
    if name == "kpca_poly":
        return KernelPCA(n_components=d, kernel="poly", degree=params["degree"],
                         gamma=params["gamma"], coef0=1.0, eigen_solver="dense")
    if name == "kpca_rbf":
        return KernelPCA(n_components=d, kernel="rbf", gamma=params["gamma"],
                         eigen_solver="dense")
    raise ValueError(name)


def candidate_grid(name, Xtr):
    """At most three settings per kernel (as in the README)."""
    if name == "pca":
        return [{}]
    if name == "kpca_poly":
        g = 1.0 / Xtr.shape[1]
        return [{"degree": 2, "gamma": g}, {"degree": 3, "gamma": g},
                {"degree": 2, "gamma": 4 * g}]
    if name == "kpca_rbf":
        g = 1.0 / median_sqdist(Xtr)
        return [{"gamma": g / 2}, {"gamma": g}, {"gamma": 2 * g}]
    raise ValueError(name)


def tune(name, Xtr, ytr, n_folds=3, seed=0):
    """Pick the setting with the best 1-NN accuracy on validation folds."""
    grid = candidate_grid(name, Xtr)
    if len(grid) == 1:
        return grid[0]
    counts = np.bincount(ytr)
    n_folds = max(2, min(n_folds, counts[counts > 0].min()))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    best, best_acc = grid[0], -1
    for p in grid:
        accs = []
        for a, b in skf.split(Xtr, ytr):
            # d cannot exceed the number of fitting samples
            red = make_reducer(name, p, d=min(D, len(a) - 1))
            Za = red.fit_transform(Xtr[a])
            Zb = red.transform(Xtr[b])
            knn = KNeighborsClassifier(1).fit(Za, ytr[a])
            accs.append(knn.score(Zb, ytr[b]))
        if np.mean(accs) > best_acc:
            best, best_acc = p, np.mean(accs)
    return best


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------
def classifiers():
    return {
        "1-NN": KNeighborsClassifier(1),
        "soft k-NN": KNeighborsClassifier(5, weights="distance"),
        "LDA": LinearDiscriminantAnalysis(),
        "linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000)),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    }


# --------------------------------------------------------------------------
# One experiment = one train/test split
# --------------------------------------------------------------------------
def run_split(X, y, tr, te, methods=("pca", "kpca_poly", "kpca_rbf")):
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    rows = []
    for m in methods:
        t0 = time.time()
        params = tune(m, Xtr, ytr)
        red = make_reducer(m, params, d=min(D, len(tr) - 1))
        Ztr = red.fit_transform(Xtr)      # centring happens inside, train only
        Zte = red.transform(Xte)
        fit_time = time.time() - t0
        for cname, clf in classifiers().items():
            clf.fit(Ztr, ytr)
            rows.append({"method": m, "classifier": cname,
                         "accuracy": clf.score(Zte, yte),
                         "fit_time_s": fit_time, "params": str(params)})
    return rows


def run_orl(X, y, n_train, repeats=10, seed=0, methods=("pca", "kpca_poly", "kpca_rbf")):
    rows = []
    for r in range(repeats):
        rng = np.random.RandomState(seed + r)
        tr, te = split_per_person(y, n_train, rng)
        for row in run_split(X, y, tr, te, methods):
            row.update(n_train=n_train, repeat=r)
            rows.append(row)
        print(f"  n_train={n_train} repeat {r + 1}/{repeats} done", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["n_train", "method", "classifier"])["accuracy"]
    out = g.agg(["mean", "std"]).reset_index()
    out["mean"] *= 100
    out["std"] *= 100
    return out.rename(columns={"mean": "acc_mean_%", "std": "acc_std_%"}).round(2)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to .npz (e.g. data/processed/orl.npz)")
    ap.add_argument("--x-key", default=None)
    ap.add_argument("--y-key", default=None)
    ap.add_argument("--inspect", action="store_true", help="print array names/shapes and exit")
    ap.add_argument("--n-train", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="results/pca_kpca_results.csv")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.data)
        raise SystemExit

    X, y = load_npz(a.data, a.x_key, a.y_key)
    print(f"Loaded {X.shape[0]} images, {X.shape[1]} pixels, {len(np.unique(y))} people")

    frames = [run_orl(X, y, n, a.repeats) for n in a.n_train]
    df = pd.concat(frames, ignore_index=True)

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    summ = summarize(df)
    summ.to_csv(a.out.replace(".csv", "_summary.csv"), index=False)
    print(summ.to_string(index=False))
