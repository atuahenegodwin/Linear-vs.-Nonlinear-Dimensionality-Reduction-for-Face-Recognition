"""
CCA (and PCA as reference) on the three face datasets.

Reads the team's data/raw.zip directly (no need to unzip), reduces every
image to d = 60 features with CCA and with PCA, and classifies with the five
models from the midterm slides. Both methods use exactly the same images,
splits and classifiers, so their numbers can be compared side by side.

Protocols
---------
ORL           5 training images per person, 10 random splits
Yale          leave-one-condition-out, 11 folds (one per condition)
Ext. Yale B   random: 30 training images per person (as in the team PCA), 5 random splits
              lighting: train on subset 1, test on subsets 2-5
              (subsets by angle between light and camera: <=12, 13-25,
              26-50, 51-77, >77 degrees)

Preprocessing
-------------
grayscale -> crop to the 46:56 aspect ratio -> resize to 46 x 56 = 2,576
pixels -> scale to [0, 1]. Fitting (CCA, PCA, classifiers) uses training
images only.

Run
---
    python scripts/03_run_cca.py            # everything, about 5-10 min
    python scripts/03_run_cca.py --quick    # 2 splits each, ~2 min (saved to results/cca/quick/)
    python scripts/03_run_cca.py --zip "path/to/raw.zip"

Run from the repository root. Results are printed and saved to results/cca/:
    results/cca/cca_face_results.csv     one row per split x classifier x method
    results/cca/cca_face_summary.csv     mean and SD per dataset x classifier x method
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cca import CurvilinearComponentAnalysis                    # noqa: E402
from cca.data import DEFAULT_ZIP, all_splits, load_all          # noqa: E402

D = 60                   # number of features, as in the midterm slides


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def classifiers():
    return {
        "1-NN": KNeighborsClassifier(n_neighbors=1),
        "soft k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "LDA": LinearDiscriminantAnalysis(),
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000)),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
    }


def reducers():
    return {
        "PCA": lambda: PCA(n_components=D, random_state=0),
        "CCA": lambda: CurvilinearComponentAnalysis(n_components=D, random_state=0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="path to the team's raw.zip")
    ap.add_argument("--quick", action="store_true", help="2 random splits per dataset, for a fast check")
    args = ap.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"raw.zip not found at {args.zip}\nPass it with --zip \"path\\to\\raw.zip\"")

    print("Loading images from", args.zip)
    data = load_all(args.zip)
    for k, v in data.items():
        print(f"  {k:9s} {v['X'].shape[0]:5d} images, {len(np.unique(v['y']))} people")

    out = ROOT / "results" / "cca" / ("quick" if args.quick else "")   # --quick never overwrites the full results
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, setting, split, tr, te in all_splits(data, 2 if args.quick else 10):
        if args.quick and setting == "Yale" and split >= 2:
            continue
        X, y = data[dataset]["X"], data[dataset]["y"]
        mean = X[tr].mean(axis=0)               # centre with training images only
        Xtr, Xte = X[tr] - mean, X[te] - mean
        for method, make in reducers().items():
            t0 = time.perf_counter()
            red = make()
            Ztr = red.fit_transform(Xtr)
            Zte = red.transform(Xte)
            seconds = time.perf_counter() - t0
            for clf_name, clf in classifiers().items():
                acc = (clf.fit(Ztr, y[tr]).predict(Zte) == y[te]).mean()
                rows.append(dict(setting=setting, split=split, method=method,
                                 classifier=clf_name, accuracy=acc, seconds=seconds))
        last = [r for r in rows if r["setting"] == setting and r["split"] == split]
        best = {m: max(r["accuracy"] for r in last if r["method"] == m) for m in ("PCA", "CCA")}
        print(f"  {setting:30s} split {split:2d}   best PCA {best['PCA']:.3f}   best CCA {best['CCA']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "cca_face_results.csv", index=False)
    summary = (df.groupby(["setting", "classifier", "method"], sort=False)
                 .agg(mean_acc=("accuracy", "mean"), sd_acc=("accuracy", "std"),
                      n_splits=("accuracy", "size"), mean_seconds=("seconds", "mean"))
                 .reset_index())
    summary.to_csv(out / "cca_face_summary.csv", index=False)

    table = summary.pivot_table(index=["setting", "classifier"], columns="method",
                                values="mean_acc", sort=False)[["PCA", "CCA"]]
    table["CCA - PCA"] = table["CCA"] - table["PCA"]
    pd.set_option("display.width", 120)
    print("\nMean test accuracy (d = 60)\n")
    print((table * 100).round(1).to_string())
    print("\nAverage fit + transform time (s):")
    print(df.groupby(["setting", "method"], sort=False)["seconds"].mean().unstack().round(2).to_string())
    print(f"\nSaved {out / 'cca_face_results.csv'} and {out / 'cca_face_summary.csv'}")


if __name__ == "__main__":
    main()
