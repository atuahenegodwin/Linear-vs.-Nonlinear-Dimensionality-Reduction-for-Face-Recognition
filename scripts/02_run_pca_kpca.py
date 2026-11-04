"""
Run PCA, polynomial Kernel PCA and RBF Kernel PCA on every protocol in the README.

  orl          3-6 train images/person, 10 random splits each
  yale         leave-one-condition-out (11 folds)
  eyb_random   10/20/30 train images/person, 5 random splits each
  eyb_light    train on lighting subset 1, test on subsets 2, 3, 4, 5

Outputs
  results/pca_kpca/accuracy_long.csv   one row per (protocol, split, method, classifier)
  results/pca_kpca/predictions.pkl     per-image predictions (needed for McNemar)

Usage:  python scripts/02_run_pca_kpca.py [--only orl yale eyb_random eyb_light]
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from pca_kpca.core import (classifiers, load_npz, make_reducer, split_per_person, tune, D)  # noqa: E402

METHODS = ("pca", "kpca_poly", "kpca_rbf")
OUT = "results/pca_kpca"
os.makedirs(OUT, exist_ok=True)


def fit_predict(X, y, tr, te):
    """Fit each reducer on the training images, return predictions on test images."""
    out = {}
    for m in METHODS:
        t0 = time.time()
        params = tune(m, X[tr], y[tr])
        red = make_reducer(m, params, d=min(D, len(tr) - 1))
        Ztr = red.fit_transform(X[tr])
        Zte = red.transform(X[te])
        fit_time = time.time() - t0
        for cname, clf in classifiers().items():
            clf.fit(Ztr, y[tr])
            out[(m, cname)] = dict(pred=clf.predict(Zte), fit_time=fit_time, params=str(params))
    return out


def record(rows, preds, protocol, setting, split, y, tr, te, res, test_group=None):
    for (m, c), r in res.items():
        acc = float(np.mean(r["pred"] == y[te]))
        rows.append(dict(protocol=protocol, setting=setting, split=split, method=m,
                         classifier=c, accuracy=acc, n_train=len(tr), n_test=len(te),
                         fit_time_s=r["fit_time"], params=r["params"]))
        preds[(protocol, setting, split, m, c)] = dict(pred=r["pred"], truth=y[te])


def run_orl(rows, preds):
    X, y = load_npz("data/processed/orl.npz")
    for n in (3, 4, 5, 6):
        for r in range(10):
            tr, te = split_per_person(y, n, np.random.RandomState(r))
            record(rows, preds, "orl", f"{n}/person", r, y, tr, te, fit_predict(X, y, tr, te))
        print(f"orl {n}/person done", flush=True)


def run_yale(rows, preds):
    z = np.load("data/processed/yale.npz")
    X, y, cond = z["images"].reshape(165, -1).astype(np.float64), z["labels"], z["cond"]
    for k in range(11):
        tr, te = np.where(cond != k)[0], np.where(cond == k)[0]
        record(rows, preds, "yale", "leave-one-condition-out", k, y, tr, te,
               fit_predict(X, y, tr, te))
        print(f"yale fold {k + 1}/11 ({z['cond_names'][k]}) done", flush=True)


def run_eyb_random(rows, preds):
    z = np.load("data/processed/eyb.npz")
    X, y = z["images"].reshape(len(z["labels"]), -1).astype(np.float64), z["labels"]
    for n in (10, 20, 30):
        for r in range(5):
            tr, te = split_per_person(y, n, np.random.RandomState(100 + r))
            record(rows, preds, "eyb_random", f"{n}/person", r, y, tr, te,
                   fit_predict(X, y, tr, te))
        print(f"eyb_random {n}/person done", flush=True)


def run_eyb_light(rows, preds):
    z = np.load("data/processed/eyb.npz")
    X, y, s = z["images"].reshape(len(z["labels"]), -1).astype(np.float64), z["labels"], z["subset"]
    tr = np.where(s == 1)[0]
    # fit once, test on every subset separately
    for test_subset in (2, 3, 4, 5):
        te = np.where(s == test_subset)[0]
        record(rows, preds, "eyb_light", f"train S1 -> test S{test_subset}", 0, y, tr, te,
               fit_predict(X, y, tr, te))
    print("eyb_light done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["orl", "yale", "eyb_random", "eyb_light"])
    a = ap.parse_args()
    rows, preds = [], {}
    runners = dict(orl=run_orl, yale=run_yale, eyb_random=run_eyb_random, eyb_light=run_eyb_light)
    for name in a.only:
        runners[name](rows, preds)
        pd.DataFrame(rows).to_csv(f"{OUT}/accuracy_long.csv", index=False)   # checkpoint
        with open(f"{OUT}/predictions.pkl", "wb") as f:
            pickle.dump(preds, f)
    print(f"saved {OUT}/accuracy_long.csv and {OUT}/predictions.pkl")
