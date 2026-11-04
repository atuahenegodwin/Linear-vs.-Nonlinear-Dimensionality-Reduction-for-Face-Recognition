"""
Compare Kernel PCA (poly, RBF) against PCA, per classifier and protocol setting.

  * random splits (orl, eyb_random): corrected resampled t-test (Nadeau & Bengio 2003)
        t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)),  df = J - 1
  * fixed splits (yale, eyb_light): exact McNemar test on per-image predictions
        (pooled over the 11 folds for Yale)
  * all p-values Holm-adjusted together, two-sided

Outputs  results/pca_kpca/summary_table.csv  and  results/pca_kpca/significance.csv
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats

long = pd.read_csv("results/pca_kpca/accuracy_long.csv")
preds = pickle.load(open("results/pca_kpca/predictions.pkl", "rb"))
BASE, OTHERS = "pca", ["kpca_poly", "kpca_rbf"]


def corrected_t(diff, n_train, n_test):
    J = len(diff)
    v = np.var(diff, ddof=1)
    if v == 0:
        return 0.0, 1.0
    t = np.mean(diff) / np.sqrt((1 / J + n_test / n_train) * v)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def mcnemar_exact(a_correct, b_correct):
    """a = PCA, b = other. Returns (b_only_right, a_only_right, p)."""
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))
    n = only_a + only_b
    p = 1.0 if n == 0 else stats.binomtest(min(only_a, only_b), n, 0.5).pvalue
    return only_b, only_a, p


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


rows = []
for (protocol, setting), g in long.groupby(["protocol", "setting"], sort=False):
    for clf in g.classifier.unique():
        base = g[(g.method == BASE) & (g.classifier == clf)].sort_values("split")
        for m in OTHERS:
            oth = g[(g.method == m) & (g.classifier == clf)].sort_values("split")
            diff = oth.accuracy.values - base.accuracy.values
            row = dict(protocol=protocol, setting=setting, classifier=clf, method=m,
                       pca_acc=100 * base.accuracy.mean(), method_acc=100 * oth.accuracy.mean(),
                       diff_pts=100 * diff.mean())
            if protocol in ("orl", "eyb_random"):
                t, p = corrected_t(diff, base.n_train.iloc[0], base.n_test.iloc[0])
                row.update(test="corrected t", stat=t, p=p)
            else:
                a = np.concatenate([preds[(protocol, setting, s, BASE, clf)]["pred"] ==
                                    preds[(protocol, setting, s, BASE, clf)]["truth"]
                                    for s in base.split])
                b = np.concatenate([preds[(protocol, setting, s, m, clf)]["pred"] ==
                                    preds[(protocol, setting, s, m, clf)]["truth"]
                                    for s in base.split])
                only_b, only_a, p = mcnemar_exact(a, b)
                row.update(test="McNemar", stat=only_b - only_a, p=p,
                           pca_acc=100 * a.mean(), method_acc=100 * b.mean(),
                           diff_pts=100 * (b.mean() - a.mean()))
            rows.append(row)

sig = pd.DataFrame(rows)
sig["p_holm"] = holm(sig.p.values)
sig["verdict"] = np.where(sig.p_holm < 0.05,
                          np.where(sig.diff_pts > 0, "better than PCA", "worse than PCA"),
                          "no sig. difference")
sig = sig.round({"pca_acc": 2, "method_acc": 2, "diff_pts": 2, "stat": 3, "p": 4, "p_holm": 4})
sig.to_csv("results/pca_kpca/significance.csv", index=False)

# mean accuracy table, wide
summ = (long.groupby(["protocol", "setting", "classifier", "method"]).accuracy.agg(["mean", "std"])
        .mul(100).round(2).reset_index())
summ.to_csv("results/pca_kpca/summary_table.csv", index=False)

print(f"{len(sig)} comparisons  (Holm-adjusted, alpha = 0.05)")
print(sig.verdict.value_counts().to_string())
print()
print(sig[sig.verdict != "no sig. difference"]
      [["protocol", "setting", "classifier", "method", "pca_acc", "method_acc", "diff_pts", "p_holm"]]
      .to_string(index=False))
