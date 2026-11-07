"""
Check that the CCA implementation works before using it on faces.

Runs PCA and CCA on two classic curved datasets and saves results/cca/cca_check.png:
  - S-curve:    CCA should unroll it into a flat rectangle (it does).
  - Swiss roll: CCA with Euclidean distances tears or folds it, which is a
                known limitation of CCA (Isomap uses geodesic distances and
                handles it better). This is worth mentioning in the report.

It also prints trustworthiness (are the embedding's neighbours true
neighbours?) for each method; 1.0 is perfect.

Run (from the repository root):  python examples/cca_scurve_check.py
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cca import CurvilinearComponentAnalysis  # noqa: E402

datasets = {
    "S-curve": make_s_curve(1000, noise=0.02, random_state=0),
    "Swiss roll": make_swiss_roll(1000, noise=0.05, random_state=0),
}

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for row, (name, (X, t)) in zip(axes, datasets.items()):
    # Fit on 800 points, then project the other 200 with transform(),
    # the same way test images will be handled.
    X_train, X_test, t_train, t_test = X[:800], X[800:], t[:800], t[800:]

    Z_pca = PCA(n_components=2).fit_transform(X_train)
    cca = CurvilinearComponentAnalysis(n_components=2, random_state=0)
    Z_cca = cca.fit_transform(X_train)
    Z_test = cca.transform(X_test)

    for method, Z in [("PCA", Z_pca), ("CCA", Z_cca)]:
        tw = trustworthiness(X_train, Z, n_neighbors=10)
        print(f"{name:10s} {method}: trustworthiness = {tw:.3f}")

    row[0].scatter(*Z_pca.T, c=t_train, s=4)
    row[0].set_title(f"{name}: PCA")
    row[1].scatter(*Z_cca.T, c=t_train, s=4)
    row[1].set_title(f"{name}: CCA (training points)")
    row[2].scatter(*Z_cca.T, c="lightgrey", s=4)
    row[2].scatter(*Z_test.T, c=t_test, s=8)
    row[2].set_title(f"{name}: CCA (new points via transform)")

fig.tight_layout()
out = ROOT / "results" / "cca" / "cca_check.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=100)
print("Saved", out)
