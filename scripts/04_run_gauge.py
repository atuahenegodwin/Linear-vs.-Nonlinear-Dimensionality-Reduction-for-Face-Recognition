"""
Curvature gauge G (Team Outliers proposal, Section 4.2).

For each training set we fit PCA and Isomap and measure, at several embedding
dimensions d, how much of the distance structure each one loses:

    RV_m(d) = 1 - R^2( D_hat_m , D_Y,m(d) )          (Tenenbaum et al., 2000, note 42)

    D_hat_PCA     Euclidean distances between the original pixel vectors
    D_hat_Isomap  geodesic (shortest-path) distances in Isomap's k-NN graph
    D_Y,m(d)      Euclidean distances between the d-dimensional embedded points
    R             Pearson correlation over all n(n-1)/2 pairs

The gauge is the largest gap between the two curves:

    G = max_d | RV_PCA(d) - RV_Isomap(d) |,   d in {2, 5, 10, 20, 40, 60}

We also report the signed version, which only counts gaps where Isomap is
BETTER than PCA (evidence of curvature):

    G+ = max(0, max_d ( RV_PCA(d) - RV_Isomap(d) ))

If Isomap is worse than PCA, |gap| can still be large, but that is not a sign
of curvature, so G+ is the safer number to read.

Only training images are used, exactly the same training sets as the CCA
experiment (scripts/03_run_cca.py), so no test information is involved.

Isomap's neighbourhood size k is swept over 5, 10 and 15. A k whose graph is
disconnected is skipped, because geodesic distances are then undefined.

Run
---
    python scripts/04_run_gauge.py              # all training splits, a few minutes
    python scripts/04_run_gauge.py --zip "path/to/raw.zip"

Outputs (in results/gauge/)
    gauge_per_split.csv   one row per training set x k: G, d*, RV curves
    gauge_summary.csv     mean and SD of G per setting x k
    gauge_vs_cca.csv      G next to the measured CCA - PCA accuracy gains
    rv_curves.png         RV_PCA(d) and RV_Isomap(d) per setting
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                 # reuse loading and splits
from cca.data import DEFAULT_ZIP, all_splits, load_all  # noqa: E402

DIMS = [2, 5, 10, 20, 40, 60]
KS = [5, 10, 15]


def residual_variance(d_hat, Y):
    """1 - R^2 between reference distances and embedding distances."""
    r = np.corrcoef(d_hat, pdist(Y))[0, 1]
    return 1.0 - r ** 2


def rv_curves(X, k):
    """RV_PCA(d) and RV_Isomap(d) for every d in DIMS, or None if the graph is disconnected."""
    if connected_components(kneighbors_graph(X, k), directed=False)[0] > 1:
        return None
    dmax = max(DIMS)
    # Leading components do not change with the total number kept, so one
    # fit at d = 60 gives every smaller d by keeping the first d columns.
    Y_pca = PCA(n_components=dmax).fit_transform(X)
    iso = Isomap(n_neighbors=k, n_components=dmax, eigen_solver="dense")
    Y_iso = iso.fit_transform(X)
    iu = np.triu_indices(len(X), k=1)
    d_euc = pdist(X)
    d_geo = iso.dist_matrix_[iu]
    rv_pca = [residual_variance(d_euc, Y_pca[:, :d]) for d in DIMS]
    rv_iso = [residual_variance(d_geo, Y_iso[:, :d]) for d in DIMS]
    return np.array(rv_pca), np.array(rv_iso)


def training_sets(data):
    """The same training sets as the CCA experiment. The lighting test has one
    training set (subset 1) shared by all four test subsets."""
    seen_light = False
    for dataset, setting, split, tr, _ in all_splits(data, 10):
        if setting.startswith("ExtYaleB light"):
            if seen_light:
                continue
            seen_light, setting = True, "ExtYaleB light"
        yield dataset, setting, split, tr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="path to the team's raw.zip")
    args = ap.parse_args()
    if not args.zip.exists():
        raise SystemExit(f"raw.zip not found at {args.zip}\nPass it with --zip \"path\\to\\raw.zip\"")

    print("Loading images from", args.zip)
    data = load_all(args.zip)
    out = ROOT / "results" / "gauge"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, setting, split, tr in training_sets(data):
        X = data[dataset]["X"][tr]
        X = X - X.mean(axis=0)
        for k in KS:
            res = rv_curves(X, k)
            if res is None:
                print(f"  {setting:16s} split {split:2d}  k={k:2d}  graph disconnected, skipped")
                continue
            rv_pca, rv_iso = res
            gap = np.abs(rv_pca - rv_iso)
            signed = rv_pca - rv_iso                      # > 0: Isomap better
            row = dict(setting=setting, split=split, n_train=len(tr), k=k,
                       G=gap.max(), d_star=DIMS[int(gap.argmax())],
                       G_plus=max(0.0, signed.max()), d_star_plus=DIMS[int(signed.argmax())])
            row.update({f"RV_PCA_{d}": v for d, v in zip(DIMS, rv_pca)})
            row.update({f"RV_Isomap_{d}": v for d, v in zip(DIMS, rv_iso)})
            rows.append(row)
            print(f"  {setting:16s} split {split:2d}  k={k:2d}  G={row['G']:.3f}  d*={row['d_star']}  G+={row['G_plus']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "gauge_per_split.csv", index=False)
    summary = (df.groupby(["setting", "k"], sort=False)
                 .agg(G_mean=("G", "mean"), G_sd=("G", "std"), d_star=("d_star", lambda s: s.mode()[0]),
                      G_plus_mean=("G_plus", "mean"), d_star_plus=("d_star_plus", lambda s: s.mode()[0]),
                      RV_PCA_60=("RV_PCA_60", "mean"), RV_Isomap_60=("RV_Isomap_60", "mean"),
                      n_splits=("G", "size"))
                 .reset_index())
    summary.to_csv(out / "gauge_summary.csv", index=False)
    pd.set_option("display.width", 140)
    print("\nGauge G per setting (mean over training splits)\n")
    print(summary.round(3).to_string(index=False))

    compare(summary, out)
    plot(df, out)


def compare(summary, out):
    """Put G (k = 10) next to the CCA - PCA gains measured by scripts/03_run_cca.py."""
    cca_file = ROOT / "results" / "cca" / "cca_face_summary.csv"
    if not cca_file.exists():
        print("\n(No CCA results found; run scripts/03_run_cca.py first to compare.)")
        return
    acc = pd.read_csv(cca_file)
    acc["setting"] = acc["setting"].where(~acc["setting"].str.startswith("ExtYaleB light"), "ExtYaleB light")
    piv = acc.pivot_table(index=["setting", "classifier"], columns="method", values="mean_acc").reset_index()
    piv["gain"] = (piv["CCA"] - piv["PCA"]) * 100
    gains = piv.pivot_table(index="setting", columns="classifier", values="gain")[["1-NN", "LDA"]]
    best = acc.groupby(["setting", "method"])["mean_acc"].max().unstack()
    gains["best CCA - best PCA"] = (best["CCA"] - best["PCA"]) * 100
    g = summary[summary.k == 10].set_index("setting")[["G_mean", "d_star", "G_plus_mean", "d_star_plus", "RV_PCA_60", "RV_Isomap_60"]]
    table = g.join(gains).sort_values("G_mean", ascending=False)
    table.to_csv(out / "gauge_vs_cca.csv")
    print("\nG (k = 10) next to the measured CCA - PCA gains (percentage points)\n")
    print(table.round(3).to_string())
    for g in ["G_mean", "G_plus_mean"]:
        for col in ["1-NN", "LDA", "best CCA - best PCA"]:
            rho = spearmanr(table[g], table[col]).correlation
            print(f"Spearman({g}, {col} gain) = {rho:+.2f}   (only {len(table)} settings: a rough check)")


def plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = list(dict.fromkeys(df.setting))
    fig, axes = plt.subplots(1, len(settings), figsize=(3.6 * len(settings), 3.2), sharey=True)
    sub = df[df.k == 10]
    for ax, s in zip(np.atleast_1d(axes), settings):
        d = sub[sub.setting == s]
        p = d[[f"RV_PCA_{x}" for x in DIMS]].mean().values
        i = d[[f"RV_Isomap_{x}" for x in DIMS]].mean().values
        ax.plot(DIMS, p, "o-", label="PCA")
        ax.plot(DIMS, i, "o-", label="Isomap (k = 10)")
        ax.set_title(f"{s}\nG = {d.G.mean():.3f},  G+ = {d.G_plus.mean():.3f}", fontsize=10)
        ax.set_xlabel("dimension d")
    np.atleast_1d(axes)[0].set_ylabel("residual variance")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(out / "rv_curves.png", dpi=150)
    print("\nSaved results/gauge/gauge_per_split.csv, gauge_summary.csv, gauge_vs_cca.csv, rv_curves.png")


if __name__ == "__main__":
    main()
