"""
Curvilinear Component Analysis (CCA)
====================================

Implementation of Curvilinear Component Analysis from

    P. Demartines and J. Herault (1997). "Curvilinear Component Analysis:
    A Self-Organizing Neural Network for Nonlinear Mapping of Data Sets."
    IEEE Transactions on Neural Networks, 8(1), 148-154.

Not to be confused with Canonical Correlation Analysis, which is also
abbreviated CCA (sklearn.cross_decomposition.CCA).

Idea
----
CCA looks for low-dimensional points Y whose pairwise distances match the
pairwise distances of the original data X, but only for pairs that are close
together in the output space. Far-apart pairs are allowed to be wrong, which
lets CCA "unfold" a curved surface instead of squashing it like PCA does.

It minimises the stress

    E = 1/2 * sum_{i<j} (Dx_ij - Dy_ij)^2 * F(Dy_ij, lam)

    Dx_ij : Euclidean distance between images i and j in the input space
    Dy_ij : Euclidean distance between their embeddings y_i and y_j
    F     : neighbourhood weight, 1 if Dy_ij < lam and 0 otherwise

The neighbourhood radius lam starts large (all pairs count, so the result
behaves like a global distance-preserving map) and shrinks during training
(only local distances count, so the map can unfold curvature).

Training uses the stochastic rule of the original paper: pick a point i,
keep y_i fixed, and move every other point j along the line through y_i:

    y_j <- y_j + alpha * F(Dy_ij, lam) * (Dx_ij - Dy_ij) / Dy_ij * (y_j - y_i)

If j is too close to i in the embedding (Dx_ij > Dy_ij) it is pushed away,
and if it is too far it is pulled in.

Usage
-----
    from cca import CurvilinearComponentAnalysis      # after sys.path.insert(0, "src")

    cca = CurvilinearComponentAnalysis(n_components=10, random_state=0)
    Z_train = cca.fit_transform(X_train)   # fit on training images only
    Z_test = cca.transform(X_test)          # project new images

The class follows the scikit-learn fit / transform interface, so it can be
used in the same loops and pipelines as PCA, KernelPCA, LLE and Isomap.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted


class CurvilinearComponentAnalysis(TransformerMixin, BaseEstimator):
    """Curvilinear Component Analysis (Demartines & Herault, 1997).

    Parameters
    ----------
    n_components : int, default=2
        Dimension d of the embedding.

    n_epochs : int, default=50
        Number of passes over the training points. Each epoch visits every
        point once, in random order, as the fixed point i.

    alpha : tuple of float, default=(0.5, 0.005)
        Learning rate at the first and last epoch. It decays exponentially
        between the two values.

    lam : tuple of (float or None), default=(None, None)
        Neighbourhood radius at the first and last epoch, in the units of the
        input distances. It decays exponentially between the two values.
        None picks a value from the data: the start defaults to the largest
        input distance (every pair counts at first), and the end defaults to
        the 5th percentile of the input distances (only near neighbours count
        at the end). The end radius is the setting that matters most: too
        small and the map tears apart, too large and it stays close to PCA.

    init : {"pca", "random"}, default="pca"
        Starting embedding. PCA is deterministic and usually converges faster.

    transform_steps : int, default=100
        Gradient steps used to place each new point in `transform`.

    random_state : int or None, default=None
        Seed for the visiting order and for random initialisation.

    verbose : bool, default=False
        Print the stress after every epoch.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Embedding of the training data.

    stress_ : float
        Final stress E, computed with the final radius.

    stress_history_ : list of float
        Stress after each epoch, each with that epoch's radius. Useful for
        checking convergence.

    lam_ : tuple of float
        The (start, end) radius actually used.
    """

    def __init__(
        self,
        n_components=2,
        n_epochs=50,
        alpha=(0.5, 0.005),
        lam=(None, None),
        init="pca",
        transform_steps=100,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.lam = lam
        self.init = init
        self.transform_steps = transform_steps
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Learn the embedding of the training data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training images, one flattened image per row (or their
            PCA-reduced features).
        y : ignored
            Present for scikit-learn compatibility. CCA does not use labels.
        """
        X = check_array(X, dtype=np.float64)
        n = X.shape[0]
        if not 1 <= self.n_components < X.shape[1]:
            raise ValueError("n_components must be between 1 and n_features - 1")
        rng = np.random.default_rng(self.random_state)

        # Input distances never change, so compute them once.
        Dx = squareform(pdist(X))

        lam_start, lam_end = self.lam
        upper = Dx[np.triu_indices(n, k=1)]
        if lam_start is None:
            lam_start = upper.max()
        if lam_end is None:
            lam_end = np.percentile(upper, 5)
        alpha_start, alpha_end = self.alpha

        Y = self._initial_embedding(X, rng)

        self.stress_history_ = []
        for epoch in range(self.n_epochs):
            # Exponential decay from the start value to the end value.
            frac = epoch / max(self.n_epochs - 1, 1)
            alpha = alpha_start * (alpha_end / alpha_start) ** frac
            lam = lam_start * (lam_end / lam_start) ** frac

            for i in rng.permutation(n):
                diff = Y - Y[i]                      # y_j - y_i for every j
                dy = np.sqrt((diff ** 2).sum(axis=1))
                active = dy < lam                    # F(Dy_ij, lam)
                active[i] = False                    # point i does not move
                # (Dx_ij - Dy_ij) / Dy_ij, only for neighbours of i. The
                # floor avoids dividing by zero for duplicate points.
                step = np.zeros(n)
                step[active] = (Dx[i, active] - dy[active]) / np.maximum(dy[active], 1e-12)
                Y += alpha * step[:, None] * diff

            self.stress_history_.append(self._stress(Dx, Y, lam))
            if self.verbose:
                print(f"epoch {epoch + 1:3d}  alpha={alpha:.4f}  "
                      f"lam={lam:.4f}  stress={self.stress_history_[-1]:.4f}")

        self.X_fit_ = X
        self.embedding_ = Y
        self.lam_ = (lam_start, lam_end)
        self.stress_ = self._stress(Dx, Y, lam_end)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_transform(self, X, y=None):
        """Fit on X and return the embedding of X."""
        return self.fit(X).embedding_

    # ------------------------------------------------------------------
    # Projecting new data
    # ------------------------------------------------------------------
    def transform(self, X, batch_size=100):
        """Place new points in the learned embedding.

        CCA has no explicit mapping from input to output, so each new point
        x is placed by minimising its own stress against the fixed training
        embedding:

            E(y) = 1/2 * sum_j (d(x, x_j) - ||y - y_j||)^2 * F(||y - y_j||, lam_end)

        The search starts at a distance-weighted average of the embeddings of
        the new point's nearest training images, and the training embedding
        does not change. This is how test images are handled without leaking
        them into training.
        """
        check_is_fitted(self, "embedding_")
        X = check_array(X, dtype=np.float64)
        Y_train = self.embedding_
        lam = self.lam_[1]

        # Work in chunks of new points to keep memory small on a laptop.
        chunks = [self._place(X[s:s + batch_size], Y_train, lam)
                  for s in range(0, len(X), batch_size)]
        return np.vstack(chunks)

    def _place(self, X, Y_train, lam):
        Dx = cdist(X, self.X_fit_)           # new points x training points
        Z = self._nearest_neighbour_start(Dx, Y_train)

        for _ in range(self.transform_steps):
            diff = Z[:, None, :] - Y_train[None, :, :]       # (m, n, d)
            dy = np.sqrt((diff ** 2).sum(axis=2))
            dy = np.maximum(dy, 1e-12)
            # Within the radius of each new point; fall back to all training
            # points if none are within it, so every point gets a gradient.
            active = dy < lam
            empty = ~active.any(axis=1)
            active[empty] = True
            coef = np.where(active, (Dx - dy) / dy, 0.0)
            # Mean move over active neighbours: a damped, stable step.
            n_active = active.sum(axis=1, keepdims=True)
            Z += (coef[:, :, None] * diff).sum(axis=1) / n_active
        return Z

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initial_embedding(self, X, rng):
        if self.init == "pca":
            return PCA(n_components=self.n_components).fit_transform(X)
        if self.init == "random":
            scale = X.std(axis=0).mean()
            return rng.normal(scale=scale, size=(X.shape[0], self.n_components))
        raise ValueError("init must be 'pca' or 'random'")

    @staticmethod
    def _nearest_neighbour_start(Dx, Y_train, k=5):
        idx = np.argsort(Dx, axis=1)[:, :k]
        w = 1.0 / np.maximum(np.take_along_axis(Dx, idx, axis=1), 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        return (w[:, :, None] * Y_train[idx]).sum(axis=1)

    @staticmethod
    def _stress(Dx, Y, lam):
        Dy = squareform(pdist(Y))
        iu = np.triu_indices(len(Y), k=1)
        err = (Dx[iu] - Dy[iu]) ** 2
        return 0.5 * float(err[Dy[iu] < lam].sum())
