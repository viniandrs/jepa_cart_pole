"""
Linear state probes: map any latent space to the physical state of CartPole.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PairPCA(BaseEstimator, TransformerMixin):
    """
    Project both latents of a pair [z_t, z_{t-1}] onto the principal subspace
    of the latent space: the smallest number of principal components that
    explains `variance` of the latent variance.

    Directions with negligible variance can carry a little extra state
    information on true latents, but any prediction error there is as large
    as the signal itself, so a probe that relies on them turns small latent
    errors into large state errors. Restricting the probe to the principal
    subspace makes it robust and encoder-agnostic.
    """

    def __init__(self, variance=0.99):
        self.variance = variance

    def fit(self, X, y=None):
        z_dim = X.shape[1] // 2
        singles = np.concatenate([X[:, :z_dim], X[:, z_dim:]])
        full = PCA().fit(singles)
        self.n_components_ = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.variance) + 1)
        self.pca_ = PCA(self.n_components_).fit(singles)
        self.explained_variance_ = float(self.pca_.explained_variance_ratio_.sum())
        return self

    def transform(self, X):
        z_dim = X.shape[1] // 2
        return np.concatenate([self.pca_.transform(X[:, :z_dim]), self.pca_.transform(X[:, z_dim:])], axis=1)


def probe_features(z_curr, z_prev):
    """Probe input: current and previous latent concatenated, [z_t, z_{t-1}]."""
    return np.concatenate([z_curr, z_prev], axis=-1)


def fit_state_probe(z, states, dones, episodes, variance=0.99):
    """
    Fit one linear (ridge) probe [z_t, z_{t-1}] -> (x, x_dot, theta, theta_dot)
    on the true latents of the given episodes. Both latents are projected on
    the principal subspace explaining `variance` of the latent variance
    (PairPCA), the features are standardized and the ridge strength is chosen
    by cross-validation, so the probe is equally fair to latent spaces of any
    scale or dimension. The number of components used is available as
    probe[0].n_components_.

    Frames right after a reset are skipped (their previous frame belongs to
    another sub-episode).

    Args:
        z: (N, T, z_dim) true latents of the test set
        states: (N, T, 4) true states
        dones: (N, T) reset flags
        episodes: episode indices used to fit the probe

    Returns:
        sklearn Pipeline with predict(features) -> (n, 4)
    """
    X, Y = [], []
    for ep in episodes:
        valid = ~dones[ep, :-1]               # frame t+1 follows frame t
        t = np.nonzero(valid)[0] + 1
        X.append(probe_features(z[ep, t], z[ep, t - 1]))
        Y.append(states[ep, t])
    probe = make_pipeline(PairPCA(variance), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return probe.fit(np.concatenate(X), np.concatenate(Y))


def predict_states(probe, z_curr, z_prev):
    """Apply a probe to arrays of latents with any leading shape."""
    lead = z_curr.shape[:-1]
    feats = probe_features(z_curr, z_prev).reshape(-1, 2 * z_curr.shape[-1])
    return probe.predict(feats).reshape(*lead, -1)


def r2_per_variable(pred, true):
    """Coefficient of determination per state variable (last axis)."""
    pred = pred.reshape(-1, pred.shape[-1])
    true = true.reshape(-1, true.shape[-1])
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot
