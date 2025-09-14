from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def add_label_noise(dataset: tuple[np.ndarray, np.ndarray], prob: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomly flip labels in a dataset with a given probability.

    Parameters
    ----------
    dataset:
        Tuple ``(X, y)`` containing features and integer class labels.
    prob:
        Probability of replacing any given label with a randomly sampled
        different class label.
    seed:
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Dataset with potentially corrupted labels.
    """
    rng = np.random.default_rng(seed)
    X, y = dataset
    y_noisy = np.array(y, copy=True)
    classes = np.unique(y_noisy)
    mask = rng.random(size=y_noisy.shape) < prob
    for idx in np.where(mask)[0]:
        current = y_noisy[idx]
        choices = classes[classes != current]
        y_noisy[idx] = rng.choice(choices)
    return X, y_noisy


def perturb_priors(dataset: tuple[np.ndarray, np.ndarray], delta: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a dataset to perturb its class prior probabilities.

    Parameters
    ----------
    dataset:
        Tuple ``(X, y)`` containing features and integer class labels.
    delta:
        Array of additive perturbations for each class prior. The length of
        ``delta`` must match the number of unique classes in ``y`` and the
        resulting prior probabilities must remain non-negative.
    seed:
        Seed for the random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Resampled dataset with modified class priors.
    """
    rng = np.random.default_rng(seed)
    X, y = dataset
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != classes.shape:
        raise ValueError("delta must have the same length as the number of classes")
    priors = counts / len(y)
    new_priors = priors + delta
    if np.any(new_priors < 0):
        raise ValueError("Perturbation leads to negative class prior")
    new_priors = new_priors / new_priors.sum()
    weights = np.zeros(len(y), dtype=float)
    for idx, cls in enumerate(classes):
        weights[y == cls] = new_priors[idx] / counts[idx]
    indices = rng.choice(len(y), size=len(y), replace=True, p=weights)
    return X[indices], y[indices]


def sliced_w1(X: np.ndarray, Y: np.ndarray, k: int = 64, seed: int | None = None) -> float:
    """Approximate the Wasserstein-1 distance via random projections.

    Parameters
    ----------
    X, Y:
        Arrays of shape ``(n_samples, n_features)`` representing two empirical
        distributions.
    k:
        Number of random projections used in the approximation.
    seed:
        Optional seed controlling the random projections.

    Returns
    -------
    float
        Estimated sliced Wasserstein-1 distance between the empirical
        distributions represented by ``X`` and ``Y``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Input arrays must have the same number of features")
    d = X.shape[1]
    projections = rng.normal(size=(k, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    dist = 0.0
    for theta in projections:
        proj_X = X @ theta
        proj_Y = Y @ theta
        dist += wasserstein_distance(proj_X, proj_Y)
    return dist / k
