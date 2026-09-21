"""Per-setting IDN strength metrics: FRV and VDV.

These are the two within-class variance metrics reported in the paper.

Definitions
-----------
Let ``p_i ∈ ℝ^K`` be the soft label distribution for image *i* (the bincount
over the voter pool's argmax predictions, divided by the number of voters),
and ``y_i`` its ground-truth class.

**FRV (Flip-Rate Variance)**: per-image "flip rate"
``η_i = 1 − p_i[y_i]`` followed by within-class variance, averaged across
classes::

    FRV = (1/K) Σ_y Var_{i: y_i = y}(η_i)

If two images of the same class get *the same* per-voter accuracy but with
the wrong votes pointing at different classes, FRV stays the same. Use VDV
to capture that case.

**VDV (Vote-Distribution Variance)**: within-class mean Frobenius distance
of ``p_i`` to the class-mean vote distribution
``p̄_y = E_{i: y_i = y}[p_i]``::

    VDV = E_i [ || p_i − p̄_{y_i} ||² ]

Equivalently the average within-class trace of the second-moment matrix of
``p_i − p̄_y``.

Identity guarantee
------------------
The numerical output of these functions matches the canonical
implementation used to produce ``v2_all_native.json`` in the released
benchmark, verified by ``tests/check_equivalence.py``.
"""
from __future__ import annotations
import numpy as np


def flip_rate_variance(p: np.ndarray, true_y: np.ndarray) -> float:
    """Per-class variance of η_i = 1 - p_i[y_i], averaged across classes.

    Parameters
    ----------
    p
        Shape ``(N, K)``. ``p[i]`` is the soft label distribution for image *i*
        (probabilities summing to 1).
    true_y
        Shape ``(N,)`` int array of ground-truth class ids in ``[0, K)``.

    Returns
    -------
    float
        Mean across classes (with at least two images) of the per-class
        variance of ``η``.
    """
    eta = 1.0 - p[np.arange(len(p)), true_y]
    n_classes = p.shape[1]
    class_vars = []
    for y in range(n_classes):
        mask = true_y == y
        if mask.sum() < 2:
            continue
        class_vars.append(float(np.var(eta[mask])))
    return float(np.mean(class_vars))


def vote_distribution_variance(p: np.ndarray, true_y: np.ndarray) -> float:
    """Average squared Frobenius distance of each ``p_i`` from its class mean.

    Parameters
    ----------
    p
        Shape ``(N, K)``. ``p[i]`` is the soft label distribution for image *i*.
    true_y
        Shape ``(N,)`` int array of ground-truth class ids in ``[0, K)``.

    Returns
    -------
    float
        ``E_i [ || p_i - p_bar_{y_i} ||^2 ]`` averaged across all images
        (classes are implicitly weighted by their sample count).
    """
    n_classes = p.shape[1]
    class_means = np.zeros((n_classes, n_classes), dtype=np.float64)
    for y in range(n_classes):
        mask = true_y == y
        if mask.any():
            class_means[y] = p[mask].mean(axis=0)
    diff = p - class_means[true_y]
    return float((diff * diff).sum(axis=1).mean())


# Short aliases (used in paper notation)
FRV = flip_rate_variance
VDV = vote_distribution_variance


def soft_label_from_voters(voter_argmax: np.ndarray, n_classes: int) -> np.ndarray:
    """Build the soft label ``p_i = bincount(votes_i) / M`` from per-voter argmax.

    Parameters
    ----------
    voter_argmax
        Shape ``(N, M)`` int array. ``voter_argmax[i, m]`` is voter *m*'s
        predicted class for image *i*.
    n_classes
        Number of classes ``K``.

    Returns
    -------
    p : np.ndarray
        Shape ``(N, K)`` float array of soft labels summing to 1 per row.
    """
    n, m = voter_argmax.shape
    p = np.zeros((n, n_classes), dtype=np.float64)
    for k in range(m):
        np.add.at(p, (np.arange(n), voter_argmax[:, k]), 1.0)
    p /= m
    return p


def overall_noise_rate(p: np.ndarray, true_y: np.ndarray) -> float:
    """δ = mean per-image flip rate. The headline noise rate reported per setting."""
    return float((1.0 - p[np.arange(len(p)), true_y]).mean())
