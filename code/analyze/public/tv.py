"""TV distance between voter-pool soft labels and a reference noise distribution.

For CIFAR-10 settings we compare the benchmark's per-image soft labels against
three references at the same overall noise rate ``δ``:

* **CIFAR-10H upsampled** — Northcutt-style upsampling of the human-label
  counts so the overall error rate matches ``δ``.
* **Symmetric flipping** — independent per-voter symmetric noise at rate ``δ``.
* **CCN (Class-Conditional Noise)** — sample voters from a per-class transition
  matrix ``T[y] = E_{i: y_i = y}[p_i]`` estimated from the benchmark itself.

Per setting we report ``E_i [ 0.5 · ||p_i − ref_i||_1 ]`` (mean per-image total
variation), bootstrapped over ``N_BOOT`` reference resamples.

Identity guarantee
------------------
The numerical output of these functions matches the canonical implementation
used to produce ``tv_vs_cifar10h_v2.json`` in the released benchmark, verified
by ``tests/check_equivalence.py``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


N_CLASSES_CIFAR10 = 10
DEFAULT_N_VOTERS = 4
DEFAULT_N_BOOT = 50


# ---------------------------------------------------------------------------
# Reference distributions
# ---------------------------------------------------------------------------

def upsample_human_counts(
    counts: np.ndarray, true_y: np.ndarray, target_rate: float,
) -> np.ndarray:
    """Scale wrong-label counts so the overall error rate matches ``target_rate``.

    Follows Northcutt et al. 2021: the correct-label count per image is kept,
    and every wrong-label count is multiplied by a single global factor ``k``
    chosen so that ``sum(wrong*k) / sum(correct + wrong*k) == target_rate``.

    Parameters
    ----------
    counts
        Shape ``(N, K)`` integer array of raw human-label votes per image.
    true_y
        Shape ``(N,)`` ground-truth class ids.
    target_rate
        Desired overall error rate ``δ`` (0–1).

    Returns
    -------
    soft : np.ndarray
        Shape ``(N, K)`` row-normalised distribution. If the natural human
        error rate already exceeds ``target_rate`` the function returns the
        un-upsampled distribution.
    """
    n_classes = counts.shape[1]
    correct = counts[np.arange(len(true_y)), true_y]
    wrong_total = counts.sum(axis=1) - correct

    p_current = counts.astype(np.float64) / counts.sum(axis=1, keepdims=True)
    current_rate = float((1.0 - p_current[np.arange(len(true_y)), true_y]).mean())

    if target_rate <= current_rate or wrong_total.sum() == 0:
        return p_current

    sum_correct = float(correct.sum())
    sum_wrong = float(wrong_total.sum())
    k = target_rate * sum_correct / ((1.0 - target_rate) * sum_wrong)
    up = counts.astype(np.float64).copy()
    for i in range(len(true_y)):
        ty = true_y[i]
        mask = np.arange(n_classes) != ty
        up[i, mask] *= k
    return up / np.maximum(up.sum(axis=1, keepdims=True), 1e-12)


def sample_voters_from_soft(
    soft: np.ndarray, n_voters: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n_voters`` argmax labels per image from ``soft`` (N, K), return
    the resulting bincount-divided-by-``n_voters`` soft label of shape (N, K).
    """
    n, k = soft.shape
    out = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        draws = rng.choice(k, size=n_voters, p=soft[i])
        for d in draws:
            out[i, d] += 1
    out /= n_voters
    return out


def symmetric_flip_distribution(
    true_y: np.ndarray, rate: float, n_voters: int, n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent symmetric flipping baseline. For each voter and image,
    with probability ``rate`` pick a uniformly random *other* class, else keep
    the true class. Returns the per-image vote distribution (N, K).
    """
    n = len(true_y)
    out = np.zeros((n, n_classes), dtype=np.float64)
    for i in range(n):
        ty = true_y[i]
        for _ in range(n_voters):
            if rng.random() < rate:
                other = rng.integers(0, n_classes - 1)
                if other >= ty:
                    other += 1
                out[i, other] += 1
            else:
                out[i, ty] += 1
    out /= n_voters
    return out


def ccn_distribution(
    p_ours: np.ndarray, true_y: np.ndarray, n_voters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """CCN baseline: estimate per-class transition matrix ``T[y]`` as the
    within-class mean of ``p_ours``, then sample ``n_voters`` labels per image
    from ``T[true_y(i)]``.
    """
    n, k = p_ours.shape
    T = np.zeros((k, k), dtype=np.float64)
    for y in range(k):
        m = true_y == y
        if m.any():
            T[y] = p_ours[m].mean(axis=0)
    out = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        draws = rng.choice(k, size=n_voters, p=T[true_y[i]])
        for d in draws:
            out[i, d] += 1
    out /= n_voters
    return out


# ---------------------------------------------------------------------------
# TV metric + bootstrap loop
# ---------------------------------------------------------------------------

def mean_tv(p: np.ndarray, q: np.ndarray) -> float:
    """Mean per-image total variation distance ``E_i [ 0.5 · ||p_i - q_i||_1 ]``."""
    return float(0.5 * np.abs(p - q).sum(axis=1).mean())


@dataclass
class TVResult:
    noise_rate: float
    tv_ours_mean: float
    tv_ours_ci95: float
    tv_symmetric_mean: float
    tv_symmetric_ci95: float
    tv_ccn_mean: float
    tv_ccn_ci95: float


def tv_vs_human(
    p_ours: np.ndarray,
    true_y: np.ndarray,
    human_counts: np.ndarray,
    n_voters: int = DEFAULT_N_VOTERS,
    n_boot: int = DEFAULT_N_BOOT,
    seed_offset: int = 0,
) -> TVResult:
    """Bootstrap-average per-image TV of ``p_ours`` against three references at
    the same overall noise rate.

    Parameters
    ----------
    p_ours
        Shape ``(N, K)`` benchmark soft labels.
    true_y
        Shape ``(N,)`` ground-truth class ids.
    human_counts
        Shape ``(N, K)`` integer count of human votes per image (e.g. CIFAR-10H).
    n_voters
        Number of voters drawn per image when sampling references.
    n_boot
        Number of bootstrap iterations.
    seed_offset
        Additive offset applied to every per-iteration seed. Set to 0 to match
        the canonical pipeline exactly.
    """
    delta = float((1.0 - p_ours[np.arange(len(p_ours)), true_y]).mean())
    n_classes = p_ours.shape[1]

    tv_ours, tv_sym, tv_ccn = [], [], []
    for b in range(n_boot):
        ref_soft = upsample_human_counts(human_counts, true_y, delta)
        ref = sample_voters_from_soft(
            ref_soft, n_voters, np.random.default_rng(b + seed_offset)
        )
        sym = symmetric_flip_distribution(
            true_y, delta, n_voters, n_classes,
            np.random.default_rng(b + 1000 + seed_offset),
        )
        ccn = ccn_distribution(
            p_ours, true_y, n_voters,
            np.random.default_rng(b + 2000 + seed_offset),
        )
        tv_ours.append(mean_tv(p_ours, ref))
        tv_sym.append(mean_tv(sym, ref))
        tv_ccn.append(mean_tv(ccn, ref))

    return TVResult(
        noise_rate=delta,
        tv_ours_mean=float(np.mean(tv_ours)),
        tv_ours_ci95=float(1.96 * np.std(tv_ours)),
        tv_symmetric_mean=float(np.mean(tv_sym)),
        tv_symmetric_ci95=float(1.96 * np.std(tv_sym)),
        tv_ccn_mean=float(np.mean(tv_ccn)),
        tv_ccn_ci95=float(1.96 * np.std(tv_ccn)),
    )
