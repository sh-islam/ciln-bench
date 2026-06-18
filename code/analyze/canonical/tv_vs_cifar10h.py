"""Replicate Gu's §2.3 evaluation, on our 45 settings.

For each of our 45 settings (test set, 4 v2 voters):
  - Compute ours' noise rate δ on this setting.
  - Up-sample CIFAR-10H to match δ (Northcutt-style: resample wrong labels
    with replacement). Sample 4 labels per image.
  - Generate symmetric flipping baseline at δ. Sample 4 labels per image.
  - Generate CCN baseline at δ using setting's empirical class-confusion T.
    Sample 4 labels per image.
  - For each of {ours, symmetric, ccn} vs (upsampled CIFAR-10H), compute
    mean total variation distance (per-image TV, averaged).
  - Bootstrap over N_SEEDS upsampling resamples.

Outputs: results/tv_vs_cifar10h_v2.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

JR = Path('/home/misla2/thesis/python/mnist_corruptions/journal_edition')
TEST_ROOT = JR / 'output_seed0_TEST' / 'cifar10'
OUT = JR / 'idn_2x2' / 'results'
CIFAR10H = JR.parent / 'cifar10h' / 'cifar10h-counts.npy'

V2_VOTERS = ['resnet20', 'wrn28_10', 'deit3_small', 'clip']
N_VOTERS = len(V2_VOTERS)
N_CLASSES = 10
N_BOOT = 50
RNG = np.random.default_rng(0)


def soft_distribution_from_voters(setting_dir):
    """Return per-image soft label distribution (10000, 10).
    Built from 4 voter argmax labels: bincount/N."""
    votes = []
    for v in V2_VOTERS:
        sm = np.load(setting_dir / f'softmax_{v}.npy')
        votes.append(sm.argmax(axis=1))
    votes = np.stack(votes, axis=1)  # (10000, 4)
    n_img = len(votes)
    p = np.zeros((n_img, N_CLASSES), dtype=np.float64)
    for k in range(N_VOTERS):
        np.add.at(p, (np.arange(n_img), votes[:, k]), 1.0)
    p /= N_VOTERS
    return p, votes


def cifar10h_upsampled(target_rate, true_y, seed):
    """Upsample CIFAR-10H to target overall error rate.

    Per Northcutt 2021: resample wrong labels with replacement until overall
    error rate matches target. Sample N=4 noisy labels per image from the
    upsampled per-image distribution.

    Returns soft label distribution (10000, 10).
    """
    rng = np.random.default_rng(seed)
    counts = np.load(CIFAR10H)  # (10000, 10), integer counts ~50 per row
    # Per image: split correct vs wrong votes
    correct = np.array([counts[i, true_y[i]] for i in range(len(true_y))])
    wrong_total = counts.sum(axis=1) - correct  # per-image wrong count

    # Build soft distribution per image (current cifar10h)
    p_current = counts.astype(np.float64) / counts.sum(axis=1, keepdims=True)
    current_rate = (1 - p_current[np.arange(len(true_y)), true_y]).mean()

    if target_rate <= current_rate:
        # Just sample N=4 labels per image from current distribution
        soft = p_current.copy()
    else:
        # Need to upsample wrong labels.
        # Northcutt approach: scale up the wrong-label counts uniformly so
        # the overall error rate matches target. Specifically, multiply each
        # image's wrong-label counts by factor k such that overall error rate
        # = target_rate.
        # mean over images: factor*wrong[i] / (correct[i] + factor*wrong[i]) ≈ target
        # We solve numerically since denominator depends on factor.
        # Simpler: fix the per-image weighting so overall (sum wrong)/(sum total) = target.
        # New total wrong = target * (correct + factor*wrong)
        # Let k be the global scale: correct stays, wrong*k. Then
        #   target = sum(wrong*k) / sum(correct + wrong*k)
        # Solve: k = target*sum(correct) / ((1-target)*sum(wrong))
        sum_correct = correct.sum()
        sum_wrong = wrong_total.sum()
        if sum_wrong == 0:
            soft = p_current
        else:
            k = target_rate * sum_correct / ((1 - target_rate) * sum_wrong)
            # Build upsampled counts: correct unchanged, wrong rows scaled by k
            up = counts.astype(np.float64).copy()
            for i in range(len(true_y)):
                ty = true_y[i]
                wmask = np.arange(N_CLASSES) != ty
                up[i, wmask] *= k
            row_sum = up.sum(axis=1, keepdims=True)
            soft = up / np.maximum(row_sum, 1e-12)

    # Sample N=4 labels per image from soft distribution
    sampled = np.zeros((len(true_y), N_CLASSES), dtype=np.float64)
    for i in range(len(true_y)):
        draws = rng.choice(N_CLASSES, size=N_VOTERS, p=soft[i])
        for d in draws:
            sampled[i, d] += 1
    sampled /= N_VOTERS
    return sampled


def symmetric_baseline(true_y, rate, seed):
    """Independent symmetric flipping at rate δ. Sample N=4 labels per image."""
    rng = np.random.default_rng(seed)
    n = len(true_y)
    sampled = np.zeros((n, N_CLASSES), dtype=np.float64)
    for i in range(n):
        ty = true_y[i]
        # P(flip) = rate; if flip, uniformly random other class
        for _ in range(N_VOTERS):
            if rng.random() < rate:
                # random other class
                other = rng.integers(0, N_CLASSES - 1)
                if other >= ty: other += 1
                sampled[i, other] += 1
            else:
                sampled[i, ty] += 1
    sampled /= N_VOTERS
    return sampled


def ccn_baseline(true_y, ours_p, seed):
    """CCN baseline: per-class transition matrix from ours' class confusion.

    Compute T[y, y'] from ours' per-image votes (averaged within true class).
    Sample N=4 noisy labels per image from T[true_y(x)].
    """
    rng = np.random.default_rng(seed)
    T = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for y in range(N_CLASSES):
        m = true_y == y
        if m.any():
            T[y] = ours_p[m].mean(axis=0)
    n = len(true_y)
    sampled = np.zeros((n, N_CLASSES), dtype=np.float64)
    for i in range(n):
        draws = rng.choice(N_CLASSES, size=N_VOTERS, p=T[true_y[i]])
        for d in draws:
            sampled[i, d] += 1
    sampled /= N_VOTERS
    return sampled


def tv_mean(p, q):
    """Mean per-image total variation distance."""
    return float(0.5 * np.abs(p - q).sum(axis=1).mean())


def run_setting(setting_name, setting_dir):
    p_ours, votes_ours = soft_distribution_from_voters(setting_dir)
    true_y = np.load(setting_dir / 'labels.npy')

    # ours overall noise rate
    eta_ours = (1.0 - p_ours[np.arange(len(p_ours)), true_y]).mean()

    # Bootstrap
    tv_ours_list, tv_sym_list, tv_ccn_list = [], [], []
    for b in range(N_BOOT):
        # Up-sample CIFAR-10H at our noise rate
        ref = cifar10h_upsampled(eta_ours, true_y, seed=b)
        # Baselines at same rate
        sym = symmetric_baseline(true_y, eta_ours, seed=b + 1000)
        ccn = ccn_baseline(true_y, p_ours, seed=b + 2000)
        tv_ours_list.append(tv_mean(p_ours, ref))
        tv_sym_list.append(tv_mean(sym, ref))
        tv_ccn_list.append(tv_mean(ccn, ref))

    return {
        'name': setting_name,
        'noise_rate_ours': float(eta_ours),
        'tv_ours': {'mean': float(np.mean(tv_ours_list)), 'ci95': float(1.96*np.std(tv_ours_list))},
        'tv_symmetric': {'mean': float(np.mean(tv_sym_list)), 'ci95': float(1.96*np.std(tv_sym_list))},
        'tv_ccn':       {'mean': float(np.mean(tv_ccn_list)), 'ci95': float(1.96*np.std(tv_ccn_list))},
    }


def main():
    print(f'[{time.strftime("%X")}] TV vs CIFAR-10H starting...')
    results = []
    settings = []
    for c_dir in sorted(TEST_ROOT.iterdir()):
        if not c_dir.is_dir(): continue
        for s_dir in sorted(c_dir.iterdir()):
            if not s_dir.is_dir(): continue
            v2_dir = s_dir / 'v2'
            if not (v2_dir / 'labels.npy').exists(): continue
            name = f"{c_dir.name}_sev{s_dir.name.split('_')[1]}"
            settings.append((name, v2_dir))
    print(f'  {len(settings)} settings')

    for i, (name, sd) in enumerate(settings):
        t0 = time.time()
        r = run_setting(name, sd)
        results.append(r)
        print(f'  [{i+1}/{len(settings)}] {name}: noise={r["noise_rate_ours"]*100:.1f}%  '
              f'ours={r["tv_ours"]["mean"]:.4f}±{r["tv_ours"]["ci95"]:.4f}  '
              f'sym={r["tv_symmetric"]["mean"]:.4f}  '
              f'ccn={r["tv_ccn"]["mean"]:.4f}  ({time.time()-t0:.0f}s)')

    (OUT / 'tv_vs_cifar10h_v2.json').write_text(json.dumps(results, indent=2))
    print(f'[{time.strftime("%X")}] saved tv_vs_cifar10h_v2.json')


if __name__ == '__main__':
    main()
