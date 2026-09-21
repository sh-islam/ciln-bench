"""Robustness check for the TV-vs-CIFAR-10H protocol (paper Sec. 5.2).

The published TV numbers upsample the CIFAR-10H reference to each setting's
noise rate and then subsample it to 4 votes per image so both sides of the TV
share the same quantisation (see tv_vs_cifar10h.py). This script recomputes
the CILN-S TV for all 45 CIFAR-10 settings under two alternative references
and checks that the published orderings survive:

  tv_full_upsampled : upsample to the setting's noise rate, no 4-vote subsample
  tv_full_raw       : raw ~50-vote CIFAR-10H distribution, no upsampling

Checks: (1) matched-tier comparison against the published PL-IDN TV values
(0.180 / 0.301 / 0.742, Gu et al. 2022 Table 1), (2) the 10 lowest-TV settings,
(3) Spearman rank correlation with the published TV over the 45 settings.

Inputs: the released test-split voter softmaxes (settings/<s>/test/) and
cifar10h-counts.npy (Peterson et al. 2019). Output: a CSV plus a text report.
Reproduces results/cifar10/tv_full_reference_check.{csv,_report.txt}.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

VOTERS = ['resnet20', 'wrn28_10', 'deit3_small', 'clip']
N_CLASSES = 10
PLIDN_PUB_TV = {'low': 0.180, 'medium': 0.301, 'high': 0.742}
TIER_MATCH = {
    'low':    ['frost_sev1', 'pixelate_sev1'],
    'medium': ['elastic_transform_sev3', 'snow_sev3', 'gaussian_noise_sev1'],
    'high':   ['contrast_sev5', 'gaussian_noise_sev3', 'shot_noise_sev5', 'pixelate_sev5', 'glass_blur_sev3'],
}


def vote_share(setting_dir: Path):
    votes = np.stack([np.load(setting_dir / f'softmax_{v}.npy').argmax(1) for v in VOTERS], axis=1)
    p = np.zeros((len(votes), N_CLASSES))
    for k in range(votes.shape[1]):
        np.add.at(p, (np.arange(len(votes)), votes[:, k]), 1.0)
    return p / votes.shape[1], np.load(setting_dir / 'labels.npy')


def cifar10h_raw(counts):
    return counts / counts.sum(1, keepdims=True)


def cifar10h_upsampled(counts, target_rate, true_y):
    """Northcutt-style upsampling of wrong labels to a target error rate, no subsampling."""
    correct = counts[np.arange(len(true_y)), true_y]
    wrong = counts.sum(1) - correct
    p = counts / counts.sum(1, keepdims=True)
    current = float((1 - p[np.arange(len(true_y)), true_y]).mean())
    if target_rate <= current or wrong.sum() == 0:
        return p
    k = target_rate * correct.sum() / ((1 - target_rate) * wrong.sum())
    up = counts.copy()
    mask = np.ones_like(up, dtype=bool); mask[np.arange(len(true_y)), true_y] = False
    up[mask] *= k
    return up / up.sum(1, keepdims=True)


def tv_mean(p, q):
    return float(0.5 * np.abs(p - q).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='HF release folder: settings/<setting>/test/')
    ap.add_argument('--cifar10h', required=True, help='cifar10h-counts.npy')
    ap.add_argument('--published', required=True, help='results/cifar10/tv_ciln_s.json')
    ap.add_argument('--out-csv', default='tv_full_reference_check.csv')
    ap.add_argument('--out-report', default='tv_full_reference_check_report.txt')
    a = ap.parse_args()

    counts = np.load(a.cifar10h).astype(np.float64)
    pub = {r['name']: r['tv_ours']['mean'] for r in json.load(open(a.published))}
    rows = []
    for sdir in sorted(Path(a.data_root).iterdir()):
        test = sdir / 'test'
        if not (test / 'labels.npy').exists():
            continue
        p, y = vote_share(test)
        eta = float((1 - p[np.arange(len(p)), y]).mean())
        rows.append({'setting': sdir.name, 'noise_rate': eta, 'tv_published': pub.get(sdir.name),
                     'tv_full_upsampled': tv_mean(p, cifar10h_upsampled(counts, eta, y)),
                     'tv_full_raw': tv_mean(p, cifar10h_raw(counts))})
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    rep = []
    byname = {r['setting']: r for r in rows}
    for variant in ('tv_published', 'tv_full_upsampled', 'tv_full_raw'):
        rep.append(f'\n--- {variant} ---')
        for tier, matched in TIER_MATCH.items():
            for m in matched:
                v = byname[m][variant]
                rep.append(f'  {tier:6s} {m:24s} TV={v:.4f} vs PL-IDN-{tier} {PLIDN_PUB_TV[tier]:.3f} '
                           f'[{"below" if v < PLIDN_PUB_TV[tier] else "ABOVE"}]')
        top10 = sorted(rows, key=lambda r: r[variant])[:10]
        pub10 = {n for n, _ in sorted(pub.items(), key=lambda kv: kv[1])[:10]}
        rep.append(f'  top-10: {[r["setting"] for r in top10]}')
        rep.append(f'  overlap with published top-10: {len({r["setting"] for r in top10} & pub10)}/10')
        rep.append(f'  min PL-IDN TV {min(PLIDN_PUB_TV.values()):.3f} > max top-10 TV {max(r[variant] for r in top10):.4f}: '
                   f'{min(PLIDN_PUB_TV.values()) > max(r[variant] for r in top10)}')
    names = sorted(byname)
    for variant in ('tv_full_upsampled', 'tv_full_raw'):
        rho, _ = spearmanr([byname[n]['tv_published'] for n in names], [byname[n][variant] for n in names])
        rep.append(f'Spearman(tv_published, {variant}) over {len(names)} settings = {rho:.4f}')
    Path(a.out_report).write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
