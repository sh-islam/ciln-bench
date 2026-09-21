"""Compute DivideMix small-loss diagnostic table for §4.5.5.

For each setting × variant, at final epoch, computes:
  - noise rate (from is_noisy)
  - GMM AUC of P(noisy) = 1 - clean_prob1 against ground-truth is_noisy
  - DM's "clean pile" purity = fraction of samples with clean_prob >= 0.5 that are truly clean
  - Random-baseline clean-pile purity (= 1 - noise_rate)

Aggregates across 3 seeds. Reports mean and std.

Uses only per-sample NPZ files already produced by training.
"""
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

ROOT = Path('/path/to/ciln-workspace/journal_edition/robustness_transfer/results_v4/dividemix')


ROWS = [
    # (label, variant, corruption, sev)
    ('frost(1)',           'noisy',     'frost',            1),
    ('contrast(5)',        'noisy',     'contrast',         5),
    ('gaussian(3)',        'noisy',     'gaussian_noise',   3),
    ('shot(5)',            'noisy',     'shot_noise',       5),
    ('pixelate(5)',        'noisy',     'pixelate',         5),
    ('glass-blur(3)',      'noisy',     'glass_blur',       3),
]


def condition(variant, corr, sev):
    return f'ours_v2_{variant}__{corr}_sev{sev}'


def compute_row(label, variant, corr, sev):
    cond = condition(variant, corr, sev)
    files = sorted(glob.glob(str(ROOT / f'{cond}__seed*__persample.npz')))
    if not files:
        print(f'  MISSING {cond}')
        return None
    aucs = []
    clean_purities = []
    noise_rates = []
    for f in files:
        d = np.load(f)
        cp = d['clean_prob1'][-1]  # final epoch
        is_noisy = d['is_noisy'].astype(int)
        noise_rates.append(is_noisy.mean())
        try:
            auc = roc_auc_score(is_noisy, 1 - cp)
            aucs.append(auc)
        except Exception:
            pass
        dm_clean = cp >= 0.5
        if dm_clean.sum():
            clean_purities.append((1 - is_noisy[dm_clean]).mean())
    return {
        'label': label,
        'noise_rate': (np.mean(noise_rates), np.std(noise_rates)),
        'auc': (np.mean(aucs), np.std(aucs)),
        'clean_pile_purity': (np.mean(clean_purities), np.std(clean_purities)),
        'random_baseline': 1 - np.mean(noise_rates),
    }


def main():
    print(f"{'Setting':16s} {'Noise':>7s} {'GMM AUC':>13s} {'Clean-pile purity':>20s} {'Random baseline':>18s}")
    print('-' * 82)
    for r in ROWS:
        row = compute_row(*r)
        if not row:
            continue
        nr = row['noise_rate'][0]
        auc = row['auc']
        cp = row['clean_pile_purity']
        rb = row['random_baseline']
        print(f"{row['label']:16s} {nr*100:>6.1f}%  {auc[0]:.3f}±{auc[1]:.02f}   {cp[0]*100:>7.1f}±{cp[1]*100:.1f}%   {rb*100:>16.1f}%")
    print()

    # Emit LaTeX row source
    print("--- LaTeX rows for Table 4.7 ---")
    print()
    for r in ROWS:
        row = compute_row(*r)
        if not row:
            continue
        nr = row['noise_rate'][0]
        auc = row['auc']
        cp = row['clean_pile_purity']
        rb = row['random_baseline']
        # Only print mean values in the table (std in caption)
        print(f"\\ctype{{{row['label'].split('(')[0]}}}({row['label'].split('(')[1]} & "
              f"{nr*100:.1f}\\% & "
              f"${auc[0]:.3f}$ & "
              f"${cp[0]*100:.1f}\\%$ & "
              f"${rb*100:.1f}\\%$ \\\\")


if __name__ == '__main__':
    main()
