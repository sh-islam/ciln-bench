"""Compute attractor class + purity per H-tier setting for §4.5.5 Table 4.8.

For each setting, uses the four voter softmaxes at
  output_seed0/cifar10/<corr>/severity_<sev>/noisy_label_train/softmax_<voter>.npy
to derive a per-image majority-vote noisy label, then computes:

  attractor        = argmax(noisy_freq - clean_freq)
                     Clean freq is uniform 10% on CIFAR-10 balanced test.
                     But we compute from ground-truth labels for accuracy.
  attractor_delta  = 100 * (noisy_freq[attractor] - clean_freq[attractor])   in pp
  purity           = fraction of images with majority-vote label == attractor
                     whose ground-truth label == attractor

Cross-checks against TMLR's Table 3 numbers for four settings, and adds
shot_noise(5) as new row.
"""
import numpy as np
from pathlib import Path

JR = Path('/path/to/ciln-workspace/journal_edition')
NLT_ROOT = JR / 'output_seed0' / 'cifar10'

CIFAR_VOTERS = ['resnet20', 'wrn28_10', 'deit3_small', 'clip']
CLASSES = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck']

SETTINGS = [
    ('gaussian_noise', 3),
    ('contrast',        5),
    ('shot_noise',      5),  # NEW row
    ('pixelate',        5),
    ('glass_blur',      3),
]

# TMLR-published values for the four settings TMLR reports
TMLR_REFERENCE = {
    ('gaussian_noise', 3): {'rate': 49.2, 'attractor': 'frog', 'delta': 22, 'purity': 24.1},
    ('contrast',       5): {'rate': 47.3, 'attractor': 'cat',  'delta': 19, 'purity': 26.2},
    ('pixelate',       5): {'rate': 58.2, 'attractor': 'truck','delta': 20, 'purity': 27.3},
    ('glass_blur',     3): {'rate': 64.7, 'attractor': 'frog', 'delta': 30, 'purity': 19.9},
}


def compute_row(corr, sev):
    """Return dict with noise_rate, attractor, attractor_delta_pp, purity, n_images."""
    nl_dir = NLT_ROOT / corr / f'severity_{sev}' / 'noisy_label_train'
    true_y = np.load(nl_dir / 'labels.npy')
    n = len(true_y)

    # Per-image argmax per voter (four voters)
    votes = np.stack([
        np.load(nl_dir / f'softmax_{v}.npy').argmax(axis=1)
        for v in CIFAR_VOTERS
    ], axis=1)  # (n, 4)

    # Majority label per image (used for attractor and purity)
    noisy_label = np.array([
        np.bincount(votes[i], minlength=10).argmax()
        for i in range(n)
    ])

    # Noise rate follows the PL-IDN protocol: sample one voter per image
    # uniformly at random. Expected value = mean of per-voter noise rates.
    per_voter_nr = [(votes[:, i] != true_y).mean() for i in range(len(CIFAR_VOTERS))]
    noise_rate = float(np.mean(per_voter_nr)) * 100.0

    # Class frequencies
    clean_freq = np.bincount(true_y, minlength=10) / n * 100.0
    noisy_freq = np.bincount(noisy_label, minlength=10) / n * 100.0
    delta_freq = noisy_freq - clean_freq

    # Attractor = argmax of the frequency increase
    attractor_class = int(delta_freq.argmax())
    attractor_name = CLASSES[attractor_class]
    attractor_delta = float(delta_freq[attractor_class])

    # Purity = of the images DM would call the attractor class,
    # what fraction are actually the attractor class?
    attractor_labeled = noisy_label == attractor_class
    if attractor_labeled.sum() > 0:
        purity = (true_y[attractor_labeled] == attractor_class).mean() * 100.0
    else:
        purity = float('nan')

    return {
        'noise_rate': noise_rate,
        'attractor': attractor_name,
        'attractor_delta_pp': attractor_delta,
        'purity': purity,
        'n_images': n,
    }


def main():
    print(f"{'Setting':22s}  {'Rate':>7s}  {'Attractor':>12s}  {'Delta pp':>10s}  {'Purity':>8s}  Notes")
    print('-' * 100)
    for corr, sev in SETTINGS:
        row = compute_row(corr, sev)
        label = f"{corr}({sev})"
        ref = TMLR_REFERENCE.get((corr, sev), None)
        note = ''
        if ref:
            note = f"TMLR: {ref['attractor']} +{ref['delta']}pp purity {ref['purity']}%"
        else:
            note = 'NEW row (not in TMLR)'
        print(f"{label:22s}  {row['noise_rate']:6.1f}%  {row['attractor']:>12s}  "
              f"+{row['attractor_delta_pp']:6.1f}pp  {row['purity']:>7.1f}%  {note}")

    print()
    print("--- LaTeX table rows (attractor: name (+delta pp)) ---")
    print()
    for corr, sev in SETTINGS:
        row = compute_row(corr, sev)
        label = corr.replace('_', ' ')
        print(f"\\c{corr.replace('_', '').title()}{{}}({sev}) & "
              f"${row['noise_rate']:.1f}\\%$ & "
              f"{row['attractor']} (+{row['attractor_delta_pp']:.0f} pp) & "
              f"${row['purity']:.1f}\\%$ \\\\")


if __name__ == '__main__':
    main()
