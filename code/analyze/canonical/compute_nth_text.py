"""Compute headline metrics (delta, VDV) per setting from the saved softmaxes.

Uses the same VDV definition as the image/tabular pipeline.
Writes one JSON: ciln_text/results.json with per-setting numbers.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SETTINGS = ROOT / 'settings'
RESULTS  = ROOT / 'results.json'

# Reuse the public metrics implementation we shipped on GitHub.
sys.path.insert(0, str(ROOT.parent / 'ciln-bench-draft' / 'code' / 'analyze'))
from public.metrics import (
    vote_distribution_variance, soft_label_from_voters, overall_noise_rate,
)

VOTERS = ('fasttext', 'distilbert', 'roberta', 'sbert')
N_CLASSES = 4


def main():
    out_rows = []
    for sd in sorted(SETTINGS.iterdir()):
        if not sd.is_dir(): continue
        # Load voter softmaxes + true labels
        argmaxes = []
        for v in VOTERS:
            sm = np.load(sd / f'softmax_{v}.npy')
            argmaxes.append(sm.argmax(axis=1))
        argmax = np.stack(argmaxes, axis=1)  # (N, M)
        true_y = np.load(sd / 'labels.npy')

        p = soft_label_from_voters(argmax, n_classes=N_CLASSES)
        delta = overall_noise_rate(p, true_y)
        vdv   = vote_distribution_variance(p, true_y)

        # Per-voter accuracies
        per_voter_acc = {
            v: float((np.load(sd / f'softmax_{v}.npy').argmax(axis=1) == true_y).mean())
            for v in VOTERS
        }

        row = {
            'name': sd.name,
            'n_images': int(len(p)),
            'noise_rate': float(delta),
            'vdv': float(vdv),
            'per_voter_acc': per_voter_acc,
        }
        out_rows.append(row)
        print(f"  {sd.name:<32s}  delta={delta*100:5.1f}%  VDV={vdv:.4f}  "
              f"acc(ft/db/rb/sb)={per_voter_acc['fasttext']:.3f}/"
              f"{per_voter_acc['distilbert']:.3f}/{per_voter_acc['roberta']:.3f}/"
              f"{per_voter_acc['sbert']:.3f}", flush=True)

    RESULTS.write_text(json.dumps({'settings': out_rows}, indent=2))
    print(f"\nSaved {len(out_rows)} settings to {RESULTS}")


if __name__ == '__main__':
    main()
