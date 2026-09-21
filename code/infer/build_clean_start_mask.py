"""Build the clean-start (CILN-C) mask for one dataset.

An example is retained when every voter classifies the CLEAN (uncorrupted)
noisy-label-train input correctly (paper Sec. 3.3, Algorithm 1). The mask is
the same for every corruption setting of a dataset, so it is released once per
dataset as clean_start/clean_correct_mask.npy next to the clean-NLT softmaxes.

Usage:
  python build_clean_start_mask.py --clean-dir <clean_start dir with softmax_<voter>.npy + labels.npy> \
                                   --voters resnet20 wrn28_10 deit3_small clip
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean-dir', required=True)
    ap.add_argument('--voters', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = Path(a.clean_dir)
    y = np.load(d / 'labels.npy')
    correct = np.stack([np.load(d / f'softmax_{v}.npy').argmax(1) == y for v in a.voters], axis=1)
    mask = correct.all(axis=1)
    out = Path(a.out) if a.out else d / 'clean_correct_mask.npy'
    np.save(out, mask)
    summary = {'n': int(len(mask)), 'n_kept': int(mask.sum()), 'frac_kept': float(mask.mean()),
               'per_voter_clean_acc': {v: float(correct[:, i].mean()) for i, v in enumerate(a.voters)}}
    (out.parent / 'clean_correct_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
