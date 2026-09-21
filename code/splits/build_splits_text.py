"""Build Gu-style splits for AG-News.

Starting from the 120K AG-News train split, deterministic random split:
  - CleanLabelTrain  (CLT, 50K)  -> trains voter pool
  - CleanLabelValid  (CLV, 10K)  -> voter val, NOT used downstream
  - NoisyLabelTrain  (NLT, 50K)  -> corrupted, labeled by voter pool
  - NoisyLabelValid  (NLV, 10K)  -> NLT companion (optional)

Note: paper used 22,500/27,000/13,566 NLT sizes for the other modalities; we
match the framework (per the release rule doc) which calls for ~50% of train
into NLT. With AG-News at 120K train, this gives 50K NLT after holding out
10K each for CLV/NLV.

Saves split indices as .npy files in ciln_text/data/splits/.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA  = ROOT / 'data' / 'ag_news'
SPLITS = ROOT / 'data' / 'splits'
SPLITS.mkdir(exist_ok=True)

SEED  = 0
TRAIN_TOTAL = 120_000

# Sizes (each entry must add to 120K)
CLT_N = 50_000
CLV_N = 10_000
NLT_N = 50_000
NLV_N = 10_000


def main():
    print(f'[splits] loading ag_news...', flush=True)
    ds = load_from_disk(str(DATA))
    n = len(ds['train'])
    assert n == TRAIN_TOTAL, f'expected {TRAIN_TOTAL} train rows, got {n}'

    print(f'[splits] deterministic shuffle (seed={SEED})...', flush=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)

    clt_idx = perm[:CLT_N]
    clv_idx = perm[CLT_N : CLT_N + CLV_N]
    nlt_idx = perm[CLT_N + CLV_N : CLT_N + CLV_N + NLT_N]
    nlv_idx = perm[CLT_N + CLV_N + NLT_N : CLT_N + CLV_N + NLT_N + NLV_N]
    assert len(clt_idx) == CLT_N
    assert len(nlt_idx) == NLT_N
    # No overlap check
    s_clt = set(clt_idx.tolist())
    s_nlt = set(nlt_idx.tolist())
    assert len(s_clt & s_nlt) == 0, 'CLT/NLT overlap'

    # Save indices (one .npy each)
    np.save(SPLITS / 'CLT_indices.npy', clt_idx.astype(np.int64))
    np.save(SPLITS / 'CLV_indices.npy', clv_idx.astype(np.int64))
    np.save(SPLITS / 'NLT_indices.npy', nlt_idx.astype(np.int64))
    np.save(SPLITS / 'NLV_indices.npy', nlv_idx.astype(np.int64))

    # Save the manifest
    summary = {
        'dataset': 'ag_news',
        'seed': SEED,
        'train_total': TRAIN_TOTAL,
        'splits': {
            'CLT': {'size': CLT_N, 'role': 'voter training'},
            'CLV': {'size': CLV_N, 'role': 'voter validation'},
            'NLT': {'size': NLT_N, 'role': 'corrupted + voter-labeled'},
            'NLV': {'size': NLV_N, 'role': 'NLT companion'},
        },
        'test_split': 'official AG-News test, 7600 rows (held out separately)',
    }
    (SPLITS / 'splits_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[splits] CLT={CLT_N}, CLV={CLV_N}, NLT={NLT_N}, NLV={NLV_N}')
    print(f'[splits] saved indices to {SPLITS}/')


if __name__ == '__main__':
    main()
