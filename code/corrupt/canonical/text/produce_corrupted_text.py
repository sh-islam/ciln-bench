"""Apply all 4 corruptions x 3 severities to AG-News test set, save each setting
to ciln_text/settings/<corruption>_sev<n>/ with per-row params + texts.npy.
"""
from __future__ import annotations
import hashlib, json, struct, sys, time
from pathlib import Path

import numpy as np
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'corruptions'))
from text_funcs import CORRUPTIONS, load_wordnet_synonyms

# Restrict to the two release-quality corruption families.
# Other families were investigated but dropped (see paper draft):
#   * Word-level (synonym, swap, deletion, shuffle, keyword_removal): no IDN
#     signal on AG-News (content-robust topic classification).
#   * Surface-form (random_casing, diacritic_substitution): voter-pool blindness
#     (fastText and sentence-transformer lowercase internally).
RELEASE_CORRUPTIONS = {
    'butter_fingers':   CORRUPTIONS['butter_fingers'],
    'front_truncation': CORRUPTIONS['front_truncation'],
}

DATA_DIR    = ROOT / 'data' / 'ag_news'
SPLITS_DIR  = ROOT / 'data' / 'splits'
SETTINGS    = ROOT / 'settings'
SETTINGS.mkdir(exist_ok=True)

MASTER_SEED = 0
DATASET     = 'ag_news'
SPLIT       = 'NLT'  # Gu-style: corrupt the NoisyLabelTrain half of the train split


def per_row_seed(master_seed: int, dataset: str, corruption: str,
                 severity: int, index: int) -> int:
    """SHA-256 derived 64-bit seed (mirrors the image/tabular pipeline)."""
    key = f"{master_seed}|{dataset}|{corruption}|{severity}|{index}"
    digest = hashlib.sha256(key.encode()).digest()
    return struct.unpack('<Q', digest[:8])[0]


def main():
    print('loading ag_news + NLT indices...')
    ds = load_from_disk(str(DATA_DIR))
    nlt_idx = np.load(SPLITS_DIR / 'NLT_indices.npy')
    nlt_subset = ds['train'].select(nlt_idx.tolist())
    texts  = list(nlt_subset['text'])
    labels = np.array(nlt_subset['label'], dtype=np.int64)
    print(f'  {len(texts)} NLT rows')

    for corr_name, corr_fn in RELEASE_CORRUPTIONS.items():
        for severity in (1, 3, 5):
            t0 = time.time()
            setting_name = f"{corr_name}_sev{severity}"
            out_dir = SETTINGS / setting_name
            out_dir.mkdir(parents=True, exist_ok=True)

            corrupted = []
            params_log = []
            for i, text in enumerate(texts):
                seed = per_row_seed(MASTER_SEED, DATASET, corr_name, severity, i)
                rng = np.random.default_rng(seed)
                out, info = corr_fn(text, severity, rng)
                corrupted.append(out)
                # SHA-256 receipt of the produced text
                sha = hashlib.sha256(out.encode()).hexdigest()[:16]
                params_log.append({
                    'idx': i, 'seed': seed, 'sha256': sha,
                    'n_perturbed': info.get('n_perturbed', 0),
                })

            # Save outputs
            np.save(out_dir / 'texts.npy', np.array(corrupted, dtype=object), allow_pickle=True)
            np.save(out_dir / 'labels.npy', labels)
            with (out_dir / 'params.jsonl').open('w') as f:
                for rec in params_log:
                    f.write(json.dumps(rec) + '\n')
            (out_dir / 'manifest.json').write_text(json.dumps({
                'dataset': DATASET, 'split': SPLIT,
                'corruption': corr_name, 'severity': severity,
                'master_seed': MASTER_SEED, 'n_rows': len(texts),
            }, indent=2))
            dt = time.time() - t0
            print(f'  [{setting_name}] saved {len(corrupted)} rows  ({dt:.1f}s)')


if __name__ == '__main__':
    main()
