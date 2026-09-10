"""Train a fastText classifier on clean AG-News and save the model.

Bag-of-words baseline with subword embeddings. Very fast on CPU.
"""
from __future__ import annotations
import json, tempfile, time
from pathlib import Path

import numpy as np
import fasttext
# Patch the numpy 2.0 incompat in fasttext's predict
import fasttext.FastText as _ft
_orig_predict = _ft._FastText.predict
def _patched_predict(self, *args, **kwargs):
    # Monkey-patch np.array to allow copies in this call
    real = np.array
    def shim(obj, *a, **kw):
        kw.pop('copy', None)
        return real(obj, *a, **kw)
    np.array = shim
    try:
        return _orig_predict(self, *args, **kwargs)
    finally:
        np.array = real
_ft._FastText.predict = _patched_predict

from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data' / 'ag_news'
OUT  = ROOT / 'voters' / 'fasttext'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print('[fasttext] loading ag_news...', flush=True)
    t0 = time.time()
    ds = load_from_disk(str(DATA))
    clt_idx = np.load(ROOT / 'data' / 'splits' / 'CLT_indices.npy')
    train_subset = ds['train'].select(clt_idx.tolist())
    print(f'[fasttext] training on CLT only: {len(train_subset)} rows', flush=True)

    # Write training file in fastText format
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for r in train_subset:
            label = f'__label__{r["label"]}'
            # Strip newlines so each line is one record.
            text = r['text'].replace('\n', ' ').replace('\r', ' ')
            f.write(f'{label} {text}\n')
        train_path = f.name

    print(f'[fasttext] training (epoch=5, dim=100, wordNgrams=2)...', flush=True)
    model = fasttext.train_supervised(
        input=train_path, epoch=5, lr=1.0, dim=100,
        wordNgrams=2, minCount=1, bucket=200000, loss='softmax',
    )
    print(f'[fasttext] trained in {time.time()-t0:.1f}s', flush=True)

    # Evaluate on test
    correct, total = 0, 0
    for r in ds['test']:
        text = r['text'].replace('\n', ' ').replace('\r', ' ')
        pred_label, _ = model.predict(text)
        pred = int(pred_label[0].replace('__label__', ''))
        correct += int(pred == r['label']); total += 1
    acc = correct / total
    print(f'[fasttext] test acc: {acc:.4f}', flush=True)

    model_path = OUT / 'model.bin'
    model.save_model(str(model_path))
    (OUT / 'train_summary.json').write_text(json.dumps({
        'voter': 'fasttext',
        'test_acc': acc,
        'elapsed_sec': time.time() - t0,
    }, indent=2))
    print(f'[fasttext] saved to {model_path}', flush=True)


if __name__ == '__main__':
    main()
