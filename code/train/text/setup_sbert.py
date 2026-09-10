"""Sentence-transformer voter: zero-shot classification by embedding similarity.

No training. The voter computes embeddings for each AG-News category label
("World news", "Sports news", "Business news", "Science and Technology news")
once, then classifies test inputs by cosine similarity to the label embeddings.
Softmax with a CLIP-style temperature of 100.0.

This script just pre-computes and caches the category embeddings so inference
can use them without recomputing.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data' / 'ag_news'
OUT  = ROOT / 'voters' / 'sbert'
OUT.mkdir(parents=True, exist_ok=True)

# Category prompts (mirrors CLIP's "a photo of a {class}" pattern)
PROMPTS = [
    'a news article about world events and politics',
    'a news article about sports',
    'a news article about business and finance',
    'a news article about science and technology',
]
CATEGORIES = ['World', 'Sports', 'Business', 'Sci/Tech']
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def main():
    print('[sbert] loading model + ag_news...', flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    ds = load_from_disk(str(DATA))

    # Precompute prompt embeddings (4 vectors)
    prompt_emb = model.encode(PROMPTS, normalize_embeddings=True)  # (4, d)

    # Evaluate test set
    test_texts = ds['test']['text']
    test_labels = np.array(ds['test']['label'], dtype=np.int64)
    print(f'[sbert] encoding {len(test_texts)} test rows...', flush=True)
    text_emb = model.encode(test_texts, batch_size=128, show_progress_bar=False,
                             normalize_embeddings=True)  # (N, d)

    logits = 100.0 * (text_emb @ prompt_emb.T)  # (N, 4)
    preds = logits.argmax(axis=1)
    acc = float((preds == test_labels).mean())
    print(f'[sbert] zero-shot test acc: {acc:.4f}', flush=True)

    np.save(OUT / 'prompt_embeddings.npy', prompt_emb.astype(np.float32))
    (OUT / 'setup.json').write_text(json.dumps({
        'voter': 'sbert',
        'model': MODEL_NAME,
        'prompts': PROMPTS,
        'categories': CATEGORIES,
        'zero_shot_test_acc': acc,
        'elapsed_sec': time.time() - t0,
    }, indent=2))
    print(f'[sbert] saved prompt embeddings to {OUT}', flush=True)


if __name__ == '__main__':
    main()
