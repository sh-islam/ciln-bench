"""Run the 4 AG-News voters on the CLEAN (un-corrupted) NLT split.

This produces the softmaxes needed to build the clean-correct mask for CILN-C.
Outputs land in ciln_text/clean_nlt/ as softmax_<voter>.npy + labels.npy.

Usage:
  python run_clean_inference.py [--gpu N]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VOTERS_DIR = ROOT / 'voters'
DATA_DIR = ROOT / 'data' / 'ag_news'
SPLITS_DIR = ROOT / 'data' / 'splits'
OUT = ROOT / 'clean_nlt'
OUT.mkdir(exist_ok=True)
N_CLASSES = 4


def _patch_fasttext_numpy():
    import numpy as np
    import fasttext.FastText as _ft
    orig = _ft._FastText.predict
    real_array = np.array
    def shim(obj, *a, **kw):
        kw.pop('copy', None)
        return real_array(obj, *a, **kw)
    def patched(self, *a, **kw):
        np.array = shim
        try: return orig(self, *a, **kw)
        finally: np.array = real_array
    _ft._FastText.predict = patched


def predict_fasttext(model, texts):
    out = np.zeros((len(texts), N_CLASSES), dtype=np.float32)
    for i, t in enumerate(texts):
        text = t.replace('\n', ' ').replace('\r', ' ')
        labels, probs = model.predict(text, k=N_CLASSES)
        for lbl, p in zip(labels, probs):
            cls = int(lbl.replace('__label__', ''))
            out[i, cls] = p
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


@torch.no_grad()
def predict_transformer(model, tokenizer, texts, device, batch_size=64):
    out = np.empty((len(texts), N_CLASSES), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        e = min(s + batch_size, len(texts))
        enc = tokenizer(list(texts[s:e]), padding=True, truncation=True,
                        max_length=128, return_tensors='pt').to(device)
        logits = model(**enc).logits
        out[s:e] = torch.softmax(logits, dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def predict_sbert(sbert_model, prompt_emb, texts, batch_size=128):
    text_emb = sbert_model.encode(list(texts), batch_size=batch_size,
                                   show_progress_bar=False, normalize_embeddings=True)
    logits = 100.0 * (text_emb @ prompt_emb.T)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f'[clean-infer] device={device}')

    # Load NLT clean texts + labels
    print('[clean-infer] loading AG-News + NLT indices...')
    ds = load_from_disk(str(DATA_DIR))
    nlt_idx = np.load(SPLITS_DIR / 'NLT_indices.npy')
    nlt = ds['train'].select(nlt_idx.tolist())
    texts = list(nlt['text'])
    labels = np.array(nlt['label'], dtype=np.int64)
    print(f'  NLT size: {len(texts)} rows')

    np.save(OUT / 'labels.npy', labels)
    print(f'  Saved {OUT / "labels.npy"}')

    # fastText
    print('[clean-infer] fastText...')
    t0 = time.time()
    _patch_fasttext_numpy()
    import fasttext
    ft_model = fasttext.load_model(str(VOTERS_DIR / 'fasttext' / 'model.bin'))
    p_ft = predict_fasttext(ft_model, texts)
    np.save(OUT / 'softmax_fasttext.npy', p_ft)
    print(f'  acc={float((p_ft.argmax(1) == labels).mean()):.4f}  saved ({time.time()-t0:.1f}s)')

    # DistilBERT
    print('[clean-infer] DistilBERT...')
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    db_tok = AutoTokenizer.from_pretrained(str(VOTERS_DIR / 'distilbert' / 'final'))
    db_mod = AutoModelForSequenceClassification.from_pretrained(
        str(VOTERS_DIR / 'distilbert' / 'final')).eval().to(device)
    p_db = predict_transformer(db_mod, db_tok, texts, device)
    np.save(OUT / 'softmax_distilbert.npy', p_db)
    print(f'  acc={float((p_db.argmax(1) == labels).mean()):.4f}  saved ({time.time()-t0:.1f}s)')

    # RoBERTa
    print('[clean-infer] RoBERTa-base...')
    t0 = time.time()
    rb_tok = AutoTokenizer.from_pretrained(str(VOTERS_DIR / 'roberta-base' / 'final'))
    rb_mod = AutoModelForSequenceClassification.from_pretrained(
        str(VOTERS_DIR / 'roberta-base' / 'final')).eval().to(device)
    p_rb = predict_transformer(rb_mod, rb_tok, texts, device)
    np.save(OUT / 'softmax_roberta.npy', p_rb)
    print(f'  acc={float((p_rb.argmax(1) == labels).mean()):.4f}  saved ({time.time()-t0:.1f}s)')

    # SBERT (zero-shot)
    print('[clean-infer] sentence-transformer (zero-shot)...')
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    prompt_emb = np.load(VOTERS_DIR / 'sbert' / 'prompt_embeddings.npy')
    p_sb = predict_sbert(sbert, prompt_emb, texts)
    np.save(OUT / 'softmax_sbert.npy', p_sb)
    print(f'  acc={float((p_sb.argmax(1) == labels).mean()):.4f}  saved ({time.time()-t0:.1f}s)')

    print('[clean-infer] DONE')


if __name__ == '__main__':
    main()
