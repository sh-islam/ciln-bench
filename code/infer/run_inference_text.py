"""Run all 4 voters on all 12 corrupted AG-News settings, save softmaxes.

Usage:
  python run_inference.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VOTERS_DIR = ROOT / 'voters'
SETTINGS = ROOT / 'settings'
N_CLASSES = 4


# ===================== fastText voter =====================
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
    """Return (N, 4) softmax-ish probs."""
    out = np.zeros((len(texts), N_CLASSES), dtype=np.float32)
    for i, t in enumerate(texts):
        text = t.replace('\n', ' ').replace('\r', ' ')
        # k=4 returns all 4 class probs.
        labels, probs = model.predict(text, k=N_CLASSES)
        for lbl, p in zip(labels, probs):
            cls = int(lbl.replace('__label__', ''))
            out[i, cls] = p
    # fastText softmax loss makes these sum to ~1 already; normalize anyway
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


# ===================== transformer voter =====================
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


# ===================== sentence-transformer voter =====================
@torch.no_grad()
def predict_sbert(sbert_model, prompt_emb, texts, batch_size=128):
    text_emb = sbert_model.encode(list(texts), batch_size=batch_size,
                                   show_progress_bar=False, normalize_embeddings=True)
    logits = 100.0 * (text_emb @ prompt_emb.T)
    # softmax
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load all voters
    print('[infer] loading voters...', flush=True)
    _patch_fasttext_numpy()
    import fasttext
    ft_model = fasttext.load_model(str(VOTERS_DIR / 'fasttext' / 'model.bin'))

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    db_tok = AutoTokenizer.from_pretrained(str(VOTERS_DIR / 'distilbert' / 'final'))
    db_mod = AutoModelForSequenceClassification.from_pretrained(
        str(VOTERS_DIR / 'distilbert' / 'final')).eval().to(device)
    rb_tok = AutoTokenizer.from_pretrained(str(VOTERS_DIR / 'roberta-base' / 'final'))
    rb_mod = AutoModelForSequenceClassification.from_pretrained(
        str(VOTERS_DIR / 'roberta-base' / 'final')).eval().to(device)

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    prompt_emb = np.load(VOTERS_DIR / 'sbert' / 'prompt_embeddings.npy')

    # Walk all settings
    setting_dirs = sorted([d for d in SETTINGS.iterdir() if d.is_dir()])
    print(f'[infer] {len(setting_dirs)} settings found', flush=True)

    for sd in setting_dirs:
        t0 = time.time()
        texts = np.load(sd / 'texts.npy', allow_pickle=True)
        texts = [str(t) for t in texts]
        print(f'  [{sd.name}] N={len(texts)}', flush=True)

        p_ft = predict_fasttext(ft_model, texts)
        p_db = predict_transformer(db_mod, db_tok, texts, device)
        p_rb = predict_transformer(rb_mod, rb_tok, texts, device)
        p_sb = predict_sbert(sbert, prompt_emb, texts)

        np.save(sd / 'softmax_fasttext.npy',   p_ft)
        np.save(sd / 'softmax_distilbert.npy', p_db)
        np.save(sd / 'softmax_roberta.npy',    p_rb)
        np.save(sd / 'softmax_sbert.npy',      p_sb)
        # Average
        avg = (p_ft + p_db + p_rb + p_sb) / 4.0
        np.save(sd / 'avg_softmax.npy', avg)
        print(f'    saved 4 softmaxes ({time.time()-t0:.1f}s)', flush=True)

    print('[infer] DONE', flush=True)


if __name__ == '__main__':
    main()
