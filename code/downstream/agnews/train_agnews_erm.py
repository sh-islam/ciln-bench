"""ERM trainer for AG-News downstream (multi-modality campaign).

TextCNN (Kim 2014) on AG-News NLT rows with argmax-uniform voter-sampled labels.
- 4 classes: World, Sports, Business, Sci-Tech
- Filter sizes {3, 4, 5}, 100 filters each, dropout 0.5
- 300-dim word embeddings, trained from scratch (fits in RAM)
- Adam lr=1e-3, batch 64, 100 epochs

Two image_source modes:
  clean: train on the uncorrupted NLT text
  noisy: train on the corrupted NLT text (setting-specific)

Usage:
  python train_agnews_erm.py --setting butter_fingers_sev5 --variant S \
      --image-source noisy --seed 0 --gpu 1
"""
from __future__ import annotations
import argparse, json, sys, time, re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
MULTI_ROOT = HERE.parent
OUT_DIR = MULTI_ROOT / "results" / "agnews"
LOG_DIR = MULTI_ROOT / "logs" / "agnews"
SAMPLING_DIR = MULTI_ROOT / "sampling" / "cache" / "agnews"
CACHE_DIR = MULTI_ROOT / "cache" / "agnews"
for d in (OUT_DIR, LOG_DIR, SAMPLING_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MULTI_ROOT / "sampling"))
from sample_labels import sample_argmax_uniform, save_sampling_artifact  # noqa: E402

CILN_TEXT_ROOT = Path("/path/to/ciln-workspace/ciln_text")
NLT_ROOT = CILN_TEXT_ROOT / "settings"
CLEAN_ROOT = CILN_TEXT_ROOT / "clean_nlt"
SPLITS_ROOT = CILN_TEXT_ROOT / "data" / "splits"
CLEAN_DS_ROOT = CILN_TEXT_ROOT / "data" / "ag_news"

AGNEWS_VOTERS = ("distilbert", "fasttext", "roberta", "sbert")
NUM_CLASSES = 4
LOG_EPOCHS = [5, 10, 20, 30, 50, 75, 99]

# Text config
MAX_LEN = 200
EMBED_DIM = 300
VOCAB_MIN_FREQ = 2
DROPOUT = 0.5
FILTER_SIZES = (3, 4, 5)
NUM_FILTERS = 100


def parse_setting(setting: str):
    parts = setting.rsplit("_sev", 1)
    return parts[0], int(parts[1])


def load_ccp_mask():
    m = np.load(CLEAN_ROOT / "clean_correct_mask.npy")
    return m.astype(bool)


def load_nlt_texts_clean():
    """Load the clean NLT texts (indexed into the full AG-News train)."""
    from datasets import load_from_disk
    ds = load_from_disk(str(CLEAN_DS_ROOT / "train"))
    nlt_idx = np.load(SPLITS_ROOT / "NLT_indices.npy")
    return np.array([ds[int(i)]["text"] for i in nlt_idx], dtype=object)


def load_nlt_texts_noisy(setting: str):
    d = NLT_ROOT / setting
    return np.load(d / "texts.npy", allow_pickle=True)


def load_test():
    from datasets import load_from_disk
    ds = load_from_disk(str(CLEAN_DS_ROOT / "test"))
    texts = np.array([r["text"] for r in ds], dtype=object)
    labels = np.array([r["label"] for r in ds], dtype=np.int64)
    return texts, labels


TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+")


def tokenize(s: str):
    return TOKEN_RE.findall(s.lower())


def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    # <pad>=0, <unk>=1
    itos = ["<pad>", "<unk>"] + [w for w, c in counter.most_common() if c >= VOCAB_MIN_FREQ]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi


def encode_texts(texts, stoi, max_len=MAX_LEN):
    unk = stoi["<unk>"]
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = tokenize(t)[:max_len]
        for j, w in enumerate(toks):
            out[i, j] = stoi.get(w, unk)
    return out


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, NUM_FILTERS, fs) for fs in FILTER_SIZES
        ])
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(NUM_FILTERS * len(FILTER_SIZES), num_classes)

    def forward(self, x):
        # x: (B, T)
        e = self.embed(x).transpose(1, 2)  # (B, E, T)
        outs = []
        for conv in self.convs:
            c = F.relu(conv(e))                # (B, F, T-fs+1)
            p = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (B, F)
            outs.append(p)
        h = torch.cat(outs, dim=1)             # (B, F*|FS|)
        h = self.dropout(h)
        return self.fc(h)


class ArrayIdxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i], i


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class_c = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_t = np.zeros(NUM_CLASSES, dtype=np.int64)
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y_t = y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y_t).sum().item(); total += y_t.numel()
            y_np = y_t.cpu().numpy(); p_np = pred.cpu().numpy()
            for c in range(NUM_CLASSES):
                m = (y_np == c)
                per_class_t[c] += int(m.sum())
                per_class_c[c] += int(((p_np == c) & m).sum())
    recall = {int(c): float(per_class_c[c] / max(1, per_class_t[c])) for c in range(NUM_CLASSES)}
    return correct / total, recall


def per_sample_loss(model, loader, device, N):
    model.eval()
    out = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device); y_t = y.to(device)
            l = F.cross_entropy(model(x), y_t, reduction="none")
            out[idx.numpy()] = l.cpu().numpy()
    return out


def load_or_build_vocab(clean_texts):
    """Build vocab from the clean NLT text (voter-side signal, deterministic).
    Cached so multi runs don't rebuild."""
    cache = CACHE_DIR / "vocab.json"
    if cache.exists():
        stoi = json.loads(cache.read_text())
        return stoi
    print("Building vocab from clean NLT...", flush=True)
    stoi = build_vocab(clean_texts)
    cache.write_text(json.dumps(stoi))
    print(f"  vocab size: {len(stoi)}", flush=True)
    return stoi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--variant", choices=("S", "C"), default="S")
    ap.add_argument("--image-source", choices=("clean", "noisy"), default="noisy")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    tag = f"{args.setting}__{args.variant}__erm__{args.image_source}__seed{args.seed}"
    print(f"[AGN ERM] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    # 1. Sample labels
    nlt_dir = NLT_ROOT / args.setting
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_dir, AGNEWS_VOTERS, args.seed, mask=ccp_mask
    )
    N = len(y_noisy)
    print(f"  N={N}  sampled_nr={sm_manifest['sampled_noise_rate']*100:.2f}%", flush=True)
    save_sampling_artifact(SAMPLING_DIR / f"{args.setting}__{args.variant}__seed{args.seed}",
                            y_noisy, voter_ids, sm_manifest, y_true)

    # 2. Load texts (clean or noisy) for training and apply CILN-C mask
    if args.image_source == "clean":
        train_texts = load_nlt_texts_clean()
    else:
        train_texts = load_nlt_texts_noisy(args.setting)
    if ccp_mask is not None:
        train_texts = train_texts[ccp_mask]
    assert len(train_texts) == N, f"text/label mismatch: {len(train_texts)} vs {N}"

    # 3. Vocab: build from clean NLT text (fixed across runs)
    clean_texts_all = load_nlt_texts_clean()
    stoi = load_or_build_vocab(clean_texts_all)
    vocab_size = max(stoi.values()) + 1

    # 4. Encode
    X = encode_texts(train_texts, stoi)
    test_texts, test_lbl = load_test()
    X_test = encode_texts(test_texts, stoi)

    train_ds = ArrayIdxDataset(X, y_noisy)
    train_eval_ds = ArrayIdxDataset(X, y_noisy)
    test_ds = ArrayIdxDataset(X_test, test_lbl)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = TextCNN(vocab_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    is_noisy = (y_noisy != y_true).astype(np.uint8)

    history = []
    per_sample_dumps = {}
    best = {"epoch": -1, "test_acc": -1.0, "recall": None}
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(device); y_t = y.to(device)
            optim.zero_grad()
            loss = F.cross_entropy(model(x), y_t)
            loss.backward(); optim.step()
            losses.append(loss.item())
        sched.step()
        test_acc, recall = evaluate(model, test_loader, device)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "test_acc": test_acc, "recall": recall,
                        "lr": float(sched.get_last_lr()[0])})
        if test_acc > best["test_acc"]:
            best = {"epoch": ep, "test_acc": test_acc, "recall": recall}
        if ep in LOG_EPOCHS or ep == args.epochs - 1:
            per_sample_dumps[ep] = per_sample_loss(model, train_eval_loader, device, N)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  train_loss={history[-1]['train_loss']:.4f}  test_acc={test_acc*100:.2f}%  "
                  f"best={best['test_acc']*100:.2f}%@{best['epoch']}", flush=True)

    result = {
        "method": "erm",
        "dataset": "agnews",
        "setting": args.setting,
        "variant": args.variant,
        "image_source": args.image_source,
        "seed": args.seed,
        "n_train": int(N),
        "best_test_acc": best["test_acc"],
        "best_epoch": best["epoch"],
        "best_per_class_recall": best["recall"],
        "final_test_acc": history[-1]["test_acc"],
        "final_per_class_recall": history[-1]["recall"],
        "history": history,
        "log_epochs": sorted(per_sample_dumps.keys()),
        "sampling_manifest": sm_manifest,
        "hyperparameters": {
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "arch": "TextCNN(Kim2014)", "embed_dim": EMBED_DIM,
            "filter_sizes": list(FILTER_SIZES), "num_filters": NUM_FILTERS,
            "dropout": DROPOUT, "max_len": MAX_LEN, "vocab_size": vocab_size,
            "optim": "Adam", "weight_decay": 1e-5, "sched": "CosineAnnealingLR",
        },
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))

    eps = sorted(per_sample_dumps.keys())
    losses_arr = np.stack([per_sample_dumps[e] for e in eps], axis=0)
    np.savez_compressed(OUT_DIR / f"{tag}__persample.npz",
                        epochs=np.array(eps, dtype=np.int64),
                        losses=losses_arr,
                        is_noisy=is_noisy,
                        noisy_labels=y_noisy.astype(np.int64),
                        true_labels=y_true.astype(np.int64),
                        voter_ids=voter_ids.astype(np.int64))
    print(f"  DONE  best={best['test_acc']*100:.2f}%  saved {tag}.json + persample.npz", flush=True)


if __name__ == "__main__":
    main()
