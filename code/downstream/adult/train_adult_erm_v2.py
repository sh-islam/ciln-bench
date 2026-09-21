"""ERM trainer for Adult downstream evaluation (v2, multi-modality campaign).

Extends train_adult_erm.py with:
  - --variant {S,C}: CILN-S (full NLT) or CILN-C (clean-correct filtered)
  - argmax-uniform sampling via shared module (paper's stated protocol)
  - full logging: per-class recall (@best and @final), per-sample training loss
    at logged epochs, per-sample is_noisy mask, sampling manifest
  - matches the CIFAR CoT logging shape so downstream analysis is uniform

Two image_source modes:
  clean: train on the uncorrupted NLT rows (paper's clean-img scenario)
  noisy: train on the corrupted NLT rows   (paper's noisy-img scenario)

Usage:
  python train_adult_erm_v2.py --setting gaussian_noise_sev5 --variant S --seed 0 \
      --image-source clean --gpu 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MULTI_ROOT = ROOT.parent / "downstream_multimodal"
OUT_DIR = MULTI_ROOT / "results" / "adult"
LOG_DIR = MULTI_ROOT / "logs" / "adult"
SAMPLING_DIR = MULTI_ROOT / "sampling" / "cache" / "adult"
for d in (OUT_DIR, LOG_DIR, SAMPLING_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Shared sampling module
sys.path.insert(0, str(MULTI_ROOT / "sampling"))
from sample_labels import sample_argmax_uniform, save_sampling_artifact  # noqa: E402

# Pull paths to the existing tabular code
JOURNAL_ROOT = ROOT.parent.parent / "journal_edition"
sys.path.insert(0, str(JOURNAL_ROOT / "tabular"))
sys.path.insert(0, str(JOURNAL_ROOT / "tabular" / "train"))
from common import load_adult, split_train_val, CATEGORICAL_COLS, NUMERICAL_COLS, TRAIN_SEED  # noqa: E402
from train_ft_transformer import FTTransformer, CFG as FT_CFG  # noqa: E402

# Voter softmaxes live under journal_edition/output_seed0/adult/{corr}/severity_{sev}/noisy_label_train/
NLT_ROOT = JOURNAL_ROOT / "output_seed0" / "adult"
# Clean-correct data (for CILN-C filter mask)
CLEAN_ROOT = JOURNAL_ROOT / "output_seed0" / "clean" / "adult" / "noisylabeltrain_clean"
# Corrupted parquets live under the HF release
RELEASE_ROOT = ROOT.parent.parent / "release_v1_hf" / "ciln-bench-adult" / "settings"

ADULT_VOTERS = ("xgboost_dummyna", "catboost", "mlp", "ft_transformer", "tabpfn")
NAN_TOKEN = "__NaN__"
EPOCHS = 100
PATIENCE = 16
LOG_EPOCHS = [5, 10, 20, 30, 50, 75, 99]


def parse_setting(setting: str):
    """gaussian_noise_sev5 -> ('gaussian_noise', 5). Handles multi-word corrs."""
    parts = setting.rsplit("_sev", 1)
    return parts[0], int(parts[1])


def build_preprocessing():
    X_train, y_train, X_test, y_test = load_adult()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

    cat_maps = {}
    for c in CATEGORICAL_COLS:
        union_vals = pd.concat([X_tr[c], X_val[c], X_test[c]]).astype(str).unique()
        m = {v: i for i, v in enumerate(sorted(union_vals))}
        m[NAN_TOKEN] = len(m)
        cat_maps[c] = m

    Xn_tr_raw = X_tr[NUMERICAL_COLS].values.astype(np.float32)
    qt = QuantileTransformer(output_distribution="normal", random_state=TRAIN_SEED)
    qt.fit(Xn_tr_raw)
    num_medians = {c: float(np.nanmedian(Xn_tr_raw[:, i])) for i, c in enumerate(NUMERICAL_COLS)}

    cat_cardinalities = [len(cat_maps[c]) for c in CATEGORICAL_COLS]
    return {
        "cat_maps": cat_maps,
        "cat_cardinalities": cat_cardinalities,
        "num_medians": num_medians,
        "qt": qt,
        "X_test": X_test, "y_test": y_test,
    }


def encode(df, pre):
    cat_arr = np.empty((len(df), len(CATEGORICAL_COLS)), dtype=np.int64)
    for j, c in enumerate(CATEGORICAL_COLS):
        col = df[c].astype("object").where(df[c].notna(), NAN_TOKEN).astype(str).values
        m = pre["cat_maps"][c]
        cat_arr[:, j] = np.array([m.get(v, m[NAN_TOKEN]) for v in col], dtype=np.int64)

    num_arr = df[NUMERICAL_COLS].values.astype(np.float32)
    for j, c in enumerate(NUMERICAL_COLS):
        col = num_arr[:, j]
        mask = np.isnan(col)
        if mask.any():
            num_arr[mask, j] = pre["num_medians"][c]
    num_arr = pre["qt"].transform(num_arr).astype(np.float32)
    return num_arr, cat_arr


def load_setting_rows(setting: str, image_source: str, pre, ccp_mask=None):
    corr, sev = parse_setting(setting)
    if image_source == "noisy":
        parquet = RELEASE_ROOT / setting / "noisy_label_train" / "adult_corrupted.parquet"
        df = pd.read_parquet(parquet)
        if ccp_mask is not None:
            df = df.iloc[ccp_mask].reset_index(drop=True)
        return encode(df, pre)
    elif image_source == "clean":
        X_train, y_train, _, _ = load_adult()
        X_tr_clean, _, _, _ = split_train_val(X_train, y_train)
        X_tr_clean = X_tr_clean.reset_index(drop=True)
        if ccp_mask is not None:
            X_tr_clean = X_tr_clean.iloc[ccp_mask].reset_index(drop=True)
        return encode(X_tr_clean, pre)
    else:
        raise ValueError(image_source)


def load_ccp_mask():
    """Load the clean-correct mask (all-voters-correct on clean NLT)."""
    m = np.load(CLEAN_ROOT / "clean_correct_mask.npy")
    return m.astype(bool)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for xn, xc, y, _ in loader:
        xn, xc, y = xn.to(device), xc.to(device), y.to(device)
        opt.zero_grad()
        logits = model(xn, xc)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


@torch.no_grad()
def eval_acc_and_recall(model, loader, device, num_classes=2):
    model.eval()
    correct, n = 0, 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    for xn, xc, y, _ in loader:
        xn, xc, y = xn.to(device), xc.to(device), y.to(device)
        pred = model(xn, xc).argmax(dim=-1)
        correct += int((pred == y).sum()); n += len(y)
        y_np = y.cpu().numpy(); p_np = pred.cpu().numpy()
        for c in range(num_classes):
            m = (y_np == c)
            per_class_total[c] += int(m.sum())
            per_class_correct[c] += int(((p_np == c) & m).sum())
    recall = {int(c): float(per_class_correct[c] / max(1, per_class_total[c])) for c in range(num_classes)}
    return correct / n, recall


@torch.no_grad()
def per_sample_loss(model, loader, device, N):
    model.eval()
    out = np.zeros(N, dtype=np.float32)
    for xn, xc, y, idx in loader:
        xn, xc, y = xn.to(device), xc.to(device), y.to(device)
        logits = model(xn, xc)
        l = F.cross_entropy(logits, y, reduction="none")
        out[idx.numpy()] = l.cpu().numpy()
    return out


class IdxTensorDataset(torch.utils.data.Dataset):
    def __init__(self, Xn, Xc, y):
        self.Xn = torch.from_numpy(Xn); self.Xc = torch.from_numpy(Xc)
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.Xn[i], self.Xc[i], self.y[i], i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--variant", choices=("S", "C"), default="S",
                    help="S = full NLT (CILN-S); C = clean-correct filtered (CILN-C)")
    ap.add_argument("--image-source", choices=("clean", "noisy"), default="noisy")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=FT_CFG["batch_size"])
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    tag = f"{args.setting}__{args.variant}__erm__{args.image_source}__seed{args.seed}"
    print(f"[ERM] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    pre = build_preprocessing()

    # Sample labels using the shared argmax-uniform module
    corr, sev = parse_setting(args.setting)
    nlt_setting_dir = NLT_ROOT / corr / f"severity_{sev}" / "noisy_label_train"
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_setting_dir, ADULT_VOTERS, args.seed, mask=ccp_mask
    )
    print(f"  N={sm_manifest['n_rows']}  sampled_nr={sm_manifest['sampled_noise_rate']*100:.2f}%  "
          f"expected_nr={sm_manifest['expected_noise_rate']*100:.2f}%", flush=True)

    # Persist sampling artifact
    save_sampling_artifact(SAMPLING_DIR / f"{args.setting}__{args.variant}__seed{args.seed}",
                            y_noisy, voter_ids, sm_manifest, y_true)

    # Encode features
    ccp_idx = np.where(ccp_mask)[0] if ccp_mask is not None else None
    Xn, Xc = load_setting_rows(args.setting, args.image_source, pre, ccp_mask=ccp_idx)
    if len(Xn) != len(y_noisy):
        raise RuntimeError(f"Feature/label length mismatch: {len(Xn)} vs {len(y_noisy)}")

    Xn_test, Xc_test = encode(pre["X_test"].reset_index(drop=True), pre)
    y_test = pre["y_test"].values.astype(np.int64)

    train_ds = IdxTensorDataset(Xn, Xc, y_noisy)
    test_ds = IdxTensorDataset(Xn_test, Xc_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(IdxTensorDataset(Xn, Xc, y_noisy),
                                    batch_size=args.batch, shuffle=False, num_workers=0)

    model = FTTransformer(
        n_num=len(NUMERICAL_COLS),
        cat_cardinalities=pre["cat_cardinalities"],
        cfg=FT_CFG,
        num_classes=2,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=FT_CFG["lr"], weight_decay=FT_CFG["weight_decay"])

    is_noisy = (y_noisy != y_true).astype(np.uint8)
    N = len(y_noisy)

    history = []
    per_sample_dumps = {}
    per_class_recalls = {}
    best = {"epoch": -1, "test_acc": -1.0, "recall": None}
    patience_left = PATIENCE
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, opt, device)
        test_acc, recall = eval_acc_and_recall(model, test_loader, device)
        history.append({"epoch": ep, "train_loss": tr_loss, "test_acc": test_acc,
                        "recall": recall})
        if test_acc > best["test_acc"]:
            best = {"epoch": ep, "test_acc": test_acc, "recall": recall}
            patience_left = PATIENCE
        else:
            patience_left -= 1
        if ep in LOG_EPOCHS or ep == args.epochs - 1:
            per_sample_dumps[ep] = per_sample_loss(model, train_eval_loader, device, N)
            per_class_recalls[ep] = recall
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  train_loss={tr_loss:.4f}  test_acc={test_acc:.4f}  "
                  f"best={best['test_acc']:.4f}@{best['epoch']}", flush=True)
        if patience_left <= 0:
            print(f"  early stop at ep {ep}", flush=True)
            break

    final_ep = history[-1]["epoch"]
    result = {
        "method": "erm",
        "dataset": "adult",
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
            "epochs": args.epochs, "batch": args.batch,
            "lr": FT_CFG["lr"], "weight_decay": FT_CFG["weight_decay"],
            "arch": "FTTransformer", "patience": PATIENCE,
        },
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))

    # Per-sample NPZ
    eps = sorted(per_sample_dumps.keys())
    losses_arr = np.stack([per_sample_dumps[e] for e in eps], axis=0)
    np.savez_compressed(OUT_DIR / f"{tag}__persample.npz",
                        epochs=np.array(eps, dtype=np.int64),
                        losses=losses_arr,
                        is_noisy=is_noisy,
                        noisy_labels=y_noisy.astype(np.int64),
                        true_labels=y_true.astype(np.int64),
                        voter_ids=voter_ids.astype(np.int64))
    print(f"  DONE  best={best['test_acc']:.4f}  saved {tag}.json + persample.npz", flush=True)


if __name__ == "__main__":
    main()
