"""Co-Teaching trainer for Adult (v2, multi-modality campaign).

Extends train_adult_coteaching.py with:
  - --variant {S,C}: CILN-S / CILN-C
  - shared argmax-uniform sampling module + sampling manifest persisted
  - full logging: per-class recall (@best and @final), per-sample losses at
    LOG_EPOCHS for both nets, selection masks, R(t), sampling voter_ids

Usage:
  python train_adult_coteaching_v2.py --setting scaling_sev5 --variant S \
      --image-source noisy --seed 0 --gpu 2
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MULTI_ROOT = ROOT.parent / "downstream_multimodal"
OUT_DIR = MULTI_ROOT / "results" / "adult"
LOG_DIR = MULTI_ROOT / "logs" / "adult"
SAMPLING_DIR = MULTI_ROOT / "sampling" / "cache" / "adult"
for d in (OUT_DIR, LOG_DIR, SAMPLING_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Shared sampling
sys.path.insert(0, str(MULTI_ROOT / "sampling"))
from sample_labels import sample_argmax_uniform, save_sampling_artifact  # noqa: E402

# Reuse ERM v2 utilities
sys.path.insert(0, str(HERE))
from train_adult_erm_v2 import (
    build_preprocessing, encode, load_setting_rows, load_ccp_mask,
    parse_setting, IdxTensorDataset, per_sample_loss,
    eval_acc_and_recall, ADULT_VOTERS, NLT_ROOT, EPOCHS, PATIENCE, LOG_EPOCHS,
)

JOURNAL_ROOT = ROOT.parent.parent / "journal_edition"
sys.path.insert(0, str(JOURNAL_ROOT / "tabular"))
sys.path.insert(0, str(JOURNAL_ROOT / "tabular" / "train"))
from common import NUMERICAL_COLS  # noqa: E402
from train_ft_transformer import FTTransformer, CFG as FT_CFG  # noqa: E402


def build_ft(cat_cardinalities, device):
    m = FTTransformer(n_num=len(NUMERICAL_COLS),
                       cat_cardinalities=cat_cardinalities,
                       cfg=FT_CFG, num_classes=2).to(device)
    o = torch.optim.AdamW(m.parameters(), lr=FT_CFG["lr"],
                           weight_decay=FT_CFG["weight_decay"])
    return m, o


def per_sample_ce(model, xn, xc, y):
    logits = model(xn, xc)
    return F.cross_entropy(logits, y, reduction="none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--variant", choices=("S", "C"), default="S")
    ap.add_argument("--image-source", choices=("clean", "noisy"), default="noisy")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=FT_CFG["batch_size"])
    ap.add_argument("--Tk", type=int, default=10)
    ap.add_argument("--ramp-exponent", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    tag = f"{args.setting}__{args.variant}__coteaching__{args.image_source}__seed{args.seed}"
    print(f"[CoT] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    pre = build_preprocessing()

    corr, sev = parse_setting(args.setting)
    nlt_setting_dir = NLT_ROOT / corr / f"severity_{sev}" / "noisy_label_train"
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_setting_dir, ADULT_VOTERS, args.seed, mask=ccp_mask
    )
    print(f"  N={sm_manifest['n_rows']}  sampled_nr={sm_manifest['sampled_noise_rate']*100:.2f}%  "
          f"expected_nr={sm_manifest['expected_noise_rate']*100:.2f}%", flush=True)
    save_sampling_artifact(SAMPLING_DIR / f"{args.setting}__{args.variant}__seed{args.seed}",
                            y_noisy, voter_ids, sm_manifest, y_true)

    ccp_idx = np.where(ccp_mask)[0] if ccp_mask is not None else None
    Xn, Xc = load_setting_rows(args.setting, args.image_source, pre, ccp_mask=ccp_idx)

    Xn_test, Xc_test = encode(pre["X_test"].reset_index(drop=True), pre)
    y_test = pre["y_test"].values.astype(np.int64)

    train_ds = IdxTensorDataset(Xn, Xc, y_noisy)
    test_ds = IdxTensorDataset(Xn_test, Xc_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(IdxTensorDataset(Xn, Xc, y_noisy),
                                    batch_size=args.batch, shuffle=False, num_workers=0)

    net_a, opt_a = build_ft(pre["cat_cardinalities"], device)
    net_b, opt_b = build_ft(pre["cat_cardinalities"], device)

    tau = sm_manifest["sampled_noise_rate"]
    N = len(y_noisy)
    is_noisy = (y_noisy != y_true).astype(np.uint8)

    def forget_rate(ep):
        ramp = min(1.0, (ep / args.Tk) ** args.ramp_exponent)
        return tau * ramp

    history = []
    per_sample_dumps_a = {}
    per_sample_dumps_b = {}
    selected_a_logs = {}
    selected_b_logs = {}
    best_ens = {"epoch": -1, "test_acc": -1.0, "recall": None}
    patience_left = PATIENCE

    for ep in range(args.epochs):
        fr = forget_rate(ep)
        keep_frac = 1.0 - fr
        net_a.train(); net_b.train()
        losses_a_batch, losses_b_batch = [], []
        sel_a = np.zeros(N, dtype=np.uint8)
        sel_b = np.zeros(N, dtype=np.uint8)
        for xn, xc, y, idx in train_loader:
            xn, xc, y = xn.to(device), xc.to(device), y.to(device)
            keep_n = max(1, int(len(y) * keep_frac))
            loss_a = per_sample_ce(net_a, xn, xc, y)
            loss_b = per_sample_ce(net_b, xn, xc, y)
            idx_b_for_a = torch.argsort(loss_b)[:keep_n]
            idx_a_for_b = torch.argsort(loss_a)[:keep_n]
            if ep in LOG_EPOCHS:
                g = idx.numpy()
                sel_b[g[idx_b_for_a.cpu().numpy()]] = 1  # B picked these for A
                sel_a[g[idx_a_for_b.cpu().numpy()]] = 1  # A picked these for B

            loss_a_sub = per_sample_ce(net_a, xn[idx_b_for_a], xc[idx_b_for_a], y[idx_b_for_a]).mean()
            loss_b_sub = per_sample_ce(net_b, xn[idx_a_for_b], xc[idx_a_for_b], y[idx_a_for_b]).mean()
            opt_a.zero_grad(); loss_a_sub.backward(); opt_a.step()
            opt_b.zero_grad(); loss_b_sub.backward(); opt_b.step()
            losses_a_batch.append(float(loss_a_sub.item()))
            losses_b_batch.append(float(loss_b_sub.item()))

        acc_a, recall_a = eval_acc_and_recall(net_a, test_loader, device)
        acc_b, recall_b = eval_acc_and_recall(net_b, test_loader, device)
        # Ensemble
        net_a.eval(); net_b.eval()
        correct, ntest = 0, 0
        per_class_c = np.zeros(2, dtype=np.int64)
        per_class_t = np.zeros(2, dtype=np.int64)
        with torch.no_grad():
            for xn, xc, y, _ in test_loader:
                xn, xc, y = xn.to(device), xc.to(device), y.to(device)
                p_a = F.softmax(net_a(xn, xc), dim=-1)
                p_b = F.softmax(net_b(xn, xc), dim=-1)
                pred = (p_a + p_b).argmax(dim=-1)
                correct += int((pred == y).sum()); ntest += len(y)
                y_np = y.cpu().numpy(); p_np = pred.cpu().numpy()
                for c in range(2):
                    m = (y_np == c)
                    per_class_t[c] += int(m.sum())
                    per_class_c[c] += int(((p_np == c) & m).sum())
        acc_ens = correct / ntest
        recall_ens = {int(c): float(per_class_c[c] / max(1, per_class_t[c])) for c in range(2)}

        history.append({"epoch": ep, "forget_rate": fr,
                        "train_loss_a": float(np.mean(losses_a_batch)),
                        "train_loss_b": float(np.mean(losses_b_batch)),
                        "test_acc_a": acc_a, "test_acc_b": acc_b,
                        "test_acc_ens": acc_ens,
                        "recall_ens": recall_ens})
        if acc_ens > best_ens["test_acc"]:
            best_ens = {"epoch": ep, "test_acc": acc_ens, "recall": recall_ens}
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if ep in LOG_EPOCHS or ep == args.epochs - 1:
            per_sample_dumps_a[ep] = per_sample_loss(net_a, train_eval_loader, device, N)
            per_sample_dumps_b[ep] = per_sample_loss(net_b, train_eval_loader, device, N)
            selected_a_logs[ep] = sel_a
            selected_b_logs[ep] = sel_b

        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  fr={fr:.3f}  a={acc_a:.4f}  b={acc_b:.4f}  ens={acc_ens:.4f}  "
                  f"best_ens={best_ens['test_acc']:.4f}@{best_ens['epoch']}", flush=True)
        if patience_left <= 0:
            print(f"  early stop at ep {ep}", flush=True)
            break

    final_ep = history[-1]["epoch"]
    result = {
        "method": "coteaching",
        "dataset": "adult",
        "setting": args.setting,
        "variant": args.variant,
        "image_source": args.image_source,
        "seed": args.seed,
        "n_train": int(N),
        "tau": float(tau),
        "Tk": args.Tk,
        "ramp_exponent": args.ramp_exponent,
        "best_test_acc": best_ens["test_acc"],
        "best_epoch": best_ens["epoch"],
        "best_per_class_recall": best_ens["recall"],
        "final_test_acc": history[-1]["test_acc_ens"],
        "final_per_class_recall": history[-1]["recall_ens"],
        "history": history,
        "log_epochs": sorted(per_sample_dumps_a.keys()),
        "sampling_manifest": sm_manifest,
        "hyperparameters": {
            "epochs": args.epochs, "batch": args.batch,
            "lr": FT_CFG["lr"], "weight_decay": FT_CFG["weight_decay"],
            "arch": "FTTransformer", "patience": PATIENCE, "Tk": args.Tk,
            "ramp_exponent": args.ramp_exponent,
        },
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))

    eps = sorted(per_sample_dumps_a.keys())
    losses_a_arr = np.stack([per_sample_dumps_a[e] for e in eps], axis=0)
    losses_b_arr = np.stack([per_sample_dumps_b[e] for e in eps], axis=0)
    sel_a_arr = np.stack([selected_a_logs[e] for e in eps], axis=0)
    sel_b_arr = np.stack([selected_b_logs[e] for e in eps], axis=0)
    R_t_arr = np.array([1.0 - forget_rate(e) for e in eps], dtype=np.float32)
    np.savez_compressed(OUT_DIR / f"{tag}__persample.npz",
                        epochs=np.array(eps, dtype=np.int64),
                        losses1=losses_a_arr, losses2=losses_b_arr,
                        selected1=sel_a_arr, selected2=sel_b_arr,
                        R_t=R_t_arr,
                        is_noisy=is_noisy,
                        noisy_labels=y_noisy.astype(np.int64),
                        true_labels=y_true.astype(np.int64),
                        voter_ids=voter_ids.astype(np.int64))
    print(f"  DONE  best_ens={best_ens['test_acc']:.4f}  saved {tag}.json + persample.npz", flush=True)


if __name__ == "__main__":
    main()
