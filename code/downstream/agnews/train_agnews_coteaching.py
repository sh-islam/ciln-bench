"""Co-Teaching trainer for AG-News (multi-modality campaign).

Two TextCNN networks trained in parallel with cross-selection.
Standard Han et al. 2018 protocol.

Usage:
  python train_agnews_coteaching.py --setting butter_fingers_sev5 --variant S \
      --image-source noisy --seed 0 --gpu 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

sys.path.insert(0, str(HERE))
from train_agnews_erm import (
    TextCNN, ArrayIdxDataset, parse_setting, load_ccp_mask,
    load_nlt_texts_clean, load_nlt_texts_noisy, load_test,
    encode_texts, load_or_build_vocab, evaluate, per_sample_loss,
    AGNEWS_VOTERS, NLT_ROOT, LOG_EPOCHS, NUM_CLASSES,
)


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
    ap.add_argument("--Tk", type=int, default=10)
    ap.add_argument("--ramp-exponent", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    tag = f"{args.setting}__{args.variant}__coteaching__{args.image_source}__seed{args.seed}"
    print(f"[AGN CoT] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    nlt_dir = NLT_ROOT / args.setting
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_dir, AGNEWS_VOTERS, args.seed, mask=ccp_mask
    )
    N = len(y_noisy)
    tau = sm_manifest["sampled_noise_rate"]
    is_noisy = (y_noisy != y_true).astype(np.uint8)
    print(f"  N={N}  tau={tau*100:.2f}%", flush=True)
    save_sampling_artifact(SAMPLING_DIR / f"{args.setting}__{args.variant}__seed{args.seed}",
                            y_noisy, voter_ids, sm_manifest, y_true)

    if args.image_source == "clean":
        train_texts = load_nlt_texts_clean()
    else:
        train_texts = load_nlt_texts_noisy(args.setting)
    if ccp_mask is not None:
        train_texts = train_texts[ccp_mask]

    clean_texts_all = load_nlt_texts_clean()
    stoi = load_or_build_vocab(clean_texts_all)
    vocab_size = max(stoi.values()) + 1

    X = encode_texts(train_texts, stoi)
    test_texts, test_lbl = load_test()
    X_test = encode_texts(test_texts, stoi)

    train_ds = ArrayIdxDataset(X, y_noisy)
    train_eval_ds = ArrayIdxDataset(X, y_noisy)
    test_ds = ArrayIdxDataset(X_test, test_lbl)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model1 = TextCNN(vocab_size).to(device)
    model2 = TextCNN(vocab_size).to(device)
    optim1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-5)
    optim2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-5)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=args.epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=args.epochs)

    def R_t(ep):
        ramp = min(1.0, (ep / args.Tk) ** args.ramp_exponent)
        return 1.0 - tau * ramp

    history = []
    per_sample_dumps1 = {}; per_sample_dumps2 = {}
    selected1_logs = {}; selected2_logs = {}
    best_ens = {"epoch": -1, "test_acc": -1.0, "recall": None}

    for ep in range(args.epochs):
        model1.train(); model2.train()
        r = R_t(ep)
        sel1 = np.zeros(N, dtype=np.uint8)
        sel2 = np.zeros(N, dtype=np.uint8)
        losses1_ep, losses2_ep = [], []
        for x, y, idx in train_loader:
            x = x.to(device); y_t = y.to(device)
            B = y_t.size(0)
            n_keep = max(1, int(round(r * B)))
            out1 = model1(x); out2 = model2(x)
            l1 = F.cross_entropy(out1, y_t, reduction='none')
            l2 = F.cross_entropy(out2, y_t, reduction='none')
            _, idx1_keep = torch.topk(l1, n_keep, largest=False)
            _, idx2_keep = torch.topk(l2, n_keep, largest=False)

            if ep in LOG_EPOCHS:
                g = idx.numpy()
                sel1[g[idx1_keep.cpu().numpy()]] = 1
                sel2[g[idx2_keep.cpu().numpy()]] = 1

            optim1.zero_grad()
            loss1 = F.cross_entropy(out1[idx2_keep], y_t[idx2_keep])
            loss1.backward(retain_graph=True); optim1.step()
            optim2.zero_grad()
            loss2 = F.cross_entropy(out2[idx1_keep], y_t[idx1_keep])
            loss2.backward(); optim2.step()
            losses1_ep.append(float(loss1.item()))
            losses2_ep.append(float(loss2.item()))
        sched1.step(); sched2.step()

        acc1, recall1 = evaluate(model1, test_loader, device)
        acc2, recall2 = evaluate(model2, test_loader, device)
        model1.eval(); model2.eval()
        correct, ntest = 0, 0
        per_class_c = np.zeros(NUM_CLASSES, dtype=np.int64)
        per_class_t = np.zeros(NUM_CLASSES, dtype=np.int64)
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device); y_t = y.to(device)
                p1 = F.softmax(model1(x), dim=-1); p2 = F.softmax(model2(x), dim=-1)
                pred = (p1 + p2).argmax(dim=-1)
                correct += (pred == y_t).sum().item(); ntest += y_t.numel()
                y_np = y_t.cpu().numpy(); p_np = pred.cpu().numpy()
                for c in range(NUM_CLASSES):
                    m = (y_np == c)
                    per_class_t[c] += int(m.sum())
                    per_class_c[c] += int(((p_np == c) & m).sum())
        acc_ens = correct / ntest
        recall_ens = {int(c): float(per_class_c[c] / max(1, per_class_t[c])) for c in range(NUM_CLASSES)}

        history.append({"epoch": ep, "R_t": r,
                        "train_loss_1": float(np.mean(losses1_ep)),
                        "train_loss_2": float(np.mean(losses2_ep)),
                        "test_acc_1": acc1, "test_acc_2": acc2,
                        "test_acc_ens": acc_ens, "recall_ens": recall_ens,
                        "lr": float(sched1.get_last_lr()[0])})
        if acc_ens > best_ens["test_acc"]:
            best_ens = {"epoch": ep, "test_acc": acc_ens, "recall": recall_ens}
        if ep in LOG_EPOCHS or ep == args.epochs - 1:
            per_sample_dumps1[ep] = per_sample_loss(model1, train_eval_loader, device, N)
            per_sample_dumps2[ep] = per_sample_loss(model2, train_eval_loader, device, N)
            selected1_logs[ep] = sel1
            selected2_logs[ep] = sel2
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  R(t)={r:.3f}  a={acc1*100:.2f}%  b={acc2*100:.2f}%  ens={acc_ens*100:.2f}%  "
                  f"best_ens={best_ens['test_acc']*100:.2f}%@{best_ens['epoch']}", flush=True)

    result = {
        "method": "coteaching",
        "dataset": "agnews",
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
        "log_epochs": sorted(per_sample_dumps1.keys()),
        "sampling_manifest": sm_manifest,
        "hyperparameters": {
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "arch": "TextCNN(Kim2014) x 2", "optim": "Adam",
            "weight_decay": 1e-5, "sched": "CosineAnnealingLR",
            "Tk": args.Tk, "ramp_exponent": args.ramp_exponent,
        },
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))

    eps = sorted(per_sample_dumps1.keys())
    losses1_arr = np.stack([per_sample_dumps1[e] for e in eps], axis=0)
    losses2_arr = np.stack([per_sample_dumps2[e] for e in eps], axis=0)
    sel1_arr = np.stack([selected1_logs[e] for e in eps], axis=0)
    sel2_arr = np.stack([selected2_logs[e] for e in eps], axis=0)
    R_t_arr = np.array([R_t(e) for e in eps], dtype=np.float32)
    np.savez_compressed(OUT_DIR / f"{tag}__persample.npz",
                        epochs=np.array(eps, dtype=np.int64),
                        losses1=losses1_arr, losses2=losses2_arr,
                        selected1=sel1_arr, selected2=sel2_arr,
                        R_t=R_t_arr,
                        is_noisy=is_noisy,
                        noisy_labels=y_noisy.astype(np.int64),
                        true_labels=y_true.astype(np.int64),
                        voter_ids=voter_ids.astype(np.int64))
    print(f"  DONE  best_ens={best_ens['test_acc']*100:.2f}%  saved {tag}.json + persample.npz", flush=True)


if __name__ == "__main__":
    main()
