"""Co-Teaching trainer for MNIST (multi-modality campaign).

Two ResNet-20 networks trained in parallel with cross-selection based on the
other's small-loss samples. Standard Han et al. 2018 protocol.

Usage:
  python train_mnist_coteaching.py --setting impulse_noise_sev5 --variant S \
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
OUT_DIR = MULTI_ROOT / "results" / "mnist"
LOG_DIR = MULTI_ROOT / "logs" / "mnist"
SAMPLING_DIR = MULTI_ROOT / "sampling" / "cache" / "mnist"
for d in (OUT_DIR, LOG_DIR, SAMPLING_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MULTI_ROOT / "sampling"))
from sample_labels import sample_argmax_uniform, save_sampling_artifact  # noqa: E402

sys.path.insert(0, str(HERE))
from train_mnist_erm import (
    LeNet5, MNISTSubIdx, parse_setting, load_ccp_mask,
    load_train_images, load_test, evaluate, per_sample_loss,
    MNIST_VOTERS, NLT_ROOT, LOG_EPOCHS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--variant", choices=("S", "C"), default="S")
    ap.add_argument("--image-source", choices=("clean", "noisy"), default="noisy")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", choices=("sgd", "adam"), default="adam")
    ap.add_argument("--Tk", type=int, default=10)
    ap.add_argument("--ramp-exponent", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    tag = f"{args.setting}__{args.variant}__coteaching__{args.image_source}__seed{args.seed}"
    print(f"[MNIST CoT] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    corr, sev = parse_setting(args.setting)
    nlt_dir = NLT_ROOT / corr / f"severity_{sev}" / "noisy_label_train"
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_dir, MNIST_VOTERS, args.seed, mask=ccp_mask
    )
    tau = sm_manifest["sampled_noise_rate"]
    N = len(y_noisy)
    is_noisy = (y_noisy != y_true).astype(np.uint8)
    print(f"  N={N}  tau={tau*100:.2f}%", flush=True)
    save_sampling_artifact(SAMPLING_DIR / f"{args.setting}__{args.variant}__seed{args.seed}",
                            y_noisy, voter_ids, sm_manifest, y_true)

    images = load_train_images(args.setting, args.image_source, ccp_mask=ccp_mask)
    test_imgs, test_lbl = load_test()

    train_ds = MNISTSubIdx(images, y_noisy)
    train_eval_ds = MNISTSubIdx(images, y_noisy)
    test_ds = MNISTSubIdx(test_imgs, test_lbl)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model1 = LeNet5(num_classes=10).to(device)
    model2 = LeNet5(num_classes=10).to(device)
    if args.optim == "adam":
        optim1 = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=5e-4)
        optim2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        optim1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optim2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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
            x = x.to(device); y_t = torch.as_tensor(y, dtype=torch.long).to(device)
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
        # Ensemble
        model1.eval(); model2.eval()
        correct, ntest = 0, 0
        per_class_c = np.zeros(10, dtype=np.int64)
        per_class_t = np.zeros(10, dtype=np.int64)
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device); y_t = torch.as_tensor(y, dtype=torch.long).to(device)
                p1 = F.softmax(model1(x), dim=-1); p2 = F.softmax(model2(x), dim=-1)
                pred = (p1 + p2).argmax(dim=-1)
                correct += (pred == y_t).sum().item(); ntest += y_t.numel()
                y_np = y_t.cpu().numpy(); p_np = pred.cpu().numpy()
                for c in range(10):
                    m = (y_np == c)
                    per_class_t[c] += int(m.sum())
                    per_class_c[c] += int(((p_np == c) & m).sum())
        acc_ens = correct / ntest
        recall_ens = {int(c): float(per_class_c[c] / max(1, per_class_t[c])) for c in range(10)}

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
        "dataset": "mnist",
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
            "arch": "LeNet5 x 2", "optim": args.optim.upper(),
            "weight_decay": 5e-4, "sched": "CosineAnnealingLR",
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
