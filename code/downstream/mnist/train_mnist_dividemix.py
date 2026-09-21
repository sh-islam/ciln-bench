"""DivideMix trainer for MNIST (multi-modality campaign).

Faithful adaptation of the CIFAR DM v4 trainer for MNIST:
  - ResNet-20 with 1-channel first conv
  - Warmup (10 epochs) with standard CE
  - Then GMM-based split + MixMatch with co-refinement
  - No augmentation on MNIST (single ToTensor + normalize)

Usage:
  python train_mnist_dividemix.py --setting impulse_noise_sev5 --variant S \
      --image-source noisy --seed 0 --gpu 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

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
    MNIST_VOTERS, NLT_ROOT, LOG_EPOCHS, MNIST_MEAN, MNIST_STD,
)

# DivideMix hyperparameters
WARM_EPOCHS = 10
P_THRESH = 0.5
ALPHA = 4.0
T_SHARPEN = 0.5
LAMBDA_U = 25.0


def fit_gmm_clean_prob(losses):
    losses = losses.reshape(-1, 1)
    lo, hi = losses.min(), losses.max()
    if hi - lo < 1e-9:
        return np.zeros_like(losses[:, 0], dtype=np.float32)
    losses_norm = (losses - lo) / (hi - lo)
    gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4, random_state=0)
    gmm.fit(losses_norm)
    probs = gmm.predict_proba(losses_norm)
    clean_idx = gmm.means_.argmin()
    return probs[:, clean_idx].astype(np.float32)


def sharpen(p, T):
    pt = p.pow(1.0 / T)
    return pt / pt.sum(dim=1, keepdim=True)


def warmup_one_epoch(model, train_loader, optim, device):
    model.train()
    for x, y, _ in train_loader:
        x = x.to(device); y_t = torch.as_tensor(y, dtype=torch.long).to(device)
        optim.zero_grad()
        out = model(x); loss = F.cross_entropy(out, y_t)
        loss.backward(); optim.step()


def normalize_np_img(img_uint8):
    """(N,28,28) uint8 or (N,1,28,28) -> normalized float32 tensor."""
    if img_uint8.ndim == 3:
        img_uint8 = img_uint8[:, None, :, :]
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - MNIST_MEAN) / MNIST_STD
    return x  # (N, 1, 28, 28)


def dividemix_train_epoch(modelA, modelB, optimA, w_A_arr, train_images_norm,
                           noisy_labels, batch_size, device):
    """One DivideMix epoch for modelA using modelB for co-refinement.
    train_images_norm: pre-normalized (N,1,28,28) float32.
    """
    modelA.train(); modelB.eval()
    N = len(noisy_labels)
    labeled_mask = w_A_arr > P_THRESH
    unlabeled_mask = ~labeled_mask
    lab_idx = np.where(labeled_mask)[0]
    unl_idx = np.where(unlabeled_mask)[0]

    if len(lab_idx) < batch_size or len(unl_idx) < batch_size:
        # Fallback to plain CE
        idx_perm = np.random.permutation(N)
        for s in range(0, N, batch_size):
            batch = idx_perm[s:s+batch_size]
            if len(batch) < 4: continue
            xs = torch.from_numpy(train_images_norm[batch]).to(device)
            ys = torch.tensor(noisy_labels[batch], dtype=torch.long, device=device)
            optimA.zero_grad()
            out = modelA(xs); loss = F.cross_entropy(out, ys)
            loss.backward(); optimA.step()
        return

    np.random.shuffle(lab_idx); np.random.shuffle(unl_idx)
    n_steps = max(1, len(lab_idx) // batch_size)
    unl_ptr = 0

    for s in range(n_steps):
        lab_batch = lab_idx[s*batch_size:(s+1)*batch_size]
        if len(lab_batch) < 4: continue
        if unl_ptr + batch_size > len(unl_idx):
            np.random.shuffle(unl_idx); unl_ptr = 0
        unl_batch = unl_idx[unl_ptr:unl_ptr + batch_size]
        unl_ptr += batch_size

        xL = torch.from_numpy(train_images_norm[lab_batch]).to(device)
        yL = torch.tensor(noisy_labels[lab_batch], dtype=torch.long, device=device)
        wL = torch.tensor(w_A_arr[lab_batch], dtype=torch.float32, device=device).unsqueeze(1)
        xU = torch.from_numpy(train_images_norm[unl_batch]).to(device)

        with torch.no_grad():
            pA_L = F.softmax(modelA(xL), dim=1)
            pB_L = F.softmax(modelB(xL), dim=1)
            pA_U = F.softmax(modelA(xU), dim=1)
            pB_U = F.softmax(modelB(xU), dim=1)

        yL_oh = F.one_hot(yL, num_classes=10).float()
        refined_L = wL * yL_oh + (1 - wL) * pA_L
        refined_L = sharpen(refined_L, T_SHARPEN).detach()
        pseudo_U = sharpen((pA_U + pB_U) / 2.0, T_SHARPEN).detach()

        all_x = torch.cat([xL, xU], 0)
        all_t = torch.cat([refined_L, pseudo_U], 0)
        beta = np.random.beta(ALPHA, ALPHA)
        beta = max(beta, 1 - beta)
        perm = torch.randperm(all_x.size(0), device=device)
        mixed_x = beta * all_x + (1 - beta) * all_x[perm]
        mixed_t = beta * all_t + (1 - beta) * all_t[perm]

        BL = xL.size(0)
        out = modelA(mixed_x)
        logp = F.log_softmax(out, dim=1)
        loss_L = -(mixed_t[:BL] * logp[:BL]).sum(dim=1).mean()
        p_U = F.softmax(out[BL:], dim=1)
        loss_U = ((p_U - mixed_t[BL:])**2).sum(dim=1).mean()
        loss = loss_L + LAMBDA_U * loss_U
        optimA.zero_grad(); loss.backward(); optimA.step()


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
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    tag = f"{args.setting}__{args.variant}__dividemix__{args.image_source}__seed{args.seed}"
    print(f"[MNIST DM] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    corr, sev = parse_setting(args.setting)
    nlt_dir = NLT_ROOT / corr / f"severity_{sev}" / "noisy_label_train"
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_dir, MNIST_VOTERS, args.seed, mask=ccp_mask
    )
    N = len(y_noisy)
    is_noisy = (y_noisy != y_true).astype(np.uint8)
    print(f"  N={N}  sampled_nr={sm_manifest['sampled_noise_rate']*100:.2f}%", flush=True)
    save_sampling_artifact(SAMPLING_DIR / f"{args.setting}__{args.variant}__seed{args.seed}",
                            y_noisy, voter_ids, sm_manifest, y_true)

    images = load_train_images(args.setting, args.image_source, ccp_mask=ccp_mask)
    train_images_norm = normalize_np_img(images)  # (N, 1, 28, 28) float32
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

    history = []
    per_sample_dumps1 = {}; per_sample_dumps2 = {}
    clean_prob1 = {}; clean_prob2 = {}
    best_ens = {"epoch": -1, "test_acc": -1.0, "recall": None}

    for ep in range(args.epochs):
        ep_t0 = time.time()
        if ep < WARM_EPOCHS:
            warmup_one_epoch(model1, train_loader, optim1, device)
            warmup_one_epoch(model2, train_loader, optim2, device)
        else:
            l1 = per_sample_loss(model1, train_eval_loader, device, N)
            l2 = per_sample_loss(model2, train_eval_loader, device, N)
            w1 = fit_gmm_clean_prob(l1)
            w2 = fit_gmm_clean_prob(l2)
            dividemix_train_epoch(model1, model2, optim1, w2, train_images_norm, y_noisy, args.batch, device)
            dividemix_train_epoch(model2, model1, optim2, w1, train_images_norm, y_noisy, args.batch, device)
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

        history.append({"epoch": ep, "test_acc_1": acc1, "test_acc_2": acc2,
                        "test_acc_ens": acc_ens, "recall_ens": recall_ens,
                        "lr": float(sched1.get_last_lr()[0])})
        if acc_ens > best_ens["test_acc"]:
            best_ens = {"epoch": ep, "test_acc": acc_ens, "recall": recall_ens}
        if ep in LOG_EPOCHS or ep == args.epochs - 1:
            per_sample_dumps1[ep] = per_sample_loss(model1, train_eval_loader, device, N)
            per_sample_dumps2[ep] = per_sample_loss(model2, train_eval_loader, device, N)
            clean_prob1[ep] = fit_gmm_clean_prob(per_sample_dumps1[ep])
            clean_prob2[ep] = fit_gmm_clean_prob(per_sample_dumps2[ep])
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  a={acc1*100:.2f}%  b={acc2*100:.2f}%  ens={acc_ens*100:.2f}%  "
                  f"best_ens={best_ens['test_acc']*100:.2f}%@{best_ens['epoch']}  [{time.time()-ep_t0:.1f}s]",
                  flush=True)

    result = {
        "method": "dividemix",
        "dataset": "mnist",
        "setting": args.setting,
        "variant": args.variant,
        "image_source": args.image_source,
        "seed": args.seed,
        "n_train": int(N),
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
            "warm_epochs": WARM_EPOCHS, "p_thresh": P_THRESH,
            "alpha": ALPHA, "T_sharpen": T_SHARPEN, "lambda_u": LAMBDA_U,
        },
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(result, indent=2))

    eps = sorted(per_sample_dumps1.keys())
    if eps:
        losses1_arr = np.stack([per_sample_dumps1[e] for e in eps], axis=0)
        losses2_arr = np.stack([per_sample_dumps2[e] for e in eps], axis=0)
        cp1_arr = np.stack([clean_prob1[e] for e in eps], axis=0)
        cp2_arr = np.stack([clean_prob2[e] for e in eps], axis=0)
        np.savez_compressed(OUT_DIR / f"{tag}__persample.npz",
                            epochs=np.array(eps, dtype=np.int64),
                            losses1=losses1_arr, losses2=losses2_arr,
                            clean_prob1=cp1_arr, clean_prob2=cp2_arr,
                            is_noisy=is_noisy,
                            noisy_labels=y_noisy.astype(np.int64),
                            true_labels=y_true.astype(np.int64),
                            voter_ids=voter_ids.astype(np.int64))
    print(f"  DONE  best_ens={best_ens['test_acc']*100:.2f}%  saved {tag}.json + persample.npz", flush=True)


if __name__ == "__main__":
    main()
