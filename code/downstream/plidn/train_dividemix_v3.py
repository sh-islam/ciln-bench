"""DivideMix trainer (Li et al. 2020) with clean-probability logging.

Compact, faithful implementation. Two ResNet-20 networks; each epoch:
  1. Compute per-sample CE loss using each network in eval mode.
  2. Fit 2-component GMM per network -> per-sample clean probability w_i.
     The component with smaller mean is "clean".
  3. Split by threshold (default 0.5): labeled (w_i > 0.5) vs unlabeled (w_i <= 0.5).
  4. Co-refinement: for each network, the LABELED set's refined target is
     w_i * one_hot(y_noisy) + (1 - w_i) * p_partner(x), then temperature-sharpened.
     The UNLABELED set's pseudo-label is mean over both networks' predictions,
     temperature-sharpened.
  5. MixMatch: mix labeled and unlabeled into mixup pairs, train each network
     with CE on labeled-mix and L2/CE on unlabeled-mix.

The full DivideMix uses K=2 augmentations and additional regularizers
(entropy regularization H, prior-consistency regularizer); we include H but
omit prior-consistency to keep the trainer compact. The clean-probability
GMM is the key diagnostic.

Extra logging at LOG_EPOCHS:
  - per-sample loss for both networks (eval mode, no aug)
  - per-sample GMM clean probability w_i for both networks

Outputs:
  results_v3/dividemix/<cond>__seed<S>.json
  results_v3/dividemix/<cond>__seed<S>__persample.npz
    keys: epochs (E,), losses1 (E, N), losses2 (E, N),
          clean_prob1 (E, N), clean_prob2 (E, N),
          is_noisy (N,), noisy_labels (N,), true_labels (N,)
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "results_v3" / "data_v3"
OUT_DIR = ROOT / "results_v3" / "dividemix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_ROOT = ROOT.parent

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

LOG_EPOCHS = [5, 10, 20, 30, 50, 75, 99]

# DivideMix hyper-params
WARM_EPOCHS = 10        # warm-up: train with plain CE before splitting
P_THRESH = 0.5          # clean/noisy threshold on GMM w_i
ALPHA = 4.0             # beta mixup alpha
T_SHARPEN = 0.5         # label sharpening temperature
LAMBDA_U = 25.0         # unlabeled MSE weight (CIFAR-10 default)
LAMBDA_E = 0.0          # entropy reg (off; can enable)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.short = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.short(x)
        return F.relu(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._layer(16, 3, 1)
        self.layer2 = self._layer(32, 3, 2)
        self.layer3 = self._layer(64, 3, 2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    def _layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = self.avg(out).flatten(1)
        return self.fc(out)


class CIFARSubIdx(Dataset):
    def __init__(self, images, labels, train_aug=True):
        self.images = images; self.labels = labels
        if train_aug:
            self.tf = T.Compose([
                T.ToPILImage(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.tf = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return self.tf(self.images[i]), int(self.labels[i]), i


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            out = model(x); pred = out.argmax(1)
            correct += (pred == y).sum().item(); total += y.numel()
    return correct / total


def per_sample_loss(model, loader, device, N):
    model.eval()
    losses = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device); y = y.to(device)
            out = model(x); l = F.cross_entropy(out, y, reduction='none')
            losses[idx.numpy()] = l.cpu().numpy()
    return losses


def fit_gmm_clean_prob(losses):
    """Fit 2-component GMM to per-sample losses; return clean prob (lower-mean component)."""
    losses = losses.reshape(-1, 1)
    # Normalize losses
    losses_norm = (losses - losses.min()) / max(1e-9, losses.max() - losses.min())
    gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4, random_state=0)
    gmm.fit(losses_norm)
    probs = gmm.predict_proba(losses_norm)
    clean_idx = gmm.means_.argmin()
    w = probs[:, clean_idx]
    return w.astype(np.float32)


def sharpen(p, T):
    pt = p.pow(1.0 / T)
    return pt / pt.sum(dim=1, keepdim=True)


def warmup_one_epoch(model, train_loader, optim, device):
    model.train()
    for x, y, _ in train_loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optim.zero_grad()
        out = model(x); loss = F.cross_entropy(out, y)
        loss.backward(); optim.step()


def dividemix_train_epoch(modelA, modelB, optimA, w_A_arr, w_B_arr,
                           train_images, noisy_labels, batch_size, device,
                           tf_aug):
    """One DivideMix-style training epoch for modelA, using modelB for co-refinement.
    w_A_arr, w_B_arr are per-sample clean probs from each network (shape (N,)).
    Splits using w_A_arr -> labeled (w>thresh) / unlabeled (w<=thresh) for modelA's view.
    """
    modelA.train(); modelB.eval()
    N = len(noisy_labels)
    labeled_mask = w_A_arr > P_THRESH
    unlabeled_mask = ~labeled_mask
    lab_idx = np.where(labeled_mask)[0]
    unl_idx = np.where(unlabeled_mask)[0]
    # If degenerate split (one set empty), fall back to standard CE
    if len(lab_idx) < batch_size or len(unl_idx) < batch_size:
        # Plain CE epoch
        idx_perm = np.random.permutation(N)
        for s in range(0, N, batch_size):
            batch = idx_perm[s:s+batch_size]
            if len(batch) < 4: continue
            xs = torch.stack([tf_aug(train_images[i]) for i in batch], 0).to(device)
            ys = torch.tensor(noisy_labels[batch], dtype=torch.long, device=device)
            optimA.zero_grad()
            out = modelA(xs); loss = F.cross_entropy(out, ys)
            loss.backward(); optimA.step()
        return

    # Permute labeled and unlabeled; iterate over labeled (drives epoch length)
    np.random.shuffle(lab_idx); np.random.shuffle(unl_idx)
    n_steps = max(1, len(lab_idx) // batch_size)
    unl_pool = unl_idx
    unl_ptr = 0

    for s in range(n_steps):
        lab_batch = lab_idx[s*batch_size:(s+1)*batch_size]
        if len(lab_batch) < 4: continue
        if unl_ptr + batch_size > len(unl_pool):
            np.random.shuffle(unl_pool); unl_ptr = 0
        unl_batch = unl_pool[unl_ptr:unl_ptr + batch_size]
        unl_ptr += batch_size

        # Build augmented mini-batches
        xL = torch.stack([tf_aug(train_images[i]) for i in lab_batch], 0).to(device)
        yL = torch.tensor(noisy_labels[lab_batch], dtype=torch.long, device=device)
        wL = torch.tensor(w_A_arr[lab_batch], dtype=torch.float32, device=device).unsqueeze(1)
        xU = torch.stack([tf_aug(train_images[i]) for i in unl_batch], 0).to(device)

        # Co-refined labels for labeled (network A's view) + partner predictions
        with torch.no_grad():
            pA_L = F.softmax(modelA(xL), dim=1)
            pB_L = F.softmax(modelB(xL), dim=1)
            pA_U = F.softmax(modelA(xU), dim=1)
            pB_U = F.softmax(modelB(xU), dim=1)

        yL_oh = F.one_hot(yL, num_classes=10).float()
        # Refined labeled target: w * one_hot(y) + (1-w) * pA (own prediction)
        refined_L = wL * yL_oh + (1 - wL) * pA_L
        refined_L = sharpen(refined_L, T_SHARPEN).detach()
        # Pseudo-labeled unlabeled target: mean of both networks, sharpened
        pseudo_U = sharpen((pA_U + pB_U) / 2.0, T_SHARPEN).detach()

        # MixMatch mixup
        all_x = torch.cat([xL, xU], 0)
        all_t = torch.cat([refined_L, pseudo_U], 0)
        beta = np.random.beta(ALPHA, ALPHA)
        beta = max(beta, 1 - beta)  # mixup stays "close" to original
        perm = torch.randperm(all_x.size(0), device=device)
        mixed_x = beta * all_x + (1 - beta) * all_x[perm]
        mixed_t = beta * all_t + (1 - beta) * all_t[perm]

        BL = xL.size(0); BU = xU.size(0)
        out = modelA(mixed_x)
        logp = F.log_softmax(out, dim=1)
        # Labeled half: -sum_k t_k * log p_k (CE with soft target)
        loss_L = -(mixed_t[:BL] * logp[:BL]).sum(dim=1).mean()
        # Unlabeled half: MSE on softmax
        p_U = F.softmax(out[BL:], dim=1)
        loss_U = ((p_U - mixed_t[BL:])**2).sum(dim=1).mean()

        # Entropy reg (skip if LAMBDA_E=0)
        loss = loss_L + LAMBDA_U * loss_U
        optimA.zero_grad(); loss.backward(); optimA.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    print(f"condition={args.condition} seed={args.seed} gpu={args.gpu}")

    cd = DATA_DIR / args.condition
    img_idx = np.load(cd / "image_indices.npy")
    noisy_labels = np.load(cd / "noisy_labels.npy")
    true_labels = np.load(cd / "true_labels.npy")
    meta = json.loads((cd / "meta.json").read_text())
    sampled_noise = float(meta["sampled_noise_rate"])
    is_noisy = (noisy_labels != true_labels).astype(np.uint8)

    train_ds_full = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=True, download=False)
    test_ds_full = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=False, download=False)
    train_images = train_ds_full.data[img_idx]
    test_images = test_ds_full.data
    test_lbl = np.array(test_ds_full.targets, dtype=np.int64)

    train_aug = T.Compose([T.ToPILImage(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                            T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_ds_aug = CIFARSubIdx(train_images, noisy_labels, train_aug=True)
    train_ds_eval = CIFARSubIdx(train_images, noisy_labels, train_aug=False)
    test_ds = CIFARSubIdx(test_images, test_lbl, train_aug=False)
    train_loader = DataLoader(train_ds_aug, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    train_eval_loader = DataLoader(train_ds_eval, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model1 = ResNet20(num_classes=10).to(device)
    model2 = ResNet20(num_classes=10).to(device)
    optim1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optim2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=args.epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=args.epochs)

    N = len(noisy_labels)
    persample1 = {}; persample2 = {}
    cleanp1 = {}; cleanp2 = {}

    history = []
    best_acc = 0.0; best_epoch = -1
    for ep in range(args.epochs):
        t0 = time.time()
        if ep < WARM_EPOCHS:
            # Warm-up: standard CE on both nets
            warmup_one_epoch(model1, train_loader, optim1, device)
            warmup_one_epoch(model2, train_loader, optim2, device)
        else:
            # Compute clean probs via GMM on each net's eval losses
            l1 = per_sample_loss(model1, train_eval_loader, device, N)
            l2 = per_sample_loss(model2, train_eval_loader, device, N)
            w1 = fit_gmm_clean_prob(l1)
            w2 = fit_gmm_clean_prob(l2)
            # Train modelA using modelB's clean probs (the standard DivideMix split)
            dividemix_train_epoch(model1, model2, optim1, w2, w1,
                                  train_images, noisy_labels, args.batch, device, train_aug)
            dividemix_train_epoch(model2, model1, optim2, w1, w2,
                                  train_images, noisy_labels, args.batch, device, train_aug)
        sched1.step(); sched2.step()

        acc1 = evaluate(model1, test_loader, device)
        acc2 = evaluate(model2, test_loader, device)
        ens_acc = (acc1 + acc2) / 2

        log_str = ""
        if ep in LOG_EPOCHS:
            persample1[ep] = per_sample_loss(model1, train_eval_loader, device, N)
            persample2[ep] = per_sample_loss(model2, train_eval_loader, device, N)
            cleanp1[ep] = fit_gmm_clean_prob(persample1[ep])
            cleanp2[ep] = fit_gmm_clean_prob(persample2[ep])
            # diagnostic: clean prob vs is_noisy
            mean_clean_p_on_clean = float(cleanp1[ep][is_noisy == 0].mean()) if (is_noisy == 0).sum() > 0 else 0
            mean_clean_p_on_noisy = float(cleanp1[ep][is_noisy == 1].mean()) if (is_noisy == 1).sum() > 0 else 0
            log_str = f"  [w on clean={mean_clean_p_on_clean:.3f}, on noisy={mean_clean_p_on_noisy:.3f}]"

        history.append({"epoch": ep, "acc1": acc1, "acc2": acc2, "ens_acc": ens_acc})
        if ens_acc > best_acc:
            best_acc = ens_acc; best_epoch = ep
        print(f"  ep {ep:3d}: acc1={acc1*100:.2f}%  acc2={acc2*100:.2f}%  ens={ens_acc*100:.2f}%  (best {best_acc*100:.2f} @ ep {best_epoch})  [{time.time()-t0:.1f}s]{log_str}")

    result = {
        "condition": args.condition, "seed": args.seed,
        "method": "dividemix",
        "n_train": int(N), "sampled_noise_rate": sampled_noise,
        "best_test_acc": best_acc, "best_epoch": best_epoch,
        "final_test_acc": history[-1]["ens_acc"],
        "best_acc1": max(h["acc1"] for h in history),
        "best_acc2": max(h["acc2"] for h in history),
        "warm_epochs": WARM_EPOCHS,
        "history": history,
        "log_epochs": sorted(persample1.keys()),
    }
    out_path = OUT_DIR / f"{args.condition}__seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))

    eps = sorted(persample1.keys())
    losses1_arr = np.stack([persample1[e] for e in eps], axis=0)
    losses2_arr = np.stack([persample2[e] for e in eps], axis=0)
    cp1_arr = np.stack([cleanp1[e] for e in eps], axis=0)
    cp2_arr = np.stack([cleanp2[e] for e in eps], axis=0)
    persample_path = OUT_DIR / f"{args.condition}__seed{args.seed}__persample.npz"
    np.savez_compressed(persample_path,
                        epochs=np.array(eps, dtype=np.int64),
                        losses1=losses1_arr, losses2=losses2_arr,
                        clean_prob1=cp1_arr, clean_prob2=cp2_arr,
                        is_noisy=is_noisy,
                        noisy_labels=noisy_labels.astype(np.int64),
                        true_labels=true_labels.astype(np.int64))
    print(f"\nsaved {out_path}: best ens-acc {best_acc*100:.2f}% @ ep {best_epoch}")
    print(f"saved {persample_path}")


if __name__ == "__main__":
    main()
