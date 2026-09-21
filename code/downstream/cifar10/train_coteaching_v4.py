"""Co-teaching trainer (v4): same Han 2018 recipe as v3 but reads conditions
from results_v4/data_v4/ and routes images via meta['image_source'] (clean or
corrupted NLT images.npy).

Outputs:
  results_v4/coteaching/<cond>__seed<S>.json
  results_v4/coteaching/<cond>__seed<S>__persample.npz
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "results_v4" / "data_v4"
OUT_DIR = ROOT / "results_v4" / "coteaching"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_ROOT = ROOT.parent

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
LOG_EPOCHS = [5, 10, 20, 30, 50, 75, 99]


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
        self.images = images
        self.labels = labels
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
            out = model(x)
            l = F.cross_entropy(out, y, reduction='none')
            losses[idx.numpy()] = l.cpu().numpy()
    return losses


def load_train_images(meta, img_idx, nlt_pos):
    src = meta["image_source"]
    if src == "clean":
        ds = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=True, download=False)
        return ds.data[img_idx]
    elif src == "noisy":
        full = np.load(Path(meta["noisy_images_path"]))
        if full.dtype != np.uint8:
            full = (full.clip(0, 1) * 255).astype(np.uint8) if full.dtype.kind == 'f' and full.max() <= 1.5 else full.astype(np.uint8)
        return full[nlt_pos]
    else:
        raise ValueError(f"unknown image_source={src}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--Tk", type=int, default=10)
    p.add_argument("--ramp_exponent", type=float, default=0.5)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    print(f"condition={args.condition} seed={args.seed} gpu={args.gpu}")

    cd = DATA_DIR / args.condition
    img_idx = np.load(cd / "image_indices.npy")
    nlt_pos = np.load(cd / "noisylabeltrain_pos.npy")
    noisy_labels = np.load(cd / "noisy_labels.npy")
    true_labels = np.load(cd / "true_labels.npy")
    meta = json.loads((cd / "meta.json").read_text())
    sampled_noise = float(meta["sampled_noise_rate"])
    tau = sampled_noise
    is_noisy = (noisy_labels != true_labels).astype(np.uint8)

    train_images = load_train_images(meta, img_idx, nlt_pos)
    test_ds_full = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=False, download=False)
    test_images = test_ds_full.data
    test_lbl = np.array(test_ds_full.targets, dtype=np.int64)

    print(f"image_source={meta['image_source']}  N={len(train_images)}  noise_rate={sampled_noise*100:.2f}%")

    train_ds_aug = CIFARSubIdx(train_images, noisy_labels, train_aug=True)
    train_ds_eval = CIFARSubIdx(train_images, noisy_labels, train_aug=False)
    test_ds = CIFARSubIdx(test_images, test_lbl, train_aug=False)
    train_loader = DataLoader(train_ds_aug, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_eval_loader = DataLoader(train_ds_eval, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model1 = ResNet20(num_classes=10).to(device)
    model2 = ResNet20(num_classes=10).to(device)
    optim1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optim2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=args.epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=args.epochs)

    def R_t(epoch):
        ramp = min(1.0, (epoch / args.Tk) ** args.ramp_exponent)
        return 1.0 - tau * ramp

    N = len(noisy_labels)
    selected1_logs = {}; selected2_logs = {}
    persample1 = {}; persample2 = {}

    history = []
    best_acc = 0.0; best_epoch = -1
    for ep in range(args.epochs):
        model1.train(); model2.train()
        t0 = time.time()
        r = R_t(ep)

        sel1 = np.zeros(N, dtype=np.uint8)
        sel2 = np.zeros(N, dtype=np.uint8)

        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            B = y.size(0); n_keep = max(1, int(round(r * B)))

            out1 = model1(x); out2 = model2(x)
            l1 = F.cross_entropy(out1, y, reduction='none')
            l2 = F.cross_entropy(out2, y, reduction='none')
            _, idx1_keep = torch.topk(l1, n_keep, largest=False)
            _, idx2_keep = torch.topk(l2, n_keep, largest=False)

            if ep in LOG_EPOCHS:
                gidx = idx.numpy()
                sel1[gidx[idx1_keep.cpu().numpy()]] = 1
                sel2[gidx[idx2_keep.cpu().numpy()]] = 1

            optim1.zero_grad()
            loss_net1 = F.cross_entropy(out1[idx2_keep], y[idx2_keep])
            loss_net1.backward(retain_graph=True)
            optim1.step()
            optim2.zero_grad()
            loss_net2 = F.cross_entropy(out2[idx1_keep], y[idx1_keep])
            loss_net2.backward()
            optim2.step()
        sched1.step(); sched2.step()

        acc1 = evaluate(model1, test_loader, device)
        acc2 = evaluate(model2, test_loader, device)
        ens_acc = (acc1 + acc2) / 2

        log_str = ""
        if ep in LOG_EPOCHS:
            persample1[ep] = per_sample_loss(model1, train_eval_loader, device, N)
            persample2[ep] = per_sample_loss(model2, train_eval_loader, device, N)
            selected1_logs[ep] = sel1
            selected2_logs[ep] = sel2
            sel_union = (sel1 | sel2).astype(bool)
            precision = float((1 - is_noisy[sel_union]).mean()) if sel_union.sum() > 0 else 0
            recall_clean = float(sel_union[(is_noisy == 0)].mean()) if (is_noisy == 0).sum() > 0 else 0
            log_str = f"  [sel1+2: prec={precision:.3f}, recall_clean={recall_clean:.3f}]"

        history.append({"epoch": ep, "r": r, "acc1": acc1, "acc2": acc2, "ens_acc": ens_acc})
        if ens_acc > best_acc:
            best_acc = ens_acc; best_epoch = ep
        print(f"  ep {ep:3d}: R(t)={r:.3f}  acc1={acc1*100:.2f}%  acc2={acc2*100:.2f}%  ens={ens_acc*100:.2f}%  (best {best_acc*100:.2f} @ ep {best_epoch})  [{time.time()-t0:.1f}s]{log_str}")

    result = {
        "condition": args.condition, "seed": args.seed,
        "method": "coteaching",
        "image_source": meta["image_source"],
        "pool": meta.get("pool"),
        "setting": meta.get("setting"),
        "regime": meta.get("regime"),
        "n_train": int(N), "sampled_noise_rate": sampled_noise,
        "Tk": args.Tk, "ramp_exponent": args.ramp_exponent,
        "best_test_acc": best_acc, "best_epoch": best_epoch,
        "final_test_acc": history[-1]["ens_acc"],
        "best_acc1": max(h["acc1"] for h in history),
        "best_acc2": max(h["acc2"] for h in history),
        "history": history,
        "log_epochs": sorted(persample1.keys()),
    }
    out_path = OUT_DIR / f"{args.condition}__seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))

    eps = sorted(persample1.keys())
    if eps:
        losses1_arr = np.stack([persample1[e] for e in eps], axis=0)
        losses2_arr = np.stack([persample2[e] for e in eps], axis=0)
        sel1_arr = np.stack([selected1_logs[e] for e in eps], axis=0)
        sel2_arr = np.stack([selected2_logs[e] for e in eps], axis=0)
        R_t_arr = np.array([R_t(e) for e in eps], dtype=np.float32)
        np.savez_compressed(OUT_DIR / f"{args.condition}__seed{args.seed}__persample.npz",
                            epochs=np.array(eps, dtype=np.int64),
                            losses1=losses1_arr, losses2=losses2_arr,
                            selected1=sel1_arr, selected2=sel2_arr,
                            R_t=R_t_arr,
                            is_noisy=is_noisy,
                            noisy_labels=noisy_labels.astype(np.int64),
                            true_labels=true_labels.astype(np.int64))
    print(f"\nsaved {out_path}: best ens-acc {best_acc*100:.2f}% @ ep {best_epoch}")


if __name__ == "__main__":
    main()
