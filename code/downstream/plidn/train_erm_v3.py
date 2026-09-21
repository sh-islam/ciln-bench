"""ERM trainer with per-sample loss logging.

Trains ResNet-20 on (clean CIFAR-10 train images, sampled noisy labels) for
one (condition, seed). Logs per-sample training loss (eval-mode, no
augmentation, no grad) at a chosen set of epochs so we can later analyse
loss bimodality (clean vs noisy).

Outputs:
  results_v3/erm/<cond>__seed<S>.json    -- per-epoch test acc, etc.
  results_v3/erm/<cond>__seed<S>__persample.npz  -- per-sample loss at logged epochs
    keys: epochs (E,), losses (E, N), is_noisy (N,), noisy_labels (N,), true_labels (N,)

Usage:
  python train_erm_v3.py --condition <name> --seed <int> [--gpu N] [--epochs 100]
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
DATA_DIR = ROOT / "results_v3" / "data_v3"
OUT_DIR = ROOT / "results_v3" / "erm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
JOURNAL_ROOT = ROOT.parent

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Epochs at which to log per-sample loss (warm-up + mid + late)
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
    """Yields (image, label, index) so we can identify samples."""
    def __init__(self, images, labels, train_aug=True):
        self.images = images
        self.labels = labels
        if train_aug:
            self.tf = T.Compose([
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.tf = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return self.tf(self.images[i]), int(self.labels[i]), i


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def per_sample_loss(model, eval_loader, device, N):
    """Return per-sample CE loss as (N,) array, computed in eval mode."""
    model.eval()
    losses = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for x, y, idx in eval_loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            l = F.cross_entropy(out, y, reduction='none')
            losses[idx.numpy()] = l.cpu().numpy()
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
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
    is_noisy = (noisy_labels != true_labels).astype(np.uint8)

    train_ds_full = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=True, download=False)
    test_ds_full = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=False, download=False)
    train_images = train_ds_full.data[img_idx]
    test_images = test_ds_full.data
    test_lbl = np.array(test_ds_full.targets, dtype=np.int64)

    train_ds_aug = CIFARSubIdx(train_images, noisy_labels, train_aug=True)
    train_ds_eval = CIFARSubIdx(train_images, noisy_labels, train_aug=False)  # for per-sample loss
    test_ds = CIFARSubIdx(test_images, test_lbl, train_aug=False)
    train_loader = DataLoader(train_ds_aug, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    train_eval_loader = DataLoader(train_ds_eval, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet20(num_classes=10).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    N = len(noisy_labels)
    persample = {}  # epoch -> (N,)

    best_acc = 0.0; best_epoch = -1
    history = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = n = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optim.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward(); optim.step()
            loss_sum += loss.item() * y.numel(); n += y.numel()
        sched.step()
        test_acc = evaluate(model, test_loader, device)
        history.append({"epoch": ep, "loss": loss_sum / n, "test_acc": test_acc})
        if test_acc > best_acc:
            best_acc = test_acc; best_epoch = ep

        log_str = ""
        if ep in LOG_EPOCHS:
            persample[ep] = per_sample_loss(model, train_eval_loader, device, N)
            log_str = f"  [logged per-sample loss: mean={persample[ep].mean():.3f}]"

        print(f"  ep {ep:3d}: loss={loss_sum/n:.4f}  test_acc={test_acc*100:.2f}%  (best {best_acc*100:.2f} @ ep {best_epoch})  [{time.time()-t0:.1f}s]{log_str}")

    result = {
        "condition": args.condition, "seed": args.seed,
        "n_train": int(len(noisy_labels)),
        "sampled_noise_rate": meta["sampled_noise_rate"],
        "best_test_acc": best_acc, "best_epoch": best_epoch,
        "final_test_acc": history[-1]["test_acc"],
        "history": history,
        "log_epochs": sorted(persample.keys()),
    }
    out_path = OUT_DIR / f"{args.condition}__seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))

    # Save per-sample logs
    eps = sorted(persample.keys())
    losses_arr = np.stack([persample[e] for e in eps], axis=0)  # (E, N)
    persample_path = OUT_DIR / f"{args.condition}__seed{args.seed}__persample.npz"
    np.savez_compressed(persample_path,
                        epochs=np.array(eps, dtype=np.int64),
                        losses=losses_arr,
                        is_noisy=is_noisy,
                        noisy_labels=noisy_labels.astype(np.int64),
                        true_labels=true_labels.astype(np.int64))
    print(f"\nsaved {out_path}: best {best_acc*100:.2f}% @ ep {best_epoch}")
    print(f"saved {persample_path}: persample (E={len(eps)}, N={N})")


if __name__ == "__main__":
    main()
