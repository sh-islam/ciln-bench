"""ERM trainer for MNIST downstream evaluation (multi-modality campaign).

Mirrors CIFAR's train_coteaching_v3.py (ERM path) but for MNIST:
  - 28x28 grayscale inputs (single channel)
  - ResNet-20 with first conv adapted to 1 channel
  - Argmax-uniform voter sampling via shared module
  - CILN-S / CILN-C variant support
  - Full logging: per-class recall, per-sample losses, sampling manifest

Usage:
  python train_mnist_erm.py --setting impulse_noise_sev5 --variant S \
      --image-source noisy --seed 0 --gpu 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
MULTI_ROOT = HERE.parent
OUT_DIR = MULTI_ROOT / "results" / "mnist"
LOG_DIR = MULTI_ROOT / "logs" / "mnist"
SAMPLING_DIR = MULTI_ROOT / "sampling" / "cache" / "mnist"
for d in (OUT_DIR, LOG_DIR, SAMPLING_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MULTI_ROOT / "sampling"))
from sample_labels import sample_argmax_uniform, save_sampling_artifact  # noqa: E402

JOURNAL_ROOT = MULTI_ROOT.parent.parent / "journal_edition"
NLT_ROOT = JOURNAL_ROOT / "output_seed0" / "mnist"
CLEAN_ROOT = JOURNAL_ROOT / "output_seed0" / "clean" / "mnist" / "noisylabeltrain_clean"
SPLITS_DIR = JOURNAL_ROOT / "output_seed0" / "splits" / "mnist"

MNIST_VOTERS = ("lenet5", "mlp", "resnet20", "deit3_small")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

LOG_EPOCHS = [5, 10, 20, 30, 50, 75, 99]


class LeNet5(nn.Module):
    """Classic LeNet-5 for MNIST: conv1 -> pool -> conv2 -> pool -> fc*3.
    Standard MNIST baseline architecture; also matches one of the MNIST voter
    architectures (voter and learner do not share training data or weights)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)   # 28x28 -> 28x28 (padded)
        self.pool1 = nn.MaxPool2d(2, 2)               # -> 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)              # -> 10x10
        self.pool2 = nn.MaxPool2d(2, 2)               # -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MNISTSubIdx(Dataset):
    """MNIST wrapper: uint8 (N,28,28) or (N,1,28,28); labels; returns (img,y,idx)."""
    def __init__(self, images, labels):
        if images.ndim == 3:
            images = images[:, None, :, :]  # add channel
        self.images = images.astype(np.float32) / 255.0
        # Normalize
        self.images = (self.images - MNIST_MEAN) / MNIST_STD
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return torch.from_numpy(self.images[i]), int(self.labels[i]), i


def parse_setting(setting: str):
    parts = setting.rsplit("_sev", 1)
    return parts[0], int(parts[1])


def load_ccp_mask():
    m = np.load(CLEAN_ROOT / "clean_correct_mask.npy")
    return m.astype(bool)


def load_train_images(setting: str, image_source: str, ccp_mask=None):
    corr, sev = parse_setting(setting)
    setting_dir = NLT_ROOT / corr / f"severity_{sev}" / "noisy_label_train"
    if image_source == "noisy":
        images = np.load(setting_dir / "images.npy")
    elif image_source == "clean":
        # Load the CLEAN version of the same NLT rows.
        # Use the raw MNIST training split via NLT indices.
        import torchvision
        ds = torchvision.datasets.MNIST(
            root=str(MULTI_ROOT / "data_cache"), train=True, download=True,
        )
        nlt_idx = np.load(SPLITS_DIR / "noisylabeltrain_indices.npy")
        all_imgs = ds.data.numpy()  # (60000, 28, 28) uint8
        images = all_imgs[nlt_idx]
    else:
        raise ValueError(image_source)
    if ccp_mask is not None:
        images = images[ccp_mask]
    return images


def load_test():
    import torchvision
    ds = torchvision.datasets.MNIST(
        root=str(MULTI_ROOT / "data_cache"), train=False, download=True,
    )
    return ds.data.numpy(), ds.targets.numpy().astype(np.int64)


def evaluate(model, loader, device, num_classes=10):
    model.eval()
    correct = total = 0
    per_class_c = np.zeros(num_classes, dtype=np.int64)
    per_class_t = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y_t = torch.as_tensor(y, dtype=torch.long).to(device)
            out = model(x); pred = out.argmax(1)
            correct += (pred == y_t).sum().item(); total += y_t.numel()
            y_np = y_t.cpu().numpy(); p_np = pred.cpu().numpy()
            for c in range(num_classes):
                m = (y_np == c)
                per_class_t[c] += int(m.sum())
                per_class_c[c] += int(((p_np == c) & m).sum())
    recall = {int(c): float(per_class_c[c] / max(1, per_class_t[c])) for c in range(num_classes)}
    return correct / total, recall


def per_sample_loss(model, loader, device, N):
    model.eval()
    out = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device); y_t = torch.as_tensor(y, dtype=torch.long).to(device)
            l = F.cross_entropy(model(x), y_t, reduction='none')
            out[idx.numpy()] = l.cpu().numpy()
    return out


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
    tag = f"{args.setting}__{args.variant}__erm__{args.image_source}__seed{args.seed}"
    print(f"[MNIST ERM] {tag}  gpu={args.gpu}", flush=True)
    t0 = time.time()

    corr, sev = parse_setting(args.setting)
    nlt_dir = NLT_ROOT / corr / f"severity_{sev}" / "noisy_label_train"
    ccp_mask = load_ccp_mask() if args.variant == "C" else None
    y_noisy, voter_ids, sm_manifest, y_true = sample_argmax_uniform(
        nlt_dir, MNIST_VOTERS, args.seed, mask=ccp_mask
    )
    print(f"  N={sm_manifest['n_rows']}  sampled_nr={sm_manifest['sampled_noise_rate']*100:.2f}%  "
          f"expected_nr={sm_manifest['expected_noise_rate']*100:.2f}%", flush=True)
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

    model = LeNet5(num_classes=10).to(device)
    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    is_noisy = (y_noisy != y_true).astype(np.uint8)
    N = len(y_noisy)

    history = []
    per_sample_dumps = {}
    best = {"epoch": -1, "test_acc": -1.0, "recall": None}
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(device); y_t = torch.as_tensor(y, dtype=torch.long).to(device)
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
        "dataset": "mnist",
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
            "arch": "LeNet5", "optim": args.optim.upper(),
            "weight_decay": 5e-4, "sched": "CosineAnnealingLR",
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
