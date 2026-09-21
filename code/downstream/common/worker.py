"""Queue worker for multi-modality downstream runs.

Reads a manifest CSV, pulls the next pending row, marks it running, executes
the trainer, marks it done (or failed). Uses a lock file to coordinate two
workers running against the same manifest.

Usage:
  python worker.py --manifest adult_grid.csv --gpu 1
"""
from __future__ import annotations
import argparse, csv, fcntl, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
MULTI_ROOT = HERE.parent
MANIFEST_DIR = MULTI_ROOT / "manifest"
LOG_DIR = MULTI_ROOT / "logs"

# Modality -> (script path, extra_args_dict_fn)
ADULT_ERM = MULTI_ROOT.parent / "downstream_adult" / "train" / "train_adult_erm_v2.py"
ADULT_COT = MULTI_ROOT.parent / "downstream_adult" / "train" / "train_adult_coteaching_v2.py"
MNIST_ERM = MULTI_ROOT / "train_mnist" / "train_mnist_erm.py"
MNIST_COT = MULTI_ROOT / "train_mnist" / "train_mnist_coteaching.py"
MNIST_DM  = MULTI_ROOT / "train_mnist" / "train_mnist_dividemix.py"
AGN_ERM = MULTI_ROOT / "train_agnews" / "train_agnews_erm.py"
AGN_COT = MULTI_ROOT / "train_agnews" / "train_agnews_coteaching.py"


def script_for(dataset: str, method: str) -> Path:
    if dataset == "adult":
        if method == "erm": return ADULT_ERM
        if method == "coteaching": return ADULT_COT
    if dataset == "mnist":
        if method == "erm": return MNIST_ERM
        if method == "coteaching": return MNIST_COT
        if method == "dividemix": return MNIST_DM
    if dataset == "agnews":
        if method == "erm": return AGN_ERM
        if method == "coteaching": return AGN_COT
    raise ValueError(f"unknown ({dataset}, {method})")


def read_manifest(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows):
    fields = ["status", "dataset", "setting", "variant", "method", "image_source", "seed"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def claim_next_pending(manifest_path: Path):
    """Atomically claim the next pending row; return the claimed row dict or None."""
    lock_path = manifest_path.with_suffix(".lock")
    with lock_path.open("w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        rows = read_manifest(manifest_path)
        for r in rows:
            if r["status"] == "pending":
                r["status"] = "running"
                write_manifest(manifest_path, rows)
                return r
        return None  # nothing pending


def mark_done(manifest_path: Path, row, status: str):
    lock_path = manifest_path.with_suffix(".lock")
    with lock_path.open("w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        rows = read_manifest(manifest_path)
        for r in rows:
            if all(r[k] == row[k] for k in ("dataset", "setting", "variant", "method", "image_source", "seed")):
                r["status"] = status
                break
        write_manifest(manifest_path, rows)


def run_one(row: dict, gpu: int) -> bool:
    script = script_for(row["dataset"], row["method"])
    tag = f"{row['dataset']}__{row['setting']}__{row['variant']}__{row['method']}__{row['image_source']}__seed{row['seed']}"
    log_dir = LOG_DIR / row["dataset"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    cmd = ["python", str(script),
           "--setting", row["setting"],
           "--variant", row["variant"],
           "--image-source", row["image_source"],
           "--seed", str(row["seed"]),
           "--gpu", str(gpu)]

    with log_path.open("w") as logf:
        logf.write(f"# {datetime.now().isoformat()} | cmd = {' '.join(cmd)}\n")
        logf.flush()
        try:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=3600)
        except subprocess.TimeoutExpired:
            logf.write("\n# TIMEOUT after 3600s\n")
            return False
        logf.write(f"\n# exit={proc.returncode}\n")
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-runs", type=int, default=10000)
    args = ap.parse_args()

    manifest_path = MANIFEST_DIR / args.manifest
    if not manifest_path.exists():
        print(f"MISSING: {manifest_path}", flush=True)
        return

    print(f"[worker gpu={args.gpu}] manifest={manifest_path.name}", flush=True)
    n_done = 0
    while n_done < args.max_runs:
        row = claim_next_pending(manifest_path)
        if row is None:
            print(f"[worker gpu={args.gpu}] no pending rows, done", flush=True)
            return
        tag = f"{row['setting']}/{row['variant']}/{row['method']}/{row['image_source']}/seed{row['seed']}"
        t0 = time.time()
        print(f"[worker gpu={args.gpu}] START {tag}", flush=True)
        ok = run_one(row, args.gpu)
        elapsed = time.time() - t0
        status = "done" if ok else "failed"
        mark_done(manifest_path, row, status)
        print(f"[worker gpu={args.gpu}] {status.upper()} {tag}  ({elapsed:.1f}s)", flush=True)
        n_done += 1


if __name__ == "__main__":
    main()
