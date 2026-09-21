"""Download Gu et al. 2022's CIFAR-10 noisy-label release from their public GCS bucket.

Pulls all 3 noise levels (low / medium / high) into:
    journal_edition/gu_compare/data/cifar10/<low|medium|high>/
        train-*-of-*
        valid-*-of-*
        rater_features.json
        manifest.json  (added by this script: per-file sha256 + total size)

237 MB total (79 MB per noise level). Anonymous public read; no auth needed.

Usage:
    python gu_compare/download_gu_cifar10.py
    python gu_compare/download_gu_cifar10.py --resume   # skip files already downloaded
    python gu_compare/download_gu_cifar10.py --noise-levels low medium  # subset
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from google.cloud import storage

BUCKET = "noisy_label_synthetic_datasets"
PREFIX_TPL = "public/cifar10/{level}/"
HERE = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = HERE / "data" / "cifar10"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_level(client, bucket, level: str, out_root: Path, resume: bool):
    prefix = PREFIX_TPL.format(level=level)
    dest = out_root / level
    dest.mkdir(parents=True, exist_ok=True)
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        print(f"  [WARN] no blobs found under gs://{BUCKET}/{prefix}")
        return None
    print(f"\n=== {level}  ({len(blobs)} files, "
          f"{sum(b.size for b in blobs)/1024/1024:.1f} MB) ===", flush=True)

    file_records = []
    total_downloaded = 0
    for blob in sorted(blobs, key=lambda b: b.name):
        fname = blob.name.split("/")[-1]
        target = dest / fname
        if resume and target.exists() and target.stat().st_size == blob.size:
            print(f"  [SKIP] {fname}  ({blob.size/1024/1024:.1f} MB, already present)", flush=True)
        else:
            t0 = time.time()
            blob.download_to_filename(str(target))
            wall = time.time() - t0
            mb = blob.size / 1024 / 1024
            print(f"  downloaded {fname}  ({mb:.1f} MB in {wall:.1f}s = {mb/wall:.1f} MB/s)", flush=True)
            total_downloaded += blob.size
        file_records.append({
            "file": fname,
            "size_bytes": blob.size,
            "sha256_local": sha256_of(target),
            "gcs_md5_hash": blob.md5_hash,
            "gcs_etag": blob.etag,
        })

    manifest = {
        "source": f"gs://{BUCKET}/{prefix}",
        "paper": "Gu et al. 2022, An Instance-Dependent Simulation Framework "
                 "for Learning with Label Noise (arXiv:2107.11413)",
        "noise_level": level,
        "n_files": len(file_records),
        "total_size_bytes": sum(r["size_bytes"] for r in file_records),
        "downloaded_at_unix": int(time.time()),
        "files": file_records,
    }
    with open(dest / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"Download destination (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--noise-levels", nargs="+", default=["low", "medium", "high"],
                    choices=["low", "medium", "high"])
    ap.add_argument("--resume", action="store_true",
                    help="Skip files that already exist with the expected size")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)

    overall = {"levels": {}, "started_at_unix": int(time.time())}
    for level in args.noise_levels:
        m = download_level(client, bucket, level, args.out_root, args.resume)
        if m is not None:
            overall["levels"][level] = {
                "n_files": m["n_files"],
                "total_size_bytes": m["total_size_bytes"],
            }
    overall["ended_at_unix"] = int(time.time())
    with open(args.out_root / "download_manifest.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote top-level manifest: {args.out_root / 'download_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
