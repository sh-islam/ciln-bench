"""Argmax-uniform voter sampling protocol.

For each row, take every voter's argmax prediction (hard label), then sample
one voter uniformly at random and use its argmax as the observed noisy label.

This matches the paper's stated protocol (benchmark.tex:65) which is the same
as PL-IDN's uniform-rater sampling. Given the same seed, the sampled label set
is deterministic and reproducible.

Inputs: per-voter softmax .npy files at
    {setting_dir}/noisy_label_train/softmax_{voter}.npy

Outputs:
    y_noisy: (N,) int64 sampled noisy label per row
    voter_ids: (N,) int64 which voter each label came from
    manifest: dict with {seed, voters, per_voter_argmax_noise_rates,
                         sampled_noise_rate, n_rows}
"""
import json
import numpy as np
from pathlib import Path


def sample_argmax_uniform(setting_dir: Path, voters, seed: int, mask: np.ndarray = None):
    """Sample one voter uniformly per row; return that voter's argmax label.

    setting_dir: path containing softmax_{voter}.npy files and labels.npy.
    voters: list of voter name strings, in canonical order.
    seed: RNG seed for reproducibility.
    mask: optional bool array of length N to restrict to a subset (CILN-C mask).

    Returns (y_noisy, voter_ids, manifest, y_true).
    """
    # Load per-voter argmax predictions
    voter_argmaxes = []
    for v in voters:
        sm = np.load(setting_dir / f"softmax_{v}.npy")
        voter_argmaxes.append(sm.argmax(axis=1))
    voter_argmaxes = np.stack(voter_argmaxes, axis=1)  # (N, M)

    # Load ground truth
    y_true = np.load(setting_dir / "labels.npy").astype(np.int64)

    # Optional clean-start mask
    if mask is not None:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        voter_argmaxes = voter_argmaxes[mask]
        y_true = y_true[mask]

    n, m = voter_argmaxes.shape

    # Sample one voter per row uniformly
    rng = np.random.default_rng(seed)
    voter_ids = rng.integers(0, m, size=n).astype(np.int64)
    y_noisy = voter_argmaxes[np.arange(n), voter_ids].astype(np.int64)

    # Diagnostics
    per_voter_nr = [(voter_argmaxes[:, i] != y_true).mean() for i in range(m)]
    sampled_nr = (y_noisy != y_true).mean()

    manifest = {
        "seed": int(seed),
        "voters": list(voters),
        "n_rows": int(n),
        "per_voter_argmax_noise_rates": [float(r) for r in per_voter_nr],
        "sampled_noise_rate": float(sampled_nr),
        "expected_noise_rate": float(np.mean(per_voter_nr)),
        "n_voters": int(m),
    }
    return y_noisy, voter_ids, manifest, y_true


def save_sampling_artifact(out_dir: Path, y_noisy, voter_ids, manifest, y_true):
    """Persist the sampling artifact for reproducibility and re-analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_noisy.npy", y_noisy)
    np.save(out_dir / "voter_ids.npy", voter_ids)
    np.save(out_dir / "y_true.npy", y_true)
    (out_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    # Smoke test using Adult
    import sys
    setting = Path("/path/to/ciln-workspace/journal_edition/output_seed0/adult/gaussian_noise/severity_5/noisy_label_train")
    voters = ["xgboost_dummyna", "catboost", "mlp", "ft_transformer", "tabpfn"]
    y, vids, mani, yt = sample_argmax_uniform(setting, voters, seed=0)
    print(f"n={mani['n_rows']}")
    print(f"per-voter noise rates: {mani['per_voter_argmax_noise_rates']}")
    print(f"expected noise rate (mean of per-voter): {mani['expected_noise_rate']*100:.2f}%")
    print(f"sampled noise rate (this seed): {mani['sampled_noise_rate']*100:.2f}%")
