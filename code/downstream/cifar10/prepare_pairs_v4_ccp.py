"""V4 CCP prep: build clean-correct-precondition (CCP) variants of the three
high-noise matched-tier settings for the downstream CCP ablation.

CCP row set: keep only rows where ALL 4 voters classify the CLEAN version
of the NLT image correctly. Same noisy labels (produced from voter argmaxes on
the CORRUPTED image) as the non-CCP variant, just restricted to this subset.

Settings (Table 4 high tier = ~48% noise):
    contrast_sev5         (47.3% full, ~43.7% CCP per Table 6)
    gaussian_noise_sev3   (49.2% full, ~44.7% CCP per Table 6)
    shot_noise_sev5       (~54.5% full)

For each setting we produce TWO conditions (matches v4 naming):
    ours_v2_ccp_clean__<setting>
    ours_v2_ccp_noisy__<setting>

The training scripts (train_erm_v4.py, train_coteaching_v4.py) read meta.json
"image_source" to route training images; "clean" -> clean CIFAR rows by
image_indices, "noisy" -> corrupted images.npy by noisylabeltrain_pos. The
CCP filter just shrinks both index arrays consistently.
"""
import hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES_DIR = ROOT / "results_v4"
DATA_DIR = RES_DIR / "data_v4"

JOURNAL_ROOT = ROOT.parent
SPLITS_DIR = JOURNAL_ROOT / "output_seed0" / "splits" / "cifar10"
NLT_ROOT = JOURNAL_ROOT / "output_seed0" / "cifar10"
CLEAN_INFER = JOURNAL_ROOT / "output_seed0" / "clean" / "cifar10" / "noisylabeltrain_clean"

VOTERS = ["resnet20", "wrn28_10", "deit3_small", "clip"]

SETTINGS = [
    ("contrast",       5, "high", "digital"),
    ("gaussian_noise", 3, "high", "noise"),
    ("shot_noise",     5, "high", "noise"),
    ("pixelate",       5, "high", "digital"),
    ("glass_blur",     3, "high", "blur"),
]


def main():
    nlt_idx = np.load(SPLITS_DIR / "noisylabeltrain_indices.npy")
    n_nlt = len(nlt_idx)
    nlt_true = np.load(CLEAN_INFER / "labels.npy")
    assert len(nlt_true) == n_nlt

    # CCP mask: all 4 voters correct on the CLEAN NLT image.
    clean_argmax = np.stack(
        [np.load(CLEAN_INFER / f"softmax_{v}.npy").argmax(axis=1) for v in VOTERS],
        axis=1,
    )  # (n_nlt, 4)
    all_correct_clean = (clean_argmax == nlt_true[:, None]).all(axis=1)
    ccp_pos = np.where(all_correct_clean)[0]
    print(f"CCP mask: {len(ccp_pos):>6d} / {n_nlt} kept ({len(ccp_pos)/n_nlt*100:.2f}%)")

    nlt_idx_ccp = nlt_idx[ccp_pos]
    nlt_true_ccp = nlt_true[ccp_pos]

    summary = []
    for family_name, sev, regime, super_family in SETTINGS:
        setting_key = f"{family_name}_sev{sev}"
        nlt_dir = NLT_ROOT / family_name / f"severity_{sev}" / "noisy_label_train"
        miss = [v for v in VOTERS if not (nlt_dir / f"softmax_{v}.npy").exists()]
        if miss:
            print(f"  SKIP {setting_key}: missing voter softmax(es) {miss}")
            continue

        per_voter_argmax = np.stack(
            [np.load(nlt_dir / f"softmax_{v}.npy").argmax(axis=1) for v in VOTERS],
            axis=1,
        )  # (n_nlt, 4)

        # Same sampling seed convention as prepare_pairs_v4.py so labels match.
        seed = int(hashlib.sha256(f"v4__{setting_key}".encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        voter_pick = rng.integers(0, len(VOTERS), size=n_nlt)
        sampled_full = per_voter_argmax[np.arange(n_nlt), voter_pick].astype(np.int64)

        # Apply CCP filter to the sampled labels.
        sampled = sampled_full[ccp_pos]
        nr = float((sampled != nlt_true_ccp).mean())
        expected_full_nr = float((per_voter_argmax != nlt_true[:, None]).mean())

        for variant in ("clean", "noisy"):
            cond_name = f"ours_v2_ccp_{variant}__{setting_key}"
            out = DATA_DIR / cond_name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "image_indices.npy", nlt_idx_ccp)
            np.save(out / "noisylabeltrain_pos.npy", ccp_pos)
            np.save(out / "noisy_labels.npy", sampled)
            np.save(out / "true_labels.npy", nlt_true_ccp)
            meta = {
                "condition": cond_name,
                "pipeline": "ours_v2_ccp",
                "image_source": variant,
                "setting": setting_key,
                "family": super_family,
                "regime": regime,
                "pool": VOTERS,
                "n_rows": int(len(ccp_pos)),
                "n_rows_full": int(n_nlt),
                "ccp_keep_frac": float(len(ccp_pos) / n_nlt),
                "sampled_noise_rate": nr,
                "expected_noise_rate_full": expected_full_nr,
                "sampling_seed": seed,
                "row_set": "CCP: all 4 voters correct on clean NLT image",
                "noisy_images_path": str(nlt_dir / "images.npy"),
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  {cond_name:>52s}: N={len(ccp_pos)} sampled={nr*100:5.2f}% "
                  f"(full-tier expected={expected_full_nr*100:5.2f}%, regime {regime})")
            summary.append(meta)

    (RES_DIR / "pairs_v4_ccp.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {len(summary)} CCP conditions to {DATA_DIR}")


if __name__ == "__main__":
    main()
