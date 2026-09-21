"""V4 prep: build per-condition (image_indices, noisy_labels, true_labels)
arrays for the 4-arch v2 voter pool, paired with the 5 matched-rate settings
from EXP4_PLAN.md.

Pool (canonical paper-wide): resnet20, wrn28_10, deit3_small, clip.

For each of the 5 settings we produce TWO conditions:
  ours_v2_clean__<setting>  -- training images are clean CIFAR-10
  ours_v2_noisy__<setting>  -- training images are the corrupted NLT images
Both share the same noisy labels.

Sampling protocol (mirrors Gu's "pick 1 of 10 raters per image" and the
Table 2 noise-rate convention in the walkthrough):
  Each image's noisy label is one of the 4 voters' argmax prediction on the
  CORRUPTED image, chosen uniformly at random per image with a fixed seed.
  Result: noise rate is the average voter error rate, identical (up to
  finite-sample noise) to Table 2's argmax-voting `noise_rate` field.

Why this is the right split:
  Labels come from voters seeing the corrupted image -- that is the
  corruption-mediated noise generation process. The clean/noisy-image
  condition then asks: given the same noisy label set, does Co-teaching's
  robustness gain depend on whether the *learner* also sees the corruption?

Row set: the full NoisyLabelTrain split (22,500 rows). Matches Table 2.
No clean-correct restriction.

Outputs:
  results_v4/data_v4/<condition>/
    image_indices.npy        -- CIFAR-10 train indices for the row set
    noisylabeltrain_pos.npy  -- positions within the NLT split (for slicing corrupted images.npy)
    noisy_labels.npy
    true_labels.npy
    meta.json
"""
import hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES_DIR = ROOT / "results_v4"
DATA_DIR = RES_DIR / "data_v4"
RES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

JOURNAL_ROOT = ROOT.parent
SPLITS_DIR = JOURNAL_ROOT / "output_seed0" / "splits" / "cifar10"
NLT_ROOT = JOURNAL_ROOT / "output_seed0" / "cifar10"
CLEAN_INFER = JOURNAL_ROOT / "output_seed0" / "clean" / "cifar10" / "noisylabeltrain_clean"

VOTERS = ["resnet20", "wrn28_10", "deit3_small", "clip"]

# Table 2 settings (4-pool noise-rate-matched to Gu's 3 tiers).
# Format: (corruption_family, severity, gu_tier, super_family).
SETTINGS = [
    # gu_low (~10.8%)
    ("frost",             1, "low",    "weather"),
    ("pixelate",          1, "low",    "digital"),
    ("defocus_blur",      1, "low",    "blur"),
    # gu_medium (~19.4%)
    ("snow",              3, "medium", "weather"),
    ("elastic_transform", 3, "medium", "geometric"),
    ("gaussian_noise",    1, "medium", "noise"),
    # gu_high (~47.8%)
    ("contrast",          5, "high",   "digital"),
    ("gaussian_noise",    3, "high",   "noise"),
    ("shot_noise",        5, "high",   "noise"),
]


def main():
    nlt_idx = np.load(SPLITS_DIR / "noisylabeltrain_indices.npy")
    n_nlt = len(nlt_idx)
    print(f"NLT split size: {n_nlt}")

    # True labels for every NLT row -- the labels.npy in any NLT subdir is the
    # true class (same across all severities, by NLT construction).
    nlt_true = np.load(CLEAN_INFER / "labels.npy")
    assert len(nlt_true) == n_nlt
    all_pos = np.arange(n_nlt)

    # ---- For each setting, sample labels per Table 2 protocol ----
    # Pick one of 4 voter argmaxes per image, uniformly at random.
    summary = []
    for family_name, sev, regime, super_family in SETTINGS:
        setting_key = f"{family_name}_sev{sev}"
        nlt_dir = NLT_ROOT / family_name / f"severity_{sev}" / "noisy_label_train"
        if not (nlt_dir / f"softmax_{VOTERS[0]}.npy").exists():
            print(f"  SKIP {setting_key}: missing voter softmax in {nlt_dir}")
            continue

        # (N, 4) matrix of per-voter argmax predictions on the corrupted image.
        per_voter_argmax = np.stack(
            [np.load(nlt_dir / f"softmax_{v}.npy").argmax(axis=1) for v in VOTERS],
            axis=1,
        )  # (n_nlt, 4)

        # Reproducible per-condition sampling seed.
        seed = int(hashlib.sha256(f"v4__{setting_key}".encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        voter_pick = rng.integers(0, len(VOTERS), size=n_nlt)
        sampled = per_voter_argmax[np.arange(n_nlt), voter_pick].astype(np.int64)

        nr = float((sampled != nlt_true).mean())
        # Sanity check: this should ~= Table 2's noise_rate field, which is
        # the average voter error rate.
        per_voter_err = (per_voter_argmax != nlt_true[:, None]).mean(axis=0)
        expected_nr = float(per_voter_err.mean())

        for variant in ("clean", "noisy"):
            cond_name = f"ours_v2_{variant}__{setting_key}"
            out = DATA_DIR / cond_name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "image_indices.npy", nlt_idx)
            np.save(out / "noisylabeltrain_pos.npy", all_pos)
            np.save(out / "noisy_labels.npy", sampled)
            np.save(out / "true_labels.npy", nlt_true)
            meta = {
                "condition": cond_name,
                "pipeline": "ours_v2",
                "image_source": variant,  # clean | noisy
                "setting": setting_key,
                "family": super_family,
                "regime": regime,
                "pool": VOTERS,
                "n_rows": int(n_nlt),
                "sampled_noise_rate": nr,
                "expected_noise_rate_table2": expected_nr,
                "sampling_seed": seed,
                "row_set": "full NoisyLabelTrain (matches Table 2 protocol)",
                "noisy_images_path": str(nlt_dir / "images.npy"),
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  {cond_name:>52s}: N={n_nlt} sampled={nr*100:5.2f}% (Table2-expected={expected_nr*100:5.2f}%, regime {regime})")
            summary.append(meta)

    (RES_DIR / "pairs_v4.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {len(summary)} conditions to {DATA_DIR}")


if __name__ == "__main__":
    main()
