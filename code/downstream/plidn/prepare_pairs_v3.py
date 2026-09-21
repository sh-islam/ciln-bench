"""Build per-rater-matched Exp 4 pairs (v3): for each Gu noise level, pair
with our v3-admitted setting(s) at matching per-rater (avg-voter) noise rate.

Label sampling (the design we agreed on):
  - Gu: for each image, sample 1 of the 10 raters uniformly -> use its hard label.
  - Ours: for each image, sample 1 class from the 7-voter average softmax
    distribution -> use that class as the hard label.

This makes both pipelines equivalent: "sample one noisy label per image from
the pipeline's per-image label distribution." By construction the expected
noise rate of each sampled set matches the per-rater (avg-voter) rate Gu
advertises (10.8% / 19.5% / 47.8%).

Sampling is fixed by a single seed (RNG seeded with the pair name) so both
ERM and Co-teaching see the SAME noisy label set across all training seeds.

Pairs chosen (v3, per-rater matching):
  gu_low (10.8%)      <-> elastic_transform_sev3 (10.5%)
  gu_medium (19.5%)   <-> fog_sev3 (20.9%)
  gu_medium (19.5%)   <-> motion_blur_sev3 (22.1%)
  gu_high (47.8%)     <-> impulse_noise_sev5 (48.4%)
  gu_high (47.8%)     <-> contrast_sev3 (45.3%)

Output:
  results_v3/pairs.json
  results_v3/data/<pair_name>/
    image_indices.npy        (CIFAR-10 train indices for the rows used)
    true_labels.npy
    ours_noisy_labels.npy    (sampled hard labels, dtype int64)
    gu_noisy_labels.npy      (sampled hard labels, dtype int64)
    meta.json
"""
import glob, hashlib, json
from pathlib import Path
import numpy as np
import torchvision
import tfrecord

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES_DIR = ROOT / "results_v3"
DATA_DIR = RES_DIR / "data"
RES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

JOURNAL_ROOT = ROOT.parent
SPLITS_DIR = JOURNAL_ROOT / "output_seed0" / "splits" / "cifar10"
CIFAR_NEW_POOL = JOURNAL_ROOT / "gu_arch_voters" / "inference" / "per_setting"
CIFAR_CLEAN_DIR = JOURNAL_ROOT / "gu_arch_voters" / "inference" / "per_setting_clean"
CIFAR_SOURCE = JOURNAL_ROOT / "output_seed0" / "cifar10"
GU_DIR = JOURNAL_ROOT / "gu_compare" / "data" / "cifar10"

CIFAR_VOTERS = ["mobilenetv1", "mobilenetv2", "vgg16", "resnet50", "resnet101", "nasnetmobile", "inception_v4"]

# Pairs to build
PAIRS = [
    {"gu_level": "low",    "ours": "elastic_transform_sev3"},
    {"gu_level": "medium", "ours": "fog_sev3"},
    {"gu_level": "medium", "ours": "motion_blur_sev3"},
    {"gu_level": "high",   "ours": "impulse_noise_sev5"},
    {"gu_level": "high",   "ours": "contrast_sev3"},
]


GU_SCHEMA = {"image/raw": "byte", "image/class/label": "int",
             "noisy_labels": "int", "rater_ids": "byte"}


def load_gu_by_hash(level):
    out = {}
    for shard in sorted(glob.glob(str(GU_DIR / level / "train-*"))):
        for ex in tfrecord.tfrecord_loader(shard, None, description=GU_SCHEMA):
            h = hashlib.sha256(bytes(ex["image/raw"])).hexdigest()
            v = list(map(int, ex["noisy_labels"]))
            t = int(ex["image/class/label"][0]) if hasattr(ex["image/class/label"], "__len__") else int(ex["image/class/label"])
            out[h] = (v, t)
    return out


def main():
    # 1) Build NLT row index -> (CIFAR-10 train idx, sha-hash)
    NLT_IDX = np.load(SPLITS_DIR / "noisylabeltrain_indices.npy")
    ds = torchvision.datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=True, download=False)
    images_all = ds.data
    labels_all = np.array(ds.targets)
    hash_to_pos = {hashlib.sha256(images_all[idx].tobytes()).hexdigest(): (pos, idx)
                   for pos, idx in enumerate(NLT_IDX)}

    # 2) Load all 3 Gu levels by hash
    print("Loading Gu's 3 levels by hash ...")
    gu = {lvl: load_gu_by_hash(lvl) for lvl in ("low", "medium", "high")}

    # 3) Intersection: rows that are (a) in our NLT, (b) in Gu's 3 levels (same hash set)
    # NOTE: we deliberately do NOT restrict to clean-correct here. The Exp 4 question
    # is "given the released noise, how well can ERM/Co-teaching recover clean test
    # accuracy?" The released noise lives on the full NLT. Restricting to clean-
    # correct biases toward easy rows where Gu's raters also rarely err, making
    # noise rates much lower than the published per-rater rates and breaking the
    # matched-rate comparison. Clean-correct provenance was an Exp 1/2/3 claim
    # about the noise structure; Exp 4 is about downstream learning on the
    # released noise.
    common_hashes = sorted(set(gu["low"].keys()) & set(gu["medium"].keys()) & set(gu["high"].keys()) & set(hash_to_pos.keys()))
    keep_hashes = common_hashes
    common_pos = np.array([hash_to_pos[h][0] for h in keep_hashes])
    common_idx = np.array([hash_to_pos[h][1] for h in keep_hashes])
    common_true = labels_all[common_idx]
    print(f"Intersection (NLT cap Gu): {len(keep_hashes)} rows")

    # 5) For each pair: build sampled noisy labels (Gu: pick 1 of 10 raters; ours: sample from avg-softmax)
    pairs_out = []
    for p in PAIRS:
        lvl = p["gu_level"]
        ours_setting = p["ours"]
        pair_name = f"gu_{lvl}__vs__{ours_setting}"
        out = DATA_DIR / pair_name
        out.mkdir(exist_ok=True)

        # Seed by pair name for reproducibility (stable across reruns)
        seed = int(hashlib.sha256(pair_name.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)

        # ----- Gu: sample one rater per image -----
        gu_votes_per_image = np.stack(
            [np.asarray(gu[lvl][h][0], dtype=np.int64) for h in keep_hashes], axis=0
        )  # (N, 10)
        N, K_gu = gu_votes_per_image.shape
        rater_idx = rng.integers(0, K_gu, size=N)
        gu_sampled = gu_votes_per_image[np.arange(N), rater_idx]
        gu_noise_rate = float((gu_sampled != common_true).mean())

        # ----- Ours: sample from avg-softmax of the 7 voters -----
        per_voter_softs = []
        for arch in CIFAR_VOTERS:
            sm = np.load(CIFAR_NEW_POOL / ours_setting / f"softmax_{arch}.npy").astype(np.float64)
            per_voter_softs.append(sm[common_pos])
        avg_soft = np.mean(per_voter_softs, axis=0)  # (N, 10)
        # row-wise normalize (avg softmax is already row-stochastic, but renormalize for safety)
        avg_soft = avg_soft / avg_soft.sum(axis=1, keepdims=True)
        # sample one class per row
        ours_sampled = np.zeros(N, dtype=np.int64)
        cum = np.cumsum(avg_soft, axis=1)
        u = rng.random(size=N)
        for i in range(N):
            ours_sampled[i] = np.searchsorted(cum[i], u[i])
            if ours_sampled[i] >= 10:
                ours_sampled[i] = 9
        ours_noise_rate = float((ours_sampled != common_true).mean())

        # Save arrays
        np.save(out / "image_indices.npy", common_idx)
        np.save(out / "true_labels.npy", common_true)
        np.save(out / "ours_noisy_labels.npy", ours_sampled)
        np.save(out / "gu_noisy_labels.npy", gu_sampled)

        meta = {
            "name": pair_name,
            "gu_level": lvl,
            "ours_setting": ours_setting,
            "gu_noise_rate_sampled": gu_noise_rate,
            "ours_noise_rate_sampled": ours_noise_rate,
            "noise_rate_gap": ours_noise_rate - gu_noise_rate,
            "n_rows": int(N),
            "sampling_seed": seed,
            "row_set": "clean-correct intersection (ours clean-correct mask cap Gu shared rows)",
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        pairs_out.append(meta)
        print(f"  {pair_name}: N={N}  gu={gu_noise_rate*100:5.2f}%  ours={ours_noise_rate*100:5.2f}%  gap={meta['noise_rate_gap']*100:+5.2f}pp")

    (RES_DIR / "pairs.json").write_text(json.dumps(pairs_out, indent=2))
    print(f"\nsaved pairs.json + per-pair arrays")


if __name__ == "__main__":
    main()
