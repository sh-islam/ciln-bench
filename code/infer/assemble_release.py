"""Assemble benchmark_v3/ release directories from existing cached softmaxes.

For each admitted setting:
  - Create benchmark_v3/<dataset>/<corr>/severity_<sev>/noisy_label_train/
  - Symlink or copy:
      * images.npy (the corrupted images, from output_seed0/)
      * labels.npy (true labels, from output_seed0/)
      * softmax_<voter>.npy (per-voter softmaxes)
      * avg_softmax.npy (mean over voters)
      * noisy_labels_argmax.npy (argmax of avg_softmax)
      * noisy_labels_majority_vote.npy (modal class of per-voter argmaxes)
      * noisy_labels_sampled.npy (sample from avg_softmax; seed 0)
      * manifest.json (setting metadata)

We use symlinks where possible to save disk space; corrupted images and
voter softmaxes already exist on disk.

Outputs:
  benchmark_v3/cifar10/<corr>/severity_<sev>/...
  benchmark_v3/mnist/<corr>/severity_<sev>/...
  benchmark_v3/adult/<corr>/severity_<sev>/...
  benchmark_v3/manifest.json
  benchmark_v3/README.md
"""
import json, hashlib
from pathlib import Path
import numpy as np

JOURNAL_ROOT = Path('/path/to/ciln-workspace/journal_edition')
SOURCE_C10 = JOURNAL_ROOT / 'output_seed0' / 'cifar10'
SOURCE_M = JOURNAL_ROOT / 'output_seed0' / 'mnist'
SOURCE_A = JOURNAL_ROOT / 'output_seed0' / 'adult'
NEW_POOL_C10 = JOURNAL_ROOT / 'gu_arch_voters' / 'inference' / 'per_setting'
OUT = JOURNAL_ROOT / 'benchmark_v3'

# CIFAR-10 new pool (7 voters)
C10_VOTERS = ["mobilenetv1", "mobilenetv2", "vgg16", "resnet50", "resnet101", "nasnetmobile", "inception_v4"]
# MNIST pool (unchanged from v2)
M_VOTERS = ["lenet5", "mlp", "resnet20", "deit3_small"]
# Adult pool (unchanged from v2)
A_VOTERS = ["catboost", "xgboost_dummyna", "rtdl_mlp", "rtdl_fttransformer", "tabpfn_v1"]


def avg_voter_noise_rate(softmaxes, labels):
    errs = [float((sm.argmax(axis=1) != labels).mean()) for sm in softmaxes]
    return float(np.mean(errs))


def assemble_setting(setting_name, dataset, src_train, voter_softs, voter_files, out_train, links=True):
    """Create one setting's release directory."""
    out_train.mkdir(parents=True, exist_ok=True)
    labels = np.load(src_train / 'labels.npy')
    images = src_train / 'images.npy'
    # symlink images and labels
    for src_file, dst_name in [(images, 'images.npy'), (src_train / 'labels.npy', 'labels.npy')]:
        dst = out_train / dst_name
        if dst.exists() or dst.is_symlink(): dst.unlink()
        if links: dst.symlink_to(src_file)
        else:
            import shutil; shutil.copy(src_file, dst)
    # symlink per-voter softmaxes
    for v, src_voter_file in zip(C10_VOTERS if dataset == 'cifar10' else (M_VOTERS if dataset == 'mnist' else A_VOTERS), voter_files):
        dst = out_train / f'softmax_{v}.npy'
        if dst.exists() or dst.is_symlink(): dst.unlink()
        if links: dst.symlink_to(src_voter_file)
        else:
            import shutil; shutil.copy(src_voter_file, dst)
    # compute avg_softmax (mean over voters)
    avg = np.mean(voter_softs, axis=0).astype(np.float32)
    np.save(out_train / 'avg_softmax.npy', avg)
    # argmax
    argmax = avg.argmax(axis=1).astype(np.int64)
    np.save(out_train / 'noisy_labels_argmax.npy', argmax)
    # majority vote across per-voter argmaxes
    per_voter_argmax = np.stack([sm.argmax(axis=1) for sm in voter_softs], axis=1)
    majority = np.zeros(len(labels), dtype=np.int64)
    for i in range(len(labels)):
        cnts = np.bincount(per_voter_argmax[i], minlength=avg.shape[1])
        majority[i] = int(cnts.argmax())
    np.save(out_train / 'noisy_labels_majority_vote.npy', majority)
    # sampled
    rng = np.random.default_rng(0)
    sampled = np.array([rng.choice(avg.shape[1], p=avg[i]) for i in range(len(labels))], dtype=np.int64)
    np.save(out_train / 'noisy_labels_sampled.npy', sampled)
    # per-voter noise rates
    per_voter_errs = {v: float((sm.argmax(axis=1) != labels).mean()) for v, sm in zip(
        C10_VOTERS if dataset == 'cifar10' else (M_VOTERS if dataset == 'mnist' else A_VOTERS),
        voter_softs)}
    avg_v = float(np.mean(list(per_voter_errs.values())))
    nr_argmax = float((argmax != labels).mean())
    nr_majvote = float((majority != labels).mean())
    nr_sampled = float((sampled != labels).mean())
    manifest = {
        'setting': setting_name, 'dataset': dataset, 'n_rows': int(len(labels)),
        'voters': C10_VOTERS if dataset == 'cifar10' else (M_VOTERS if dataset == 'mnist' else A_VOTERS),
        'per_voter_noise_rate': per_voter_errs,
        'avg_voter_noise_rate': avg_v,
        'noise_rate_argmax': nr_argmax,
        'noise_rate_majority_vote': nr_majvote,
        'noise_rate_sampled': nr_sampled,
        'release_rule': 'avg-voter >= 15% (over all voters, no collapse exclusion)',
    }
    (out_train / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


# --- CIFAR-10: all 45 settings ---
print("=== CIFAR-10 ===")
cifar_manifests = []
for c_dir in sorted(SOURCE_C10.iterdir()):
    if not c_dir.is_dir(): continue
    for s_dir in sorted(c_dir.iterdir()):
        if not s_dir.is_dir(): continue
        src_train = s_dir / 'noisy_label_train'
        if not (src_train / 'images.npy').exists(): continue
        setting = f"{c_dir.name}_sev{s_dir.name.split('_')[1]}"
        # load per-voter softmaxes from NEW pool
        per_setting_dir = NEW_POOL_C10 / setting
        voter_files = [per_setting_dir / f"softmax_{v}.npy" for v in C10_VOTERS]
        if not all(f.exists() for f in voter_files):
            print(f"  SKIP {setting}: missing voter softmax")
            continue
        voter_softs = [np.load(f).astype(np.float32) for f in voter_files]
        # admit gate: avg-voter >=15%
        labels = np.load(src_train / 'labels.npy')
        avg_v = avg_voter_noise_rate(voter_softs, labels)
        if avg_v < 0.15:
            print(f"  REJECT {setting}: avg-v {avg_v*100:.2f}% < 15%")
            continue
        out_train = OUT / 'cifar10' / c_dir.name / s_dir.name / 'noisy_label_train'
        m = assemble_setting(setting, 'cifar10', src_train, voter_softs, voter_files, out_train)
        cifar_manifests.append(m)
        print(f"  OK   {setting:>26s}  avg-v={m['avg_voter_noise_rate']*100:5.2f}%  arg={m['noise_rate_argmax']*100:5.2f}%")
print(f"CIFAR-10 admitted: {len(cifar_manifests)}/45\n")

# --- MNIST: 15 distinct (after severity-invariant dedup) ---
# rule: avg-voter >=15%, AND for severity-invariant corruptions keep only sev1
SEV_INVARIANT = {'stripe', 'canny_edges'}
print("=== MNIST ===")
mnist_manifests = []
seen_invariant = set()
for c_dir in sorted(SOURCE_M.iterdir()):
    if not c_dir.is_dir(): continue
    for s_dir in sorted(c_dir.iterdir()):
        if not s_dir.is_dir(): continue
        src_train = s_dir / 'noisy_label_train'
        if not (src_train / 'images.npy').exists(): continue
        sev = s_dir.name.split('_')[1]
        # severity-invariant dedup
        if c_dir.name in SEV_INVARIANT:
            if c_dir.name in seen_invariant: continue
            if sev != '1': continue  # only keep sev1
            seen_invariant.add(c_dir.name)
        setting = f"{c_dir.name}_sev{sev}"
        voter_files = [src_train / f"softmax_{v}.npy" for v in M_VOTERS]
        if not all(f.exists() for f in voter_files):
            print(f"  SKIP {setting}: missing voter softmax")
            continue
        voter_softs = [np.load(f).astype(np.float32) for f in voter_files]
        labels = np.load(src_train / 'labels.npy')
        avg_v = avg_voter_noise_rate(voter_softs, labels)
        if avg_v < 0.15:
            continue
        out_train = OUT / 'mnist' / c_dir.name / s_dir.name / 'noisy_label_train'
        m = assemble_setting(setting, 'mnist', src_train, voter_softs, voter_files, out_train)
        mnist_manifests.append(m)
        print(f"  OK   {setting:>26s}  avg-v={m['avg_voter_noise_rate']*100:5.2f}%  arg={m['noise_rate_argmax']*100:5.2f}%")
print(f"MNIST admitted: {len(mnist_manifests)}\n")

# --- Adult: 4 settings, copy from v2 since rule is special ---
print("=== Adult ===")
adult_manifests = []
V2_ADULT = JOURNAL_ROOT / 'benchmark_v2' / 'adult'
import shutil
for c_dir in sorted(V2_ADULT.iterdir()):
    if not c_dir.is_dir(): continue
    for s_dir in sorted(c_dir.iterdir()):
        if not s_dir.is_dir(): continue
        for split_dir in s_dir.iterdir():
            if not split_dir.is_dir(): continue
            out = OUT / 'adult' / c_dir.name / s_dir.name / split_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in split_dir.iterdir():
                dst = out / f.name
                if dst.exists() or dst.is_symlink(): dst.unlink()
                dst.symlink_to(f)
    setting = f"{c_dir.name}_sev5"
    adult_manifests.append({'setting': setting, 'dataset': 'adult'})
    print(f"  OK   {setting} (copied from v2)")
print(f"Adult admitted: {len(adult_manifests)}\n")

# top-level manifest
total = len(cifar_manifests) + len(mnist_manifests) + len(adult_manifests)
top_manifest = {
    'release_version': 'v3',
    'created_at': '2026-05-30',
    'release_rule': 'avg-voter error rate >= 15% (averaged across all voters in the pool, no collapse exclusion). MNIST: severity-invariant corruptions (stripe, canny_edges) deduplicated to severity 1 only.',
    'total_settings': total,
    'cifar10': {'n_settings': len(cifar_manifests), 'voters': C10_VOTERS,
                'pool_recipe': 'Gu medium-noise recipe: batch 256, LR 0.01 cosine, 17000 steps, SGD/0.9/1e-4, flip+crop, 224x224 input'},
    'mnist': {'n_settings': len(mnist_manifests), 'voters': M_VOTERS},
    'adult': {'n_settings': len(adult_manifests), 'voters': A_VOTERS},
}
(OUT / 'manifest.json').write_text(json.dumps(top_manifest, indent=2))

readme = f"""# Benchmark v3 release

Created: 2026-05-30. Supersedes v2 (33 settings) with the simpler v3 release rule.

## Release rule

A setting is included iff its **avg-voter error rate** (the per-voter
`argmax(softmax) != true_label` rate, averaged across all voters in the
pool) is at least **15%**. No collapse-survival filter at the rule level.
MNIST severity-invariant corruptions (`stripe`, `canny_edges`) are
deduplicated to severity 1 only.

## Pool definitions

- **CIFAR-10** (7 voters): Inception-v4, MobileNet-v1, MobileNet-v2,
  NASNetMobile, ResNet50, ResNet101, VGG16. Retrained on `CleanLabelTrain`
  with Gu medium-noise recipe (batch 256, LR 0.01 cosine, 17,000 steps,
  SGD/0.9/1e-4, flip+crop, 224x224 input).
- **MNIST** (4 voters, unchanged from v2): LeNet-5, MLP, ResNet-20,
  DeiT3-Small.
- **Adult** (5 voters, unchanged from v2): XGBoost-dummyna, CatBoost,
  RTDL-MLP, RTDL-FT-Transformer, TabPFN-v1.

## Per-setting files

Each setting ships `noisy_label_train/`:
- `images.npy` (uint8) - the corrupted images
- `labels.npy` (int64) - true labels
- `softmax_<voter>.npy` (float32) - per-voter softmax distribution
- `avg_softmax.npy` (float32) - mean over voters (canonical label)
- `noisy_labels_argmax.npy` (int64) - argmax of avg_softmax
- `noisy_labels_majority_vote.npy` (int64) - modal of per-voter argmaxes
- `noisy_labels_sampled.npy` (int64) - sampled from avg_softmax (seed 0)
- `manifest.json` - per-setting metadata

(Many files are symlinks back to `output_seed0/` for disk efficiency.)

## Total: {total} settings

- CIFAR-10: {len(cifar_manifests)} of 45 candidates
- MNIST: {len(mnist_manifests)} (after severity-invariant dedup)
- Adult: {len(adult_manifests)}

See `manifest.json` for the full per-dataset metadata.
"""
(OUT / 'README.md').write_text(readme)
print(f"=== TOTAL: {total} settings ===")
print(f"Wrote {OUT}/manifest.json, {OUT}/README.md")
EOF