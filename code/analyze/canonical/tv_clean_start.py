"""One-off: compute filtered TV-to-CIFAR-10H on the 9 matched settings + find
3 settings per tier whose FILTERED noise rate matches PL-IDN tiers.

Reuses logic from tv_vs_cifar10h.py but applies the clean-correct test-set mask
(images where all 4 v2 voters classify the CLEAN test image correctly).

Output:
- Per-setting: noise_unfilt, noise_filt, TV_unfilt, TV_filt
- For all 45 settings, so we can re-match on filtered noise rate
"""
import json, time
from pathlib import Path
import numpy as np

JR = Path('/path/to/ciln-workspace/journal_edition')
TEST_ROOT = JR / 'output_seed0_TEST' / 'cifar10'
CLEAN_TEST = JR / 'output_seed0' / 'clean' / 'cifar10' / 'test'
OUT = JR / 'idn_2x2' / 'results'
CIFAR10H = JR.parent / 'cifar10h' / 'cifar10h-counts.npy'

V2_VOTERS = ['resnet20', 'wrn28_10', 'deit3_small', 'clip']
N_VOTERS = len(V2_VOTERS)
N_CLASSES = 10
N_BOOT = 50

# Build clean-correct mask on test set
clean_labels = np.load(CLEAN_TEST / 'labels.npy')
clean_preds = []
for v in V2_VOTERS:
    sm = np.load(CLEAN_TEST / f'softmax_{v}.npy')
    clean_preds.append(sm.argmax(axis=1))
clean_preds = np.stack(clean_preds, axis=1)
CC_MASK = (clean_preds == clean_labels[:, None]).all(axis=1)
print(f'Clean-correct mask: {CC_MASK.sum()} / {len(clean_labels)} test images = {CC_MASK.mean()*100:.2f}%')


def soft_distribution_from_voters(setting_dir):
    votes = []
    for v in V2_VOTERS:
        sm = np.load(setting_dir / f'softmax_{v}.npy')
        votes.append(sm.argmax(axis=1))
    votes = np.stack(votes, axis=1)
    n_img = len(votes)
    p = np.zeros((n_img, N_CLASSES), dtype=np.float64)
    for k in range(N_VOTERS):
        np.add.at(p, (np.arange(n_img), votes[:, k]), 1.0)
    p /= N_VOTERS
    return p


def cifar10h_upsampled(target_rate, true_y, mask, seed):
    """Upsample CIFAR-10H to target rate, restricted to masked rows."""
    rng = np.random.default_rng(seed)
    counts = np.load(CIFAR10H)
    counts_m = counts[mask]
    true_y_m = true_y[mask]
    correct = np.array([counts_m[i, true_y_m[i]] for i in range(len(true_y_m))])
    wrong_total = counts_m.sum(axis=1) - correct
    p_current = counts_m.astype(np.float64) / counts_m.sum(axis=1, keepdims=True)
    current_rate = (1 - p_current[np.arange(len(true_y_m)), true_y_m]).mean()

    if target_rate <= current_rate:
        soft = p_current.copy()
    else:
        sum_correct = correct.sum()
        sum_wrong = wrong_total.sum()
        if sum_wrong == 0:
            soft = p_current
        else:
            k = target_rate * sum_correct / ((1 - target_rate) * sum_wrong)
            up = counts_m.astype(np.float64).copy()
            for i in range(len(true_y_m)):
                ty = true_y_m[i]
                wmask = np.arange(N_CLASSES) != ty
                up[i, wmask] *= k
            row_sum = up.sum(axis=1, keepdims=True)
            soft = up / np.maximum(row_sum, 1e-12)

    sampled = np.zeros((len(true_y_m), N_CLASSES), dtype=np.float64)
    for i in range(len(true_y_m)):
        draws = rng.choice(N_CLASSES, size=N_VOTERS, p=soft[i])
        for d in draws:
            sampled[i, d] += 1
    sampled /= N_VOTERS
    return sampled


def tv_mean(p, q):
    return float(0.5 * np.abs(p - q).sum(axis=1).mean())


def run_setting_both(name, setting_dir):
    p_ours = soft_distribution_from_voters(setting_dir)
    true_y = np.load(setting_dir / 'labels.npy')

    # UNFILTERED
    eta_unfilt = (1.0 - p_ours[np.arange(len(p_ours)), true_y]).mean()
    tv_unfilt_list = []
    for b in range(N_BOOT):
        ref = cifar10h_upsampled(eta_unfilt, true_y, np.ones(len(true_y), dtype=bool), seed=b)
        tv_unfilt_list.append(tv_mean(p_ours, ref))

    # FILTERED (clean-correct mask)
    p_ours_f = p_ours[CC_MASK]
    true_y_f = true_y[CC_MASK]
    eta_filt = (1.0 - p_ours_f[np.arange(len(p_ours_f)), true_y_f]).mean()
    tv_filt_list = []
    for b in range(N_BOOT):
        ref_f = cifar10h_upsampled(eta_filt, true_y, CC_MASK, seed=b)
        tv_filt_list.append(tv_mean(p_ours_f, ref_f))

    return {
        'name': name,
        'noise_unfilt': float(eta_unfilt),
        'noise_filt': float(eta_filt),
        'tv_unfilt': float(np.mean(tv_unfilt_list)),
        'tv_unfilt_ci95': float(1.96 * np.std(tv_unfilt_list)),
        'tv_filt': float(np.mean(tv_filt_list)),
        'tv_filt_ci95': float(1.96 * np.std(tv_filt_list)),
    }


def main():
    print(f'[{time.strftime("%X")}] filtered TV starting...')
    results = []
    settings = []
    for c_dir in sorted(TEST_ROOT.iterdir()):
        if not c_dir.is_dir(): continue
        for s_dir in sorted(c_dir.iterdir()):
            if not s_dir.is_dir(): continue
            v2_dir = s_dir / 'v2'
            if not (v2_dir / 'labels.npy').exists(): continue
            name = f"{c_dir.name}_sev{s_dir.name.split('_')[1]}"
            settings.append((name, v2_dir))
    print(f'  {len(settings)} settings')

    for i, (name, sd) in enumerate(settings):
        t0 = time.time()
        r = run_setting_both(name, sd)
        results.append(r)
        print(f'  [{i+1}/{len(settings)}] {name}: '
              f'noise={r["noise_unfilt"]*100:.1f}%/{r["noise_filt"]*100:.1f}%  '
              f'TV={r["tv_unfilt"]:.3f}/{r["tv_filt"]:.3f}  '
              f'[{time.time()-t0:.1f}s]')

    out_file = OUT / 'tv_filtered_oneoff.json'
    json.dump(results, open(out_file, 'w'), indent=2)
    print(f'\nSaved to {out_file}')

    # Quick summary at PL-IDN tiers
    print('\n=== Comparison 1: 9 ORIGINAL matched settings (pre-matched to unfilt PL-IND tiers) ===')
    orig_matched = {
        'low (~11%)': ['frost_sev1', 'pixelate_sev1', 'defocus_blur_sev1'],
        'med (~19%)': ['snow_sev3', 'elastic_transform_sev3', 'gaussian_noise_sev1'],
        'high (~48%)': ['contrast_sev5', 'gaussian_noise_sev3', 'shot_noise_sev5'],
    }
    by_name = {r['name']: r for r in results}
    for tier, names in orig_matched.items():
        print(f'\n  {tier}')
        for nm in names:
            if nm in by_name:
                r = by_name[nm]
                print(f'    {nm:30s}  noise: {r["noise_unfilt"]*100:5.1f}% -> {r["noise_filt"]*100:5.1f}%  '
                      f'TV: {r["tv_unfilt"]:.3f} -> {r["tv_filt"]:.3f}')

    # Comparison 2: re-match on filt noise rate
    print('\n=== Comparison 2: 3 settings/tier matched on FILTERED noise rate ===')
    PLIDN = {'low': 10.8, 'med': 19.4, 'high': 47.8}
    for tier, target in PLIDN.items():
        # rank all 45 by |noise_filt - target|, pick top 3 from distinct super-families
        ranked = sorted(results, key=lambda r: abs(r['noise_filt']*100 - target))
        print(f'\n  PL-IND {tier} = {target}%')
        for r in ranked[:5]:
            print(f'    {r["name"]:30s}  noise_filt: {r["noise_filt"]*100:5.1f}%  '
                  f'(unfilt: {r["noise_unfilt"]*100:5.1f}%)  TV_filt: {r["tv_filt"]:.3f}')


if __name__ == '__main__':
    main()
