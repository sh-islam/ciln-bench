"""Equivalence tests: public/ vs canonical/.

These tests assert that the clean public interfaces produce bit-identical
output to the canonical implementations used to generate the released
CILN-Bench datasets.

Run with::

    python tests/check_equivalence.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "corrupt"))
sys.path.insert(0, str(ROOT / "code" / "analyze"))
sys.path.insert(0, str(ROOT / "code" / "analyze" / "canonical"))


# ============================================================
# 1. Seeding
# ============================================================
from public.seeding import derive_per_item_seed as _pub_seed
from canonical.seeding_image import derive_per_image_seed as _can_seed


def test_seeding() -> bool:
    cases = [
        (0,  "cifar10", "gaussian_noise", 3, 0,    "test"),
        (42, "mnist",   "rotate",         5, 1234, "test"),
        (0,  "cifar10", "contrast",       5, 99,   "noisy_label_train"),
        (123,"adult",   "missing_mar",    1, 7,    "test"),
    ]
    ok = True
    for c in cases:
        a, b = _pub_seed(*c), _can_seed(*c)
        if a != b:
            print(f"  SEEDING MISMATCH: {c} -> pub={a}, can={b}")
            ok = False
    if ok:
        print(f"  seeding: {len(cases)}/{len(cases)} cases match")
    return ok


# ============================================================
# 2. Image corruption (public re-exports canonical funcs)
# ============================================================
from public.image_corruptions import get_image_corruption
from canonical import cifar_funcs, mnist_funcs


def test_image_reexport() -> bool:
    pairs = [
        ("cifar10", "gaussian_noise", cifar_funcs.gaussian_noise),
        ("cifar10", "contrast",       cifar_funcs.contrast),
        ("cifar10", "brightness",     cifar_funcs.brightness),
        ("mnist",   "rotate",         mnist_funcs.rotate),
        ("mnist",   "shear",          mnist_funcs.shear),
    ]
    ok = True
    for ds, corr, expected in pairs:
        got = get_image_corruption(ds, corr)
        if got is not expected:
            print(f"  IMAGE RE-EXPORT MISMATCH: ({ds},{corr}) -> got {got} expected {expected}")
            ok = False
    if ok:
        print(f"  image re-export: {len(pairs)}/{len(pairs)} corruptions OK")
    return ok


def test_image_call_equivalence() -> bool:
    """Calling the public-looked-up function gives the same output as calling canonical directly."""
    img = (np.random.default_rng(0).integers(0, 256, (32, 32, 3))).astype(np.uint8)
    ok = True
    for corr in ["brightness", "contrast", "gaussian_noise"]:
        f_pub = get_image_corruption("cifar10", corr)
        f_can = getattr(cifar_funcs, corr)
        out_p, _ = f_pub(img, severity=3, rng=np.random.default_rng(42))
        out_c, _ = f_can(img, severity=3, rng=np.random.default_rng(42))
        if not np.array_equal(np.asarray(out_p), np.asarray(out_c)):
            print(f"  IMAGE CALL MISMATCH: cifar10 {corr}")
            ok = False
    if ok:
        print("  image call equivalence: 3/3 corruptions bit-identical")
    return ok


# ============================================================
# 3. Tabular corruption (public is dataset-agnostic; Adult call must match canonical)
# ============================================================
from public import tabular_corruptions as pub_tab
from canonical import adult_funcs as can_adult


def test_tabular_adult_equivalence() -> bool:
    cfg = pub_tab.adult_config()
    row = {
        "age": 30, "education-num": 12, "capital-gain": 0, "capital-loss": 0,
        "hours-per-week": 40, "workclass": "Private", "education": "Bachelors",
        "marital-status": "Single", "occupation": "Tech-support",
        "relationship": "Not-in-family", "race": "White", "sex": "Female",
        "native-country": "United-States",
    }
    n_checks = 0
    n_ok = 0
    for name in ["missing_mcar", "missing_mar", "scaling", "gaussian_noise"]:
        for sev in [1, 3, 5]:
            for seed in [0, 42, 1234]:
                n_checks += 1
                rng_p = np.random.default_rng(seed)
                rng_c = np.random.default_rng(seed)
                out_p, _ = pub_tab.CORRUPTION_FUNCTIONS[name](row, sev, rng_p, config=cfg)
                out_c, _ = can_adult.CORRUPTION_FUNCTIONS[name](row, sev, rng_c)
                # Compare cell by cell, handling NaN equality and float closeness
                same = set(out_p) == set(out_c)
                for k in out_p:
                    pv, cv = out_p[k], out_c[k]
                    if isinstance(pv, float) and isinstance(cv, float):
                        same &= (np.isnan(pv) and np.isnan(cv)) or pv == cv
                    else:
                        same &= pv == cv
                if same:
                    n_ok += 1
                else:
                    print(f"  TABULAR MISMATCH: {name} sev={sev} seed={seed}")
                    print(f"    pub: {out_p}")
                    print(f"    can: {out_c}")
    if n_ok == n_checks:
        print(f"  tabular Adult equivalence: {n_ok}/{n_checks} cases bit-identical")
    return n_ok == n_checks


# ============================================================
# 4. Analysis metrics (FRV / VDV)
# ============================================================
from public.metrics import flip_rate_variance, vote_distribution_variance, soft_label_from_voters
import compute_idn_2x2 as can_idn  # type: ignore


CIFAR_TEST_ROOT = Path('/home/misla2/thesis/python/mnist_corruptions/journal_edition'
                       '/output_seed0_TEST/cifar10')


def _load_real_setting(corr: str, sev: int):
    """Load p, true_y from a real journal_edition v2 setting directory."""
    sd = CIFAR_TEST_ROOT / corr / f"severity_{sev}" / "v2"
    voters = ['resnet20', 'wrn28_10', 'deit3_small', 'clip']
    argmax = np.stack(
        [np.load(sd / f"softmax_{v}.npy").argmax(axis=1) for v in voters], axis=1
    )
    p = soft_label_from_voters(argmax, n_classes=10)
    true_y = np.load(sd / "labels.npy")
    return p, true_y, sd


def test_metrics_equivalence() -> bool:
    """FRV and VDV from public match the canonical implementation bit-for-bit."""
    if not CIFAR_TEST_ROOT.exists():
        print("  metrics: skipped (no journal_edition data on disk)")
        return True
    cases = [("contrast", 5), ("gaussian_noise", 3)]
    ok = True
    for corr, sev in cases:
        sd = CIFAR_TEST_ROOT / corr / f"severity_{sev}" / "v2"
        if not (sd / "labels.npy").exists():
            print(f"  metrics: skipped {corr}_sev{sev} (missing data)")
            continue
        p, true_y, _ = _load_real_setting(corr, sev)
        frv_p = flip_rate_variance(p, true_y)
        vdv_p = vote_distribution_variance(p, true_y)
        frv_c = can_idn.m1_rate_variance(p, true_y)
        vdv_c = can_idn.m2_frobenius(p, true_y)
        if frv_p != frv_c or vdv_p != vdv_c:
            print(f"  METRICS MISMATCH: {corr}_sev{sev}  "
                  f"FRV pub={frv_p} can={frv_c}  VDV pub={vdv_p} can={vdv_c}")
            ok = False
        else:
            print(f"  {corr}_sev{sev}: FRV={frv_p:.6f}, VDV={vdv_p:.6f} (bit-identical)")
    return ok


# ============================================================
# 5. TV vs CIFAR-10H
# ============================================================
from public.tv import tv_vs_human
import tv_vs_cifar10h as can_tv  # type: ignore


CIFAR10H_COUNTS = Path('/home/misla2/thesis/python/mnist_corruptions'
                       '/cifar10h/cifar10h-counts.npy')


def test_tv_equivalence() -> bool:
    """Public TV pipeline matches canonical on at least one real setting."""
    sd = CIFAR_TEST_ROOT / "contrast" / "severity_5" / "v2"
    if not sd.exists() or not CIFAR10H_COUNTS.exists():
        print("  tv: skipped (no journal_edition or cifar10h data on disk)")
        return True
    p_ours, _ = can_tv.soft_distribution_from_voters(sd)
    true_y = np.load(sd / "labels.npy")
    human_counts = np.load(CIFAR10H_COUNTS)

    can_r = can_tv.run_setting("contrast_sev5", sd)
    pub_r = tv_vs_human(p_ours, true_y, human_counts, n_voters=4, n_boot=50)

    triples = [
        ("noise", pub_r.noise_rate, can_r["noise_rate_ours"]),
        ("tv_ours", pub_r.tv_ours_mean, can_r["tv_ours"]["mean"]),
        ("tv_symmetric", pub_r.tv_symmetric_mean, can_r["tv_symmetric"]["mean"]),
        ("tv_ccn", pub_r.tv_ccn_mean, can_r["tv_ccn"]["mean"]),
    ]
    ok = True
    for name, pv, cv in triples:
        if pv != cv:
            print(f"  TV MISMATCH on {name}: pub={pv} can={cv} (diff={abs(pv-cv):.2e})")
            ok = False
    if ok:
        print(f"  TV contrast_sev5: ours={pub_r.tv_ours_mean:.6f}, "
              f"sym={pub_r.tv_symmetric_mean:.6f}, ccn={pub_r.tv_ccn_mean:.6f} "
              "(all bit-identical)")
    return ok


# ============================================================
# Run all
# ============================================================
def main() -> int:
    print("=" * 60)
    print("CILN-Bench public vs canonical equivalence tests")
    print("=" * 60)
    results = {
        "seeding":            test_seeding(),
        "image_reexport":     test_image_reexport(),
        "image_call":         test_image_call_equivalence(),
        "tabular_adult":      test_tabular_adult_equivalence(),
        "metrics_frv_vdv":    test_metrics_equivalence(),
        "tv_vs_cifar10h":     test_tv_equivalence(),
    }
    failed = [k for k, v in results.items() if not v]
    print()
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
