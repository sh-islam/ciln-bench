"""Per-voter softmax inference for the 4 tabular voters on clean + corrupted Adult.

Voters covered (TabPFN deferred per user):
  xgboost, catboost, mlp, ft_transformer

For each setting we write:
  softmax_<voter>.npy   shape (N, 2) float32

Clean test softmax goes to output_seed{S}/clean/adult/softmax_<voter>.npy.

NaN handling strategy (corrupted inputs can have NaN cells from MCAR/MAR/MNAR):
  - XGBoost handles NaN natively.
  - CatBoost: cast categoricals to str; replace pandas NaN/'nan'/'None' with the
    sentinel string "__NaN__" so CatBoost treats it as a regular category value.
    Numerical NaN is supported natively by CatBoost.
  - MLP / FT-T: categoricals are mapped via the union-of-train-val-test dict;
    NaN cells get a reserved sentinel index appended to each column's vocab
    (`__NaN__`), and a corresponding zero embedding row is appended at load
    time. Numerical NaN is filled with the per-column train median (computed
    once from the clean train split) before the QuantileTransformer is applied.

The preprocessing helpers used by training (load_adult, split_train_val,
cat-union vocab, QuantileTransformer) are recomputed here from the clean data
so we are bit-identical with the training-time setup (deterministic seeds).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
TABULAR_ROOT = HERE.parent
JOURNAL_ROOT = TABULAR_ROOT.parent
sys.path.insert(0, str(TABULAR_ROOT))
sys.path.insert(0, str(TABULAR_ROOT / "train"))

from common import (load_adult, split_train_val, CATEGORICAL_COLS,
                    NUMERICAL_COLS, TRAIN_SEED)
from train_mlp import CatEmbeddingMLP, CFG as MLP_CFG
from train_ft_transformer import FTTransformer, CFG as FT_CFG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAN_TOKEN = "__NaN__"


# --------------------- preprocessing fitted on clean train ---------------------

def fit_preprocessing():
    """Return everything we need to preprocess clean and corrupted test rows:
      - cat_maps: per-categorical str -> int index, with NAN_TOKEN reserved
        at the highest index (cardinality - 1).
      - cat_cardinalities: list[int]
      - num_medians: per-numerical median over clean train (for NaN fill)
      - qt: fitted QuantileTransformer over clean train numerics
    """
    X_train, y_train, X_test, y_test = load_adult()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

    cat_maps = {}
    for c in CATEGORICAL_COLS:
        union_vals = pd.concat([X_tr[c], X_val[c], X_test[c]]).astype(str).unique()
        m = {v: i for i, v in enumerate(sorted(union_vals))}
        m[NAN_TOKEN] = len(m)  # sentinel slot for missing
        cat_maps[c] = m

    cat_cardinalities = [len(cat_maps[c]) for c in CATEGORICAL_COLS]

    # quantile transformer fit on clean train numerics
    Xn_tr = X_tr[NUMERICAL_COLS].values.astype(np.float32)
    qt = QuantileTransformer(output_distribution="normal", random_state=TRAIN_SEED)
    qt.fit(Xn_tr)
    num_medians = {c: float(np.nanmedian(Xn_tr[:, i])) for i, c in enumerate(NUMERICAL_COLS)}

    return {
        "cat_maps": cat_maps,
        "cat_cardinalities": cat_cardinalities,
        "num_medians": num_medians,
        "qt": qt,
        "X_test_clean": X_test, "y_test_clean": y_test,
        "X_train_full": X_train, "y_train_full": y_train,
    }


def encode_for_mlp_or_ft(df: pd.DataFrame, pre: dict):
    """Return (num_arr float32 (N, 5), cat_arr int64 (N, 8)) ready for MLP / FT-T.
    NaN cells are filled / sentinel-encoded per the strategy in the module docstring.
    """
    cat_arr = np.empty((len(df), len(CATEGORICAL_COLS)), dtype=np.int64)
    for j, c in enumerate(CATEGORICAL_COLS):
        col = df[c]
        col_str = col.astype("object").where(col.notna(), NAN_TOKEN).astype(str)
        m = pre["cat_maps"][c]
        # any unseen string -> NaN sentinel index
        cat_arr[:, j] = col_str.map(lambda v: m.get(v, m[NAN_TOKEN])).values

    num_arr = np.empty((len(df), len(NUMERICAL_COLS)), dtype=np.float32)
    for j, c in enumerate(NUMERICAL_COLS):
        v = df[c].astype(np.float64).values
        fill = pre["num_medians"][c]
        v = np.where(np.isnan(v), fill, v).astype(np.float32)
        num_arr[:, j] = v
    num_arr = pre["qt"].transform(num_arr).astype(np.float32)
    return num_arr, cat_arr


def encode_for_xgboost(df: pd.DataFrame, pre: dict, X_full_template: pd.DataFrame,
                       dummy_na: bool = False):
    """One-hot encoding aligned to training-time columns of train_xgboost*.py.

    dummy_na=False (original xgboost_adult.json):
        Map NaN cat cells -> NAN_TOKEN string then get_dummies(dummy_na=False).
        NaN rows in cat cells become an extra '<col>_<NAN_TOKEN>' column whose
        train support is zero, so they fall through to a default split branch.

    dummy_na=True (xgboost_adult_dummyna.json):
        Pass cat cells through unchanged (NaN stays NaN) and call
        get_dummies(dummy_na=True). pandas adds an explicit '<col>_nan' column
        per categorical, with 1.0 for the NaN rows. XGBoost sees a distinct
        'this cell is missing' indicator.

    The clean (train, val, test) frame is concatenated so the union of category
    values yields the same column schema XGBoost was trained on.
    """
    df2 = df.copy()
    if not dummy_na:
        for c in CATEGORICAL_COLS:
            df2[c] = df2[c].astype("object").where(df2[c].notna(), NAN_TOKEN).astype(str)
    full = pd.concat([X_full_template, df2], ignore_index=True)
    full = pd.get_dummies(full, columns=CATEGORICAL_COLS, drop_first=False, dummy_na=dummy_na)
    n_template = len(X_full_template)
    enc = full.iloc[n_template:].reset_index(drop=True)
    return enc


def encode_for_catboost(df: pd.DataFrame):
    """Just cast categoricals to str (CatBoost rejects mixed dtypes), NaN cells
    -> '__NaN__' string so they become a regular category. Numerical NaN is
    preserved (CatBoost handles it natively)."""
    df2 = df.copy()
    for c in CATEGORICAL_COLS:
        df2[c] = df2[c].astype("object").where(df2[c].notna(), NAN_TOKEN).astype(str)
    return df2


# --------------------- voter loaders ---------------------

def load_xgboost(ckpt_dir: Path, dummy_na: bool = False):
    import xgboost as xgb
    clf = xgb.XGBClassifier()
    fname = "xgboost_adult_dummyna.json" if dummy_na else "xgboost_adult.json"
    clf.load_model(ckpt_dir / fname)
    return clf


def load_catboost(ckpt_dir: Path):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier()
    clf.load_model(str(ckpt_dir / "catboost_adult.cbm"))
    return clf


def load_mlp(ckpt_dir: Path, pre: dict):
    """The training-time MLP was built with cardinalities that did not include
    our NaN sentinel slot. To stay compatible with the saved weights we re-build
    using the SAME cardinalities as training (no sentinel), then expand each
    embedding to add one extra zero row for the sentinel."""
    cards_train = [n - 1 for n in pre["cat_cardinalities"]]  # exclude sentinel
    model = CatEmbeddingMLP(
        n_num=len(NUMERICAL_COLS),
        cat_cardinalities=cards_train,
        d_embedding=MLP_CFG["d_embedding"],
        d_layers=MLP_CFG["d_layers"],
        dropout=0.0,
    )
    state = torch.load(ckpt_dir / "mlp_adult_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    # Expand each embedding by one (zero) row for NaN sentinel
    new_embs = nn.ModuleList()
    for emb, card_full in zip(model.embeddings, pre["cat_cardinalities"]):
        new_emb = nn.Embedding(card_full, emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight.data[:emb.num_embeddings] = emb.weight.data
            new_emb.weight.data[emb.num_embeddings:] = 0.0  # sentinel row
        new_embs.append(new_emb)
    model.embeddings = new_embs
    return model.to(DEVICE).eval()


def load_ft_transformer(ckpt_dir: Path, pre: dict):
    """Same NaN-sentinel-expansion strategy as MLP: train-time cardinalities,
    then expand the categorical embeddings inside the tokenizer."""
    cards_train = [n - 1 for n in pre["cat_cardinalities"]]
    model = FTTransformer(
        n_num=len(NUMERICAL_COLS),
        cat_cardinalities=cards_train,
        cfg=FT_CFG,
    )
    state = torch.load(ckpt_dir / "ft_transformer_adult_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    new_embs = nn.ModuleList()
    for emb, card_full in zip(model.tokenizer.cat_embeddings, pre["cat_cardinalities"]):
        new_emb = nn.Embedding(card_full, emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight.data[:emb.num_embeddings] = emb.weight.data
            new_emb.weight.data[emb.num_embeddings:] = 0.0
        new_embs.append(new_emb)
    model.tokenizer.cat_embeddings = new_embs
    return model.to(DEVICE).eval()


# --------------------- inference ---------------------

@torch.no_grad()
def torch_softmax(model: nn.Module, num_arr: np.ndarray, cat_arr: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    n = len(num_arr)
    out = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        xn = torch.from_numpy(num_arr[sl]).to(DEVICE)
        xc = torch.from_numpy(cat_arr[sl]).to(DEVICE)
        logits = model(xn, xc)
        probs = torch.softmax(logits, dim=1).float().cpu().numpy()
        out[i:i + probs.shape[0]] = probs
    return out


def softmax_path(setting_path: Path, voter_name: str) -> Path:
    return setting_path / f"softmax_{voter_name}.npy"


def eval_xgboost(clf, df: pd.DataFrame, pre: dict, xgb_template: pd.DataFrame,
                 dummy_na: bool = False) -> np.ndarray:
    enc = encode_for_xgboost(df, pre, xgb_template, dummy_na=dummy_na)
    expected = clf.get_booster().feature_names
    enc = enc.reindex(columns=expected, fill_value=0)
    probs = clf.predict_proba(enc)
    return probs.astype(np.float32)


def eval_catboost(clf, df: pd.DataFrame) -> np.ndarray:
    df2 = encode_for_catboost(df)
    probs = clf.predict_proba(df2)
    return probs.astype(np.float32)


def eval_mlp_or_ft(model, df: pd.DataFrame, pre: dict) -> np.ndarray:
    num_arr, cat_arr = encode_for_mlp_or_ft(df, pre)
    return torch_softmax(model, num_arr, cat_arr)


# --------------------- TabPFN (32-subset ensemble) ---------------------
#
# TabPFN v1 caps context at 1000 train rows. Adult train has ~27K rows, so we
# average softmaxes over 32 different 1000-row stratified subsamples. Each call
# is one forward pass through TabPFN; the same test rows are passed in 32
# times with different training context each time.
#
# For corrupted test rows: TabPFN does NOT handle NaN natively. We fill NaN
# cat cells with the NAN_TOKEN string (one-hot will create a distinct column
# that maps to a never-trained-on indicator), and NaN numeric cells with the
# per-column train median (consistent with the MLP/FT-T NaN handling).
#
# Performance: each TabPFN call on 15K test rows takes ~20s on an A5000. 32
# members = ~10 min per setting × 16 settings = ~160 min total.

TABPFN_N_ENSEMBLE = 32
TABPFN_ROWS_PER_MEMBER = 1000


def fit_tabpfn_encoder(pre: dict):
    """Build the one-hot column schema and per-column train mean/std used by
    eval_tabpfn.py's training. Returns:
      - columns: list[str] (final feature names, post one-hot, in fixed order)
      - means, stds: np.ndarray each (n_features,) computed on clean train enc
      - X_tr_enc, y_tr: np.ndarray train-encoded (for subsampling at eval time)
    """
    X_train = pre["X_train_full"]
    X_test = pre["X_test_clean"]
    X_tr_split, y_tr, X_val_split, y_val = split_train_val(X_train, pre["y_train_full"])
    X_full = pd.concat([X_tr_split, X_val_split, X_test], ignore_index=True)
    X_full = pd.get_dummies(X_full, columns=CATEGORICAL_COLS, drop_first=False)
    X_full = X_full.astype(np.float32)
    columns = X_full.columns.tolist()
    n_tr = len(X_tr_split)
    X_tr_enc = X_full.iloc[:n_tr].values
    means = X_tr_enc.mean(axis=0)
    stds = X_tr_enc.std(axis=0) + 1e-8
    X_tr_norm = (X_tr_enc - means) / stds
    return {
        "columns": columns,
        "means": means,
        "stds": stds,
        "X_tr_enc": X_tr_norm,
        "y_tr": y_tr.values,
    }


def encode_for_tabpfn(df: pd.DataFrame, tab_enc: dict, xgb_template: pd.DataFrame) -> np.ndarray:
    """One-hot+normalize a (possibly corrupted) test df into the TabPFN feature
    matrix. NaN cat cells -> NAN_TOKEN; NaN numeric cells -> 0.0 (post-norm is
    fine since means/stds came from clean train; equivalently this is the train
    mean for that feature, then normalized to ~0).
    """
    df2 = df.copy()
    for c in CATEGORICAL_COLS:
        df2[c] = df2[c].astype("object").where(df2[c].notna(), NAN_TOKEN).astype(str)
    for c in NUMERICAL_COLS:
        df2[c] = df2[c].astype(np.float64)

    full = pd.concat([xgb_template, df2], ignore_index=True)
    full = pd.get_dummies(full, columns=CATEGORICAL_COLS, drop_first=False).astype(np.float32)
    enc = full.iloc[len(xgb_template):].reset_index(drop=True)
    enc = enc.reindex(columns=tab_enc["columns"], fill_value=0).astype(np.float32)
    arr = enc.values
    # NaN numerics become NaN here; fill with the column's train mean (so post-
    # normalization they're 0).
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        col_means_tile = np.tile(tab_enc["means"], (arr.shape[0], 1))
        arr = np.where(nan_mask, col_means_tile, arr).astype(np.float32)
    arr = (arr - tab_enc["means"]) / tab_enc["stds"]
    return arr.astype(np.float32)


def eval_tabpfn(tab_enc: dict, df: pd.DataFrame, xgb_template: pd.DataFrame,
                rng: np.random.Generator, device: str = "cuda",
                n_ensemble: int = TABPFN_N_ENSEMBLE,
                rows_per_member: int = TABPFN_ROWS_PER_MEMBER) -> np.ndarray:
    """Run TabPFN v1 with 32-subsample stratified ensemble; average softmaxes.

    Returns (N_test, 2) float32 mean-softmax over n_ensemble members.
    """
    from tabpfn import TabPFNClassifier

    X_test_enc = encode_for_tabpfn(df, tab_enc, xgb_template)
    # Feature subsample if > 100 (TabPFN v1 cap). Adult ohe ~106 features.
    X_tr_full = tab_enc["X_tr_enc"]
    y_tr_full = tab_enc["y_tr"]
    n_features = X_tr_full.shape[1]
    if n_features > 100:
        var = X_tr_full.var(axis=0)
        top_idx = np.argsort(var)[-100:]
    else:
        top_idx = np.arange(n_features)

    pos_idx = np.where(y_tr_full == 1)[0]
    neg_idx = np.where(y_tr_full == 0)[0]
    pos_frac = float(y_tr_full.mean())
    n_pos = max(1, int(round(rows_per_member * pos_frac)))
    n_neg = rows_per_member - n_pos

    probs_acc = np.zeros((len(X_test_enc), 2), dtype=np.float64)
    for i in range(n_ensemble):
        sub_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sub_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        sub_idx = np.concatenate([sub_pos, sub_neg])
        rng.shuffle(sub_idx)
        X_sub = X_tr_full[sub_idx][:, top_idx]
        y_sub = y_tr_full[sub_idx]
        X_test_sub = X_test_enc[:, top_idx]
        clf = TabPFNClassifier(device=device, N_ensemble_configurations=3, seed=TRAIN_SEED + i)
        clf.fit(X_sub, y_sub)
        p_test = clf.predict_proba(X_test_sub)
        probs_acc += p_test

    probs_acc /= n_ensemble
    return probs_acc.astype(np.float32)


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voters", nargs="+",
                    default=["xgboost_dummyna", "catboost", "mlp", "ft_transformer"],
                    help="subset of voters to run. choices: xgboost, "
                         "xgboost_dummyna, catboost, mlp, ft_transformer, tabpfn")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_root = JOURNAL_ROOT / f"output_seed{args.seed}"
    clean_root = out_root / "clean" / "adult"
    clean_root.mkdir(parents=True, exist_ok=True)
    corrupt_root = out_root / "adult"

    print("[setup] fitting preprocessing on clean Adult (CleanLabelTrain)...", flush=True)
    pre = fit_preprocessing()
    X_test_clean = pre["X_test_clean"].reset_index(drop=True)
    y_test_clean = pre["y_test_clean"].reset_index(drop=True)

    # XGBoost template: train+val+test in the order used during training
    X_tr, y_tr, X_val, y_val = split_train_val(pre["X_train_full"], pre["y_train_full"])
    xgb_template = pd.concat([X_tr, X_val, X_test_clean], ignore_index=True)

    # Load CleanLabelValid + the pre-corruption views of NoisyLabelTrain/Valid
    # for eval-time reporting. The latter two are the same rows that the
    # corrupted parquet files were derived from, BEFORE any corruption was
    # applied. We need their voter softmaxes to compute the "all voters
    # correct on the clean version" filter in the noise-rate table downstream.
    splits_root = JOURNAL_ROOT / "output_seed0" / "splits" / "adult"
    clv_idx = np.load(splits_root / "cleanlabelvalid_indices.npy")
    nlt_idx = np.load(splits_root / "noisylabeltrain_indices.npy")
    nlv_idx = np.load(splits_root / "noisylabelvalid_indices.npy")

    X_clv = pre["X_train_full"].iloc[clv_idx].reset_index(drop=True)
    y_clv = pre["y_train_full"].iloc[clv_idx].values
    X_nlt_clean = pre["X_train_full"].iloc[nlt_idx].reset_index(drop=True)
    y_nlt_clean = pre["y_train_full"].iloc[nlt_idx].values
    X_nlv_clean = pre["X_train_full"].iloc[nlv_idx].reset_index(drop=True)
    y_nlv_clean = pre["y_train_full"].iloc[nlv_idx].values

    # Save labels to the clean output dirs once
    for cs, labels in [
        ("cleanlabelvalid",        y_clv),
        ("test",                   y_test_clean.values),
        ("noisylabeltrain_clean",  y_nlt_clean),
        ("noisylabelvalid_clean",  y_nlv_clean),
    ]:
        (clean_root / cs).mkdir(parents=True, exist_ok=True)
        np.save(clean_root / cs / "labels.npy", labels)

    ckpt_dir = TABULAR_ROOT / "checkpoints"
    voters = {}
    if "xgboost" in args.voters:
        voters["xgboost"] = ("xgb", load_xgboost(ckpt_dir, dummy_na=False))
    if "xgboost_dummyna" in args.voters:
        voters["xgboost_dummyna"] = ("xgb_dummyna", load_xgboost(ckpt_dir, dummy_na=True))
    if "catboost" in args.voters:
        voters["catboost"] = ("cat", load_catboost(ckpt_dir))
    if "mlp" in args.voters:
        voters["mlp"] = ("torch", load_mlp(ckpt_dir, pre))
    if "ft_transformer" in args.voters:
        voters["ft_transformer"] = ("torch", load_ft_transformer(ckpt_dir, pre))
    if "tabpfn" in args.voters:
        print("[setup] fitting TabPFN one-hot encoder + train means/stds...", flush=True)
        tab_enc = fit_tabpfn_encoder(pre)
        voters["tabpfn"] = ("tabpfn", tab_enc)

    print(f"[setup] loaded voters: {list(voters)}", flush=True)

    # Build settings list: clean splits + corrupted noisy_label_train/valid subfolders
    settings = [
        ("clean/cleanlabelvalid",       X_clv,             y_clv,             clean_root / "cleanlabelvalid"),
        ("clean/test",                  X_test_clean,      y_test_clean.values, clean_root / "test"),
        ("clean/noisylabeltrain_clean", X_nlt_clean,       y_nlt_clean,       clean_root / "noisylabeltrain_clean"),
        ("clean/noisylabelvalid_clean", X_nlv_clean,       y_nlv_clean,       clean_root / "noisylabelvalid_clean"),
    ]
    if not args.clean_only:
        for corr_dir in sorted(corrupt_root.iterdir()):
            if not corr_dir.is_dir(): continue
            for sev_dir in sorted(corr_dir.iterdir()):
                if not sev_dir.is_dir(): continue
                for split_name in ("noisy_label_train", "noisy_label_valid"):
                    split_dir = sev_dir / split_name
                    pq = split_dir / "adult_corrupted.parquet"
                    if not pq.exists():
                        continue
                    df_c = pd.read_parquet(pq)
                    labels_c = np.load(split_dir / "labels.npy")
                    settings.append((f"adult/{corr_dir.name}/{sev_dir.name}/{split_name}",
                                     df_c, labels_c, split_dir))

    print(f"[plan] {len(settings)} settings x {len(voters)} voters = {len(settings)*len(voters)} files", flush=True)

    tabpfn_rng = np.random.default_rng(TRAIN_SEED)
    t0 = time.time()
    for (name, df, labels, path) in settings:
        for vname, (kind, model_or_clf) in voters.items():
            target = softmax_path(path, vname)
            if args.resume and target.exists():
                continue
            try:
                if kind == "xgb":
                    probs = eval_xgboost(model_or_clf, df, pre, xgb_template, dummy_na=False)
                elif kind == "xgb_dummyna":
                    probs = eval_xgboost(model_or_clf, df, pre, xgb_template, dummy_na=True)
                elif kind == "cat":
                    probs = eval_catboost(model_or_clf, df)
                elif kind == "tabpfn":
                    probs = eval_tabpfn(model_or_clf, df, xgb_template, tabpfn_rng,
                                        device="cuda" if torch.cuda.is_available() else "cpu")
                else:
                    probs = eval_mlp_or_ft(model_or_clf, df, pre)
                np.save(target, probs)
                pred = probs.argmax(axis=1)
                acc = float((pred == labels).mean())
                print(f"  [{vname:14s}] {name:42s} n={len(df):>6}  acc={acc:.4f}  -> {target.name}", flush=True)
            except Exception as e:
                print(f"  [{vname:14s}] {name:42s} FAILED: {e}", flush=True)
                raise

    total = time.time() - t0
    summary = {
        "voters": list(voters),
        "n_settings": len(settings),
        "total_sec": round(total, 1),
    }
    with open(out_root / "_eval_tabular_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== ALL DONE in {total:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
