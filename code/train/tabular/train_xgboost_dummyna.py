"""XGBoost voter on Adult — RTDL recipe, dummy_na=True variant.

Same hyperparameters as train_xgboost.py. The only difference is that
pd.get_dummies is called with dummy_na=True, which adds one extra one-hot
column per categorical column ("<col>_nan") so that "this cell is missing"
is a distinct signal XGBoost can split on.

Why a separate checkpoint:
  - Train Adult has no NaN cells (we drop them in load_adult), so the new
    `_nan` dummy columns will be all-zero during training. XGBoost will
    learn no split on them. At eval time on corrupted (MCAR/MAR/MNAR)
    data, those columns become non-zero — XGBoost will then route those
    rows through its default missing-direction at each split, which IS
    learned (over the rest of the column block) rather than treating
    them as all-zero out-of-distribution rows.
  - More importantly: by passing dummy_na=True we ensure the column
    schema XGBoost sees at eval matches what we want — i.e. there is a
    distinct "missing" indicator per categorical.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (load_adult, split_train_val, VoterResult, print_result,
                    CATEGORICAL_COLS, TRAIN_SEED)


def main():
    X_train, y_train, X_test, y_test = load_adult()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

    # One-hot encode with dummy_na=True so a "<col>_nan" column exists for each
    # categorical. Train Adult has no NaN cells so these are all-zero during
    # training, but the schema lets eval-time NaN rows flag themselves.
    X_full = pd.concat([X_tr, X_val, X_test], ignore_index=True)
    X_full = pd.get_dummies(X_full, columns=CATEGORICAL_COLS, drop_first=False, dummy_na=True)
    n_tr, n_val = len(X_tr), len(X_val)
    X_tr_enc = X_full.iloc[:n_tr]
    X_val_enc = X_full.iloc[n_tr:n_tr+n_val]
    X_test_enc = X_full.iloc[n_tr+n_val:]

    cfg = dict(
        booster="gbtree",
        n_estimators=2000,
        learning_rate=0.0280,
        max_depth=10,
        min_child_weight=4.06,
        subsample=0.953,
        colsample_bytree=0.541,
        colsample_bylevel=0.617,
        reg_lambda=2.9e-8,
        reg_alpha=0,
        gamma=0,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=TRAIN_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    t0 = time.time()
    clf = xgb.XGBClassifier(**cfg)
    clf.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
    wall = time.time() - t0

    val_acc = clf.score(X_val_enc, y_val)
    test_acc = clf.score(X_test_enc, y_test)
    train_acc = clf.score(X_tr_enc, y_tr)

    res = VoterResult(
        voter_name="xgboost_dummyna",
        dataset="adult",
        best_val_acc=val_acc,
        test_acc=test_acc,
        train_acc=train_acc,
        best_iter_or_epoch=int(clf.best_iteration) if hasattr(clf, "best_iteration") else None,
        total_wall_sec=wall,
        config={**cfg, "dummy_na": True},
        notes="RTDL tuned per-Adult; one-hot with dummy_na=True so '__missing__' is a distinct feature.",
    )
    print_result(res)
    res.save()

    from common import CKPT_DIR
    clf.save_model(str(CKPT_DIR / "xgboost_adult_dummyna.json"))


if __name__ == "__main__":
    main()
