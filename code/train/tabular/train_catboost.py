"""CatBoost voter on Adult — paper-aligned recipe with library defaults.

Source: Prokhorenkova et al. NeurIPS 2018 (paper notes Adult benefits most from
ordered boosting; no per-Adult tuned values published in the main text).
We use library defaults (lr=0.03, depth=6, l2_leaf_reg=3) plus the categorical
feature list so ordered target statistics fire.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from catboost import CatBoostClassifier, Pool

from common import (load_adult, split_train_val, VoterResult, print_result,
                    CATEGORICAL_COLS, TRAIN_SEED)


def main():
    X_train, y_train, X_test, y_test = load_adult()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

    # CatBoost wants categoricals as strings (or object dtype)
    for c in CATEGORICAL_COLS:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    cat_idx = [X_tr.columns.get_loc(c) for c in CATEGORICAL_COLS]

    cfg = dict(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=TRAIN_SEED,
        od_type="Iter",
        od_wait=50,
        verbose=False,
        allow_writing_files=False,
    )

    t0 = time.time()
    clf = CatBoostClassifier(**cfg)
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)
    clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
    wall = time.time() - t0

    val_acc = (clf.predict(val_pool).flatten().astype(int) == y_val.values).mean()
    test_acc = (clf.predict(test_pool).flatten().astype(int) == y_test.values).mean()
    train_acc = (clf.predict(train_pool).flatten().astype(int) == y_tr.values).mean()

    res = VoterResult(
        voter_name="catboost",
        dataset="adult",
        best_val_acc=val_acc,
        test_acc=test_acc,
        train_acc=train_acc,
        best_iter_or_epoch=clf.tree_count_,
        total_wall_sec=wall,
        config=cfg,
        notes="Library defaults + cat_features explicit (ordered boosting active)",
    )
    print_result(res)
    res.save()

    from common import CKPT_DIR
    clf.save_model(str(CKPT_DIR / "catboost_adult.cbm"))


if __name__ == "__main__":
    main()
