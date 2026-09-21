"""MLP voter on Adult — RTDL per-Adult Optuna-tuned recipe.

Source: yandex-research/rtdl-revisiting-models output/adult/mlp/tuned/0.toml.
Architecture: per-categorical learned embedding (d_embedding=421) + 5 hidden
layers [42, 503, 503, 503, 111], AdamW lr=3.21e-4, wd=7.17e-5, quantile-normalized
numerical features, early-stop patience=16 on val loss.
"""
import os, sys, time, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from common import (load_adult, split_train_val, VoterResult, print_result,
                    CATEGORICAL_COLS, NUMERICAL_COLS, TRAIN_SEED, LOG_DIR, CKPT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RTDL-tuned values
CFG = dict(
    optimizer="AdamW", lr=3.21e-4, weight_decay=7.17e-5, batch_size=256,
    d_layers=[42, 503, 503, 503, 111], d_embedding=421, dropout=0.0,
    normalization="quantile", patience=16, max_epochs=200,
)


def prepare_adult():
    X_train, y_train, X_test, y_test = load_adult()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

    # Build categorical -> integer index per column, fitted on the union
    cat_maps = {}
    for c in CATEGORICAL_COLS:
        union_vals = pd.concat([X_tr[c], X_val[c], X_test[c]]).astype(str).unique()
        cat_maps[c] = {v: i for i, v in enumerate(sorted(union_vals))}

    def encode(df):
        cat_arr = np.stack([df[c].astype(str).map(cat_maps[c]).values for c in CATEGORICAL_COLS], axis=1)
        num_arr = df[NUMERICAL_COLS].values.astype(np.float32)
        return num_arr, cat_arr.astype(np.int64)

    Xn_tr, Xc_tr = encode(X_tr)
    Xn_val, Xc_val = encode(X_val)
    Xn_test, Xc_test = encode(X_test)

    # Quantile-normalize numerical features (fit on train)
    qt = QuantileTransformer(output_distribution="normal", random_state=TRAIN_SEED)
    Xn_tr = qt.fit_transform(Xn_tr).astype(np.float32)
    Xn_val = qt.transform(Xn_val).astype(np.float32)
    Xn_test = qt.transform(Xn_test).astype(np.float32)

    cat_cardinalities = [len(cat_maps[c]) for c in CATEGORICAL_COLS]
    return (Xn_tr, Xc_tr, y_tr.values), (Xn_val, Xc_val, y_val.values), \
           (Xn_test, Xc_test, y_test.values), cat_cardinalities


class CatEmbeddingMLP(nn.Module):
    """RTDL-style MLP: shared embedding dim per categorical, concat with numerical,
    then a stack of [Linear, ReLU, Dropout] layers, final Linear -> num_classes.
    """
    def __init__(self, n_num: int, cat_cardinalities, d_embedding: int,
                 d_layers, dropout: float, num_classes: int = 2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_embedding) for card in cat_cardinalities]
        )
        in_dim = n_num + len(cat_cardinalities) * d_embedding
        layers = []
        prev = in_dim
        for d in d_layers:
            layers += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x)


def acc(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def main():
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    (Xn_tr, Xc_tr, y_tr), (Xn_val, Xc_val, y_val), (Xn_test, Xc_test, y_test), cat_card = prepare_adult()

    def to_loader(Xn, Xc, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(Xn), torch.from_numpy(Xc), torch.from_numpy(y).long()
        )
        return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle)

    train_loader = to_loader(Xn_tr, Xc_tr, y_tr, shuffle=True)
    val_loader = to_loader(Xn_val, Xc_val, y_val, shuffle=False)
    test_loader = to_loader(Xn_test, Xc_test, y_test, shuffle=False)

    model = CatEmbeddingMLP(
        n_num=Xn_tr.shape[1], cat_cardinalities=cat_card,
        d_embedding=CFG["d_embedding"], d_layers=CFG["d_layers"],
        dropout=CFG["dropout"],
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    log_path = LOG_DIR / "mlp_adult.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    ckpt_path = CKPT_DIR / "mlp_adult_best.pt"
    best_val_acc = -1.0
    best_epoch = -1
    patience_left = CFG["patience"]
    t0 = time.time()

    for epoch in range(1, CFG["max_epochs"] + 1):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for xn, xc, y in train_loader:
            xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            logits = model(xn, xc)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * y.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            tot_n += y.size(0)
        tr_loss, tr_acc = tot_loss / tot_n, tot_correct / tot_n

        # val
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        with torch.no_grad():
            for xn, xc, y in val_loader:
                xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
                logits = model(xn, xc)
                v_loss += criterion(logits, y).item() * y.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_n += y.size(0)
        val_loss, val_acc = v_loss / v_n, v_correct / v_n

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

        print(f"[mlp/adult] ep {epoch}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            patience_left = CFG["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[mlp/adult] early stop at epoch {epoch} (best ep {best_epoch})", flush=True)
                break

    # Load best, eval test
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    t_correct, t_n, tr_correct, tr_n = 0, 0, 0, 0
    with torch.no_grad():
        for xn, xc, y in test_loader:
            xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            t_correct += (model(xn, xc).argmax(1) == y).sum().item()
            t_n += y.size(0)
        for xn, xc, y in train_loader:
            xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            tr_correct += (model(xn, xc).argmax(1) == y).sum().item()
            tr_n += y.size(0)
    test_acc, train_acc = t_correct / t_n, tr_correct / tr_n
    wall = time.time() - t0

    res = VoterResult(
        voter_name="mlp", dataset="adult",
        best_val_acc=best_val_acc, test_acc=test_acc, train_acc=train_acc,
        best_iter_or_epoch=best_epoch, total_wall_sec=wall,
        config=CFG, notes="RTDL tuned per-Adult: AdamW, quantile-norm, embeddings, early-stop patience=16",
    )
    print_result(res)
    res.save()


if __name__ == "__main__":
    main()
