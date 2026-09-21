"""FT-Transformer voter on Adult — RTDL per-Adult Optuna-tuned recipe.

Source: yandex-research/rtdl-revisiting-models output/adult/ft_transformer/tuned/0.toml.
Architecture: per-feature learned token (numerical via linear+bias, categorical
via embedding), [CLS] token prepended, 3-layer prenorm Transformer with ReGLU,
linear head on [CLS]. AdamW lr=2.67e-5, no LR schedule (RTDL paper §4).
"""
import os, sys, time, csv, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from common import (load_adult, split_train_val, VoterResult, print_result,
                    CATEGORICAL_COLS, NUMERICAL_COLS, TRAIN_SEED, LOG_DIR, CKPT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = dict(
    optimizer="AdamW", lr=2.67e-5, weight_decay=1.88e-5, batch_size=256,
    n_layers=3, n_heads=8, d_token=352, d_ffn_factor=2.04,
    attention_dropout=0.290, ffn_dropout=0.161, residual_dropout=0.0,
    activation="reglu", prenormalization=True,
    normalization="quantile", patience=16, max_epochs=200,
)


def prepare_adult():
    X_train, y_train, X_test, y_test = load_adult()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

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

    qt = QuantileTransformer(output_distribution="normal", random_state=TRAIN_SEED)
    Xn_tr = qt.fit_transform(Xn_tr).astype(np.float32)
    Xn_val = qt.transform(Xn_val).astype(np.float32)
    Xn_test = qt.transform(Xn_test).astype(np.float32)

    cat_cardinalities = [len(cat_maps[c]) for c in CATEGORICAL_COLS]
    return (Xn_tr, Xc_tr, y_tr.values), (Xn_val, Xc_val, y_val.values), \
           (Xn_test, Xc_test, y_test.values), cat_cardinalities


class FeatureTokenizer(nn.Module):
    """Per-feature learnable token: numerical -> Linear(1, d_token) + bias,
    categorical -> Embedding(cardinality, d_token)."""
    def __init__(self, n_num: int, cat_cardinalities, d_token: int):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.num_bias, a=math.sqrt(5))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )

    def forward(self, x_num, x_cat):
        # x_num: (B, n_num), x_cat: (B, n_cat)
        # numeric tokens: x_num[..., None] broadcasts against num_weight (n_num, d_token)
        num_tok = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        cat_tok = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        return torch.cat([num_tok, cat_tok], dim=1)  # (B, n_num+n_cat, d_token)


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class FTBlock(nn.Module):
    def __init__(self, d_token, n_heads, d_ffn, attn_dropout, ffn_dropout, residual_dropout, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_token)
        # ReGLU has 2x output, gated; so first linear projects to 2*d_ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_token, 2 * d_ffn),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        if self.prenorm:
            h = self.attn_norm(x)
            a, _ = self.attn(h, h, h)
            x = x + self.res_drop(a)
            h = self.ffn_norm(x)
            x = x + self.res_drop(self.ffn(h))
        else:
            a, _ = self.attn(x, x, x)
            x = self.attn_norm(x + self.res_drop(a))
            x = self.ffn_norm(x + self.res_drop(self.ffn(x)))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, cfg, num_classes=2):
        super().__init__()
        d = cfg["d_token"]
        d_ffn = int(d * cfg["d_ffn_factor"])
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([
            FTBlock(d, cfg["n_heads"], d_ffn,
                    cfg["attention_dropout"], cfg["ffn_dropout"], cfg["residual_dropout"],
                    prenorm=cfg["prenormalization"])
            for _ in range(cfg["n_layers"])
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)  # (B, n_feat, d)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)


def main():
    torch.manual_seed(TRAIN_SEED); np.random.seed(TRAIN_SEED)

    (Xn_tr, Xc_tr, y_tr), (Xn_val, Xc_val, y_val), (Xn_test, Xc_test, y_test), cat_card = prepare_adult()

    def to_loader(Xn, Xc, y, shuffle):
        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Xc), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle)

    train_loader = to_loader(Xn_tr, Xc_tr, y_tr, True)
    val_loader = to_loader(Xn_val, Xc_val, y_val, False)
    test_loader = to_loader(Xn_test, Xc_test, y_test, False)

    model = FTTransformer(Xn_tr.shape[1], cat_card, CFG).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    log_path = LOG_DIR / "ft_transformer_adult.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    ckpt_path = CKPT_DIR / "ft_transformer_adult_best.pt"
    best_val_acc, best_epoch, patience_left = -1.0, -1, CFG["patience"]
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
        print(f"[ft/adult] ep {epoch}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
            patience_left = CFG["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[ft/adult] early stop ep {epoch} (best ep {best_epoch})", flush=True)
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    t_correct, t_n, tr_correct, tr_n = 0, 0, 0, 0
    with torch.no_grad():
        for xn, xc, y in test_loader:
            xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            t_correct += (model(xn, xc).argmax(1) == y).sum().item(); t_n += y.size(0)
        for xn, xc, y in train_loader:
            xn, xc, y = xn.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            tr_correct += (model(xn, xc).argmax(1) == y).sum().item(); tr_n += y.size(0)
    test_acc, train_acc = t_correct / t_n, tr_correct / tr_n
    wall = time.time() - t0

    res = VoterResult(
        voter_name="ft_transformer", dataset="adult",
        best_val_acc=best_val_acc, test_acc=test_acc, train_acc=train_acc,
        best_iter_or_epoch=best_epoch, total_wall_sec=wall,
        config=CFG, notes="RTDL tuned per-Adult; FT-Transformer; ReGLU; prenorm; no LR sched",
    )
    print_result(res)
    res.save()


if __name__ == "__main__":
    main()
