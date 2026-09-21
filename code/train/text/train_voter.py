"""Train a transformer voter on clean AG-News, save the checkpoint.

Usage:
  python train_voter.py --voter distilbert --gpu 0
  python train_voter.py --voter roberta-base --gpu 1
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data' / 'ag_news'
VOTERS = ROOT / 'voters'
VOTERS.mkdir(exist_ok=True)


def voter_to_hf(name):
    return {
        'distilbert':   'distilbert-base-cased',
        'roberta-base': 'roberta-base',
    }[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voter', required=True, choices=['distilbert', 'roberta-base'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()

    import os; os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(0); np.random.seed(0)

    hf = voter_to_hf(args.voter)
    print(f'[train] voter={args.voter} hf={hf} gpu={args.gpu}', flush=True)
    t0 = time.time()

    ds = load_from_disk(str(DATA))
    # Gu-style: voters train on CLT (clean-label train) only — never see NLT.
    clt_idx = np.load(ROOT / 'data' / 'splits' / 'CLT_indices.npy')
    clv_idx = np.load(ROOT / 'data' / 'splits' / 'CLV_indices.npy')
    train = ds['train'].select(clt_idx.tolist()).shuffle(seed=0)
    val   = ds['train'].select(clv_idx.tolist())
    test  = ds['test']
    n_classes = 4
    print(f'[train] CLT={len(train)} CLV={len(val)} held-out-test={len(test)}', flush=True)

    tok = AutoTokenizer.from_pretrained(hf)
    def encode(batch):
        return tok(batch['text'], truncation=True, max_length=128)
    train = train.map(encode, batched=True)
    test  = test.map(encode, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=n_classes)
    out_dir = VOTERS / args.voter
    out_dir.mkdir(exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out_dir / 'trainer'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=200,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='none',
        seed=0,
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {'accuracy': float((preds == eval_pred.label_ids).mean())}

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train, eval_dataset=test,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    final_eval = trainer.evaluate()
    print(f'[train] final test acc: {final_eval["eval_accuracy"]:.4f}', flush=True)

    # Save the final model (not just the trainer dir) for easy reload.
    final_dir = out_dir / 'final'
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    (out_dir / 'train_summary.json').write_text(json.dumps({
        'voter': args.voter, 'hf': hf,
        'epochs': args.epochs, 'lr': args.lr, 'bs': args.bs,
        'final_test_acc': final_eval['eval_accuracy'],
        'elapsed_sec': time.time() - t0,
    }, indent=2))
    print(f'[train] DONE  saved to {final_dir}', flush=True)


if __name__ == '__main__':
    main()
