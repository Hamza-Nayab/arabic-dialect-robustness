# -*- coding: utf-8 -*-
"""Shared full 12-layer fine-tuning engine (GPU, FP16 AMP) with atomic, resumable per-run bookkeeping.

Used by 05_finetune_transfer_fullft.py (binary grid) and 05b_three_class_replication.py (3-class).
"""
from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
BATCH_SIZE = 16
EVAL_BATCH = 64
MAX_LEN = 128
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
CLIP = 1.0


class Tokenized:
    """Pre-tokenized texts (no padding) so every run only pads per batch."""

    def __init__(self, tokenizer, texts):
        enc = tokenizer([str(t) for t in texts], truncation=True, max_length=MAX_LEN)
        self.ids = enc["input_ids"]
        self.lengths = np.array([len(x) for x in self.ids])
        self.pad_id = tokenizer.pad_token_id

    def batch(self, idx):
        L = int(max(self.lengths[i] for i in idx))
        x = torch.full((len(idx), L), self.pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            s = self.ids[i]
            x[r, : len(s)] = torch.tensor(s)
            m[r, : len(s)] = 1
        return x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)


def bucketed_batches(lengths, rng, bs):
    idx = rng.permutation(len(lengths))
    chunk = bs * 50
    out = []
    for c in range(0, len(idx), chunk):
        part = idx[c : c + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        out.extend(part[i : i + bs] for i in range(0, len(part), bs))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


@torch.no_grad()
def predict(model, tok: Tokenized, rows) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows)
    order = rows[np.argsort(tok.lengths[rows], kind="stable")]
    pred = {}
    for i in range(0, len(order), EVAL_BATCH):
        b = order[i : i + EVAL_BATCH]
        x, m = tok.batch(b)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            logits = model(input_ids=x, attention_mask=m).logits
        for j, p in zip(b, logits.argmax(-1).cpu().numpy()):
            pred[int(j)] = int(p)
    return np.array([pred[int(r)] for r in rows], dtype=int)


def train_one(ckpt, tok_tr: Tokenized, tr_rows, y_tr, tok_dv: Tokenized, dv_rows, y_dv, lr, seed, epochs,
              num_labels=2, class_weighted=False, max_steps=None):
    """Full fine-tuning: every parameter trainable; best-dev-macro-F1 epoch restored."""
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=num_labels).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = True
    tr_rows, y_tr = np.asarray(tr_rows), np.asarray(y_tr, dtype=int)
    if class_weighted:
        counts = np.bincount(y_tr, minlength=num_labels)
        w = len(y_tr) / (num_labels * np.maximum(counts, 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(len(tr_rows) / BATCH_SIZE))
    total = epochs * steps_per_epoch if max_steps is None else min(max_steps, epochs * steps_per_epoch)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    rng = np.random.default_rng(seed)
    lengths = tok_tr.lengths[tr_rows]
    best_f1, best_state, best_epoch, step = -1.0, None, -1, 0
    for epoch in range(epochs):
        model.train()
        for b in bucketed_batches(lengths, rng, BATCH_SIZE):
            x, m = tok_tr.batch(tr_rows[b])
            y = torch.tensor(y_tr[b], dtype=torch.long, device=DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        dv_pred = predict(model, tok_dv, dv_rows)
        f1 = float(f1_score(y_dv, dv_pred, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if max_steps is not None and step >= max_steps:
            break
    model.load_state_dict(best_state)
    return model, best_f1, best_epoch


def free(model=None):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------- atomic bookkeeping ----------------
class RunStore:
    """Resumable store: a run counts as done only if its JSON line AND all its prediction files exist."""

    def __init__(self, out_dir: Path, jsonl_name: str, read_names: list[str], backup_every=10,
                 backup_path: Path | None = None):
        self.out = out_dir
        self.preds = out_dir / "raw_preds"
        self.preds.mkdir(parents=True, exist_ok=True)
        self.jsonl = out_dir / jsonl_name
        self.done = {}
        for name in read_names:
            p = out_dir / name
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # torn last line from a hard kill
                        if all((self.preds / f).exists() for f in r.get("pred_files", [])):
                            self.done[r["run_key"]] = r
        self.n_new = 0
        self.backup_every = backup_every
        self.backup_path = backup_path

    def is_done(self, key):
        return key in self.done

    def save_pred(self, name, arr):
        tmp = self.preds / (name + ".tmp.npy")
        np.save(tmp, np.asarray(arr, dtype=int))
        os.replace(tmp, self.preds / name)

    def append(self, rec):
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done[rec["run_key"]] = rec
        self.n_new += 1
        if self.backup_path is not None and self.n_new % self.backup_every == 0:
            self.backup()

    def backup(self):
        if self.backup_path is None:
            return
        tmp = Path(str(self.backup_path) + ".tmp")
        with tarfile.open(tmp, "w:gz") as t:
            for p in sorted(self.out.glob("*.jsonl")) + sorted(self.out.glob("selected_lrs*.json")):
                t.add(p, arcname=f"outputs_v2/{p.name}")
            t.add(self.preds, arcname="outputs_v2/raw_preds")
        os.replace(tmp, self.backup_path)


class Deadline:
    def __init__(self, seconds_from_start: float | None):
        self.t_end = None if not seconds_from_start else time.time() + seconds_from_start

    def expired(self):
        return self.t_end is not None and time.time() > self.t_end
