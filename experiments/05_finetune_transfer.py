# -*- coding: utf-8 -*-
"""Step 5: multi-seed fine-tuning across PID, PID_weighted, LODO (size-matched) and MSA->Dialect.

CPU-feasible protocol (see 05_finetune_transfer_fullft.py for the GPU full fine-tuning variant):
the bottom (L - TOP_K) transformer blocks and embeddings are frozen, their outputs are computed once
per checkpoint by _encode_cache.py, and only the top TOP_K blocks + pooler + classification head are
trained. Every seed/regime/target-dialect run re-initialises the trainable blocks from the pretrained
checkpoint and draws a fresh classification head with that seed.

Runs are appended to outputs_v2/finetune_runs.jsonl as they finish, so the script is resumable.
"""
from __future__ import annotations

import json
import os
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
from transformers.masking_utils import create_bidirectional_mask

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
PREDS_DIR = OUT / "raw_preds"
PREDS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_JSONL = OUT / "finetune_runs.jsonl"

step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
cache_mod = SourceFileLoader("cache_mod", str(ROOT / "_encode_cache.py")).load_module()

MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]
SEEDS = [42, 43, 44, 45, 46]
# CPU budget: 5 seeds for the primary PID regime, 3 seeds for ablation / transfer regimes
REGIME_SEEDS = json.loads(os.environ.get("FT_REGIME_SEEDS", "null")) or {
    "PID": SEEDS,
    "PID_weighted": SEEDS[:3],
    "LODO_matched": SEEDS[:3],
    "MSA_only": SEEDS[:3],
    # Size-matched pooled controls: same training size as LODO (1,531) and MSA-only (1,905)
    "PID_n1531": SEEDS[:3],
    "PID_n1905": SEEDS[:3],
}
EPOCHS = int(os.environ.get("FT_EPOCHS", "3"))
BATCH_SIZE = 16
N_LODO_MATCHED = 1531
LR_GRID = [2e-5, 5e-5, 1e-4]
TOP_K = cache_mod.CACHE_LAYERS_FROM_TOP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which models run which regimes (core encoders get every regime; variants get PID + MSA_only)
MODEL_PLAN = json.loads(os.environ.get("FT_PLAN", "null")) or {
    "AraBERTv02": ["PID", "PID_weighted", "LODO_matched", "MSA_only"],
    "MARBERT": ["PID", "PID_weighted", "LODO_matched", "MSA_only"],
    "CAMeLBERT-Mix": ["PID", "PID_weighted", "LODO_matched", "MSA_only"],
    "CAMeLBERT-MSA": ["PID", "MSA_only"],
    "CAMeLBERT-DA": ["PID", "MSA_only"],
    "XLM-R-Base": ["PID", "MSA_only"],
}


class TopBlocksClassifier(nn.Module):
    """Top-K transformer blocks + original pooler/classification head of a *ForSequenceClassification model."""

    def __init__(self, seq_model: nn.Module, start_layer: int):
        super().__init__()
        base = seq_model.base_model
        self.config = base.config
        self.layers = nn.ModuleList(list(base.encoder.layer[start_layer:]))
        self.is_bert = hasattr(seq_model, "bert")
        if self.is_bert:
            self.pooler = base.pooler
            self.dropout = seq_model.dropout
        self.classifier = seq_model.classifier

    def forward(self, h: torch.Tensor, mask2d: torch.Tensor) -> torch.Tensor:
        m = create_bidirectional_mask(config=self.config, inputs_embeds=h, attention_mask=mask2d)
        for layer in self.layers:
            h = layer(h, m)
        if self.is_bert:
            return self.classifier(self.dropout(self.pooler(h)))
        return self.classifier(h)  # RoBERTa head consumes the sequence output


def pad_batch(states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(s.shape[0] for s in states)
    H = states[0].shape[1]
    x = torch.zeros(len(states), L, H, dtype=torch.float32)
    mask = torch.zeros(len(states), L, dtype=torch.long)
    for i, s in enumerate(states):
        x[i, : s.shape[0]] = s.float()
        mask[i, : s.shape[0]] = 1
    return x.to(DEVICE), mask.to(DEVICE)


def bucketed_batches(lengths: np.ndarray, rng: np.random.Generator, bs: int) -> list[np.ndarray]:
    """Shuffle, sort within chunks of 50 batches by length, then shuffle batch order."""
    idx = rng.permutation(len(lengths))
    chunk = bs * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = idx[c : c + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches.extend(part[i : i + bs] for i in range(0, len(part), bs))
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


@torch.no_grad()
def predict(model: TopBlocksClassifier, states: list[torch.Tensor]) -> np.ndarray:
    model.eval()
    lengths = np.array([s.shape[0] for s in states])
    order = np.argsort(lengths, kind="stable")
    preds = np.zeros(len(states), dtype=int)
    for i in range(0, len(order), 64):
        idx = order[i : i + 64]
        x, m = pad_batch([states[j] for j in idx])
        preds[idx] = model(x, m).argmax(-1).cpu().numpy()
    return preds


def train_one(ckpt, start_layer, tr_states, tr_y, dv_states, dv_y, lr, seed, class_weighted=False):
    set_seed(seed)
    seq_model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
    model = TopBlocksClassifier(seq_model, start_layer).to(DEVICE)
    del seq_model
    tr_y = np.asarray(tr_y, dtype=int)
    if class_weighted:
        counts = np.bincount(tr_y, minlength=2)
        w = len(tr_y) / (2.0 * np.maximum(counts, 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lengths = np.array([s.shape[0] for s in tr_states])
    rng = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(tr_states) / BATCH_SIZE))
    total = EPOCHS * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    best_f1, best_state, best_epoch = -1.0, None, -1
    for epoch in range(EPOCHS):
        model.train()
        for b in bucketed_batches(lengths, rng, BATCH_SIZE):
            x, m = pad_batch([tr_states[i] for i in b])
            y = torch.tensor(tr_y[b], dtype=torch.long, device=DEVICE)
            loss = loss_fn(model(x, m), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        dv_pred = predict(model, dv_states)
        f1 = float(f1_score(dv_y, dv_pred, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_f1, best_epoch


def load_done() -> dict[str, dict]:
    done = {}
    if RUNS_JSONL.exists():
        for line in RUNS_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["run_key"]] = r
    return done


def append_run(rec: dict) -> None:
    with open(RUNS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tr = pd.read_csv(OUT / "train_4.csv")
    dv = pd.read_csv(OUT / "dev_4.csv")
    te = pd.read_csv(OUT / "test_4.csv")
    mg = pd.read_csv(OUT / "test_maghrebi.csv")
    y_te, d_te = te["label"].to_numpy(), te["dialect"].to_numpy()
    done = load_done()
    lr_log_path = OUT / "finetune_lr_selection.json"
    lr_log = json.loads(lr_log_path.read_text()) if lr_log_path.exists() else {}

    for model_name, regimes in MODEL_PLAN.items():
        ckpt = step3.MODEL_REGISTRY[model_name]
        cache_mod.encode_and_cache(model_name, ckpt)
        cache = torch.load(cache_mod.HID_DIR / f"{model_name}.pt")
        start_layer = cache["cache_layer"]
        S = {k: cache[k] for k in ["tr", "dv", "te", "mg"]}
        print("=" * 70, f"\n{model_name} ({ckpt}); training blocks {start_layer}..{cache['n_layers'] - 1}", flush=True)

        # Learning-rate grid on dev_4 (PID, seed 42). Without FT_FIXED_LR the selected configuration *is* the
        # PID seed-42 run and is recorded directly; with FT_FIXED_LR the grid is only a sensitivity check.
        pid42_key = f"{model_name}|PID|42"
        best = None
        if model_name not in lr_log:
            grid, best = {}, None
            for lr in LR_GRID:
                t0 = time.time()
                m, f1, ep = train_one(ckpt, start_layer, S["tr"], tr["label"], S["dv"], dv["label"], lr, 42)
                grid[str(lr)] = {"dev_f1": f1, "best_epoch": ep}
                print(f"  LR {lr:g}: dev macro-F1={f1:.4f} (epoch {ep}) [{time.time() - t0:.0f}s]", flush=True)
                if best is None or f1 > best[1]:
                    best = (m, f1, ep, lr, time.time() - t0)
                else:
                    del m
            m, f1, ep, best_lr, secs = best
            lr_log[model_name] = {"selected_lr": best_lr, "grid": grid}
            lr_log_path.write_text(json.dumps(lr_log, indent=2))
        lr = float(os.environ.get("FT_FIXED_LR", lr_log[model_name]["selected_lr"]))
        print(f"  Learning rate used for all runs = {lr:g} (dev-selected: {lr_log[model_name]['selected_lr']:g})", flush=True)
        if best is not None and best_lr == lr and pid42_key not in done:
            p_te, p_mg = predict(m, S["te"]), predict(m, S["mg"])
            np.save(PREDS_DIR / f"PID_{model_name}_seed42.npy", p_te)
            np.save(PREDS_DIR / f"PIDmag_{model_name}_seed42.npy", p_mg)
            rec = {"run_key": pid42_key, "model": model_name, "regime": "PID", "seed": 42, "lr": best_lr,
                   "top_k_trained_blocks": TOP_K, "epochs_max": EPOCHS, "dev_f1": f1, "best_epoch": ep,
                   "maghrebi": step2.eval_maghrebi_exploratory(mg["label"].to_numpy(), p_mg),
                   **step2.compute_robustness_profile(y_te, p_te, d_te), "seconds": round(secs, 1)}
            append_run(rec)
            done[pid42_key] = rec
            del m, best
        for seed in SEEDS:
            for regime in regimes:
                if seed not in REGIME_SEEDS[regime]:
                    continue
                key = f"{model_name}|{regime}|{seed}"
                if key in done:
                    continue
                t0 = time.time()
                rec = {"run_key": key, "model": model_name, "regime": regime, "seed": seed, "lr": lr,
                       "top_k_trained_blocks": TOP_K, "epochs_max": EPOCHS}
                if regime in ("PID", "PID_weighted"):
                    m, f1, ep = train_one(ckpt, start_layer, S["tr"], tr["label"], S["dv"], dv["label"], lr, seed,
                                          class_weighted=(regime == "PID_weighted"))
                    p_te, p_mg = predict(m, S["te"]), predict(m, S["mg"])
                    rec.update(dev_f1=f1, best_epoch=ep)
                    rec["maghrebi"] = step2.eval_maghrebi_exploratory(mg["label"].to_numpy(), p_mg)
                    tag = "PID" if regime == "PID" else "PIDw"
                    np.save(PREDS_DIR / f"{tag}_{model_name}_seed{seed}.npy", p_te)
                    np.save(PREDS_DIR / f"{tag}mag_{model_name}_seed{seed}.npy", p_mg)
                elif regime == "LODO_matched":
                    p_te = np.zeros(len(te), dtype=int)
                    rec["lodo_targets"] = {}
                    for target in MAIN_DIALECTS:
                        pool = np.where(tr["dialect"].to_numpy() != target)[0]
                        sub = tr.iloc[pool].sample(n=N_LODO_MATCHED, random_state=seed).index.to_numpy()
                        dmask = np.where(dv["dialect"].to_numpy() != target)[0]
                        m, f1, ep = train_one(ckpt, start_layer, [S["tr"][i] for i in sub], tr["label"].to_numpy()[sub],
                                              [S["dv"][i] for i in dmask], dv["label"].to_numpy()[dmask], lr, seed)
                        tmask = d_te == target
                        p_te[tmask] = predict(m, [S["te"][i] for i in np.where(tmask)[0]])
                        rec["lodo_targets"][target] = {"dev_f1": f1, "best_epoch": ep, "train_n": int(len(sub))}
                        del m
                    np.save(PREDS_DIR / f"LODO_{model_name}_seed{seed}.npy", p_te)
                elif regime in ("PID_n1531", "PID_n1905"):
                    n_sub = int(regime.split("_n")[1])
                    sub = tr.sample(n=n_sub, random_state=seed).index.to_numpy()
                    m, f1, ep = train_one(ckpt, start_layer, [S["tr"][i] for i in sub], tr["label"].to_numpy()[sub],
                                          S["dv"], dv["label"], lr, seed)
                    p_te = predict(m, S["te"])
                    rec.update(dev_f1=f1, best_epoch=ep, train_n=n_sub)
                    np.save(PREDS_DIR / f"PIDn{n_sub}_{model_name}_seed{seed}.npy", p_te)
                elif regime == "MSA_only":
                    tmask = np.where(tr["dialect"].to_numpy() == "MSA")[0]
                    dmask = np.where(dv["dialect"].to_numpy() == "MSA")[0]
                    m, f1, ep = train_one(ckpt, start_layer, [S["tr"][i] for i in tmask], tr["label"].to_numpy()[tmask],
                                          [S["dv"][i] for i in dmask], dv["label"].to_numpy()[dmask], lr, seed)
                    p_te, p_mg = predict(m, S["te"]), predict(m, S["mg"])
                    rec.update(dev_f1=f1, best_epoch=ep, train_n=int(len(tmask)))
                    rec["maghrebi"] = step2.eval_maghrebi_exploratory(mg["label"].to_numpy(), p_mg)
                    np.save(PREDS_DIR / f"MSAonly_{model_name}_seed{seed}.npy", p_te)
                else:
                    raise ValueError(regime)
                rec.update(step2.compute_robustness_profile(y_te, p_te, d_te))
                rec["seconds"] = round(time.time() - t0, 1)
                append_run(rec)
                done[key] = rec
                print(f"  {key}: pooled F1={rec['pooled_macro_f1']:.4f} HM-F1={rec['hm_f1']:.4f} "
                      f"WD-F1={rec['wd_f1']:.4f} [{rec['seconds']:.0f}s]", flush=True)
        del cache, S

    runs = list(load_done().values())
    with open(OUT / "finetune_transfer_runs.json", "w", encoding="utf-8") as f:
        json.dump({"lr_log": lr_log, "runs": runs}, f, indent=2)
    print("\n[STEP 5 PASSED] Multi-seed PID / PID_weighted / LODO / MSA-only runs saved.")


if __name__ == "__main__":
    main()
