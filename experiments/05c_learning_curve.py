# -*- coding: utf-8 -*-
"""Step 5c (GPU): target-dialect learning curve on top of the size-matched LODO training set.

For each target dialect T in {Egyptian, Levantine, Gulf} and k in {25, 50, 100, 200, all}:
  train = the exact LODO_matched sample used in Step 5 (1,531 non-T tweets, same seed)
          + the first k tweets of a seeded, label-stratified permutation of T's training tweets (nested in k)
  dev   = non-T dev tweets (identical to LODO_matched, so k is the only thing that changes)
  test  = T's test tweets
k = 0 is the LODO_matched run of Step 5 and is not retrained. k values >= |T_train| collapse to "all".

Usage: python 05c_learning_curve.py --models AraBERTv02 MARBERT --runs_name lc_runs_gpu0.jsonl
"""
from __future__ import annotations

import argparse
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
C = SourceFileLoader("fullft_common", str(ROOT / "_fullft_common.py")).load_module()
S5 = SourceFileLoader("step5", str(ROOT / "05_finetune_transfer_fullft.py")).load_module()

CORE = ["AraBERTv02", "MARBERT", "CAMeLBERT-Mix", "XLM-R-Base"]
TARGETS = ["Egyptian", "Levantine", "Gulf"]
K_GRID = [25, 50, 100, 200]


def nested_target_order(tr: pd.DataFrame, target: str, seed: int) -> np.ndarray:
    """Seeded permutation of target rows, interleaving labels so every prefix is ~label-stratified."""
    rng = np.random.default_rng(seed + 1000)
    rows = np.where(tr.dialect.to_numpy() == target)[0]
    by_lab = {l: list(rng.permutation(rows[tr.label.to_numpy()[rows] == l])) for l in (0, 1)}
    p_neg = len(by_lab[0]) / len(rows)
    out, c = [], {0: 0.0, 1: 0.0}
    while by_lab[0] or by_lab[1]:
        # pick the label whose share in the prefix is furthest below its overall share
        want0 = p_neg * (len(out) + 1) - c[0]
        lab = 0 if (by_lab[0] and (want0 >= 0.5 or not by_lab[1])) else 1
        out.append(by_lab[lab].pop(0))
        c[lab] += 1
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=CORE)
    ap.add_argument("--out_dir", default=str(ROOT / "outputs_v2"))
    ap.add_argument("--runs_name", default="lc_runs.jsonl")
    ap.add_argument("--deadline_hours", type=float, default=None)
    ap.add_argument("--backup", default=None)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()

    data = ROOT / "outputs_v2"
    tr, dv, te = (pd.read_csv(data / f) for f in ["train_4.csv", "dev_4.csv", "test_4.csv"])
    out = Path(a.out_dir)
    seeds, epochs, max_steps, n_lodo, k_grid, models = S5.SEEDS, S5.EPOCHS, None, S5.N_LODO, K_GRID, a.models
    if a.smoke_test:
        import os
        out = Path(os.environ.get("SMOKE_DIR", str(ROOT / "smoke_out")))
        seeds, epochs, max_steps, n_lodo, k_grid, models = [42], 1, 2, 20, [5], ["AraBERTv02"]
    out.mkdir(parents=True, exist_ok=True)
    reads = sorted({p.name for p in out.glob("lc_runs*.jsonl")} | {a.runs_name})
    store = C.RunStore(out, a.runs_name, reads, backup_path=Path(a.backup) if a.backup else None)
    deadline = C.Deadline(a.deadline_hours * 3600 if a.deadline_hours else None)
    import json
    lrs = json.loads((data / "selected_lrs.json").read_text())
    d_te = te.dialect.to_numpy()
    print(f"[lc] device={C.DEVICE} models={models} done={len(store.done)}", flush=True)

    for model_name in models:
        ckpt = step3.MODEL_REGISTRY[model_name]
        lr = float(lrs[model_name]["selected_lr"]) if model_name in lrs else 2e-5
        tok = AutoTokenizer.from_pretrained(ckpt)
        T = {k: C.Tokenized(tok, df["text"]) for k, df in [("tr", tr), ("dv", dv), ("te", te)]}
        for seed in seeds:
            for target in TARGETS:
                pool = tr[tr.dialect != target]
                base = np.sort(pool.sample(n=n_lodo, random_state=seed).index.to_numpy())  # == Step 5 LODO rows
                order = nested_target_order(tr, target, seed)
                n_all = len(order)
                ks = sorted({min(k, n_all) for k in k_grid} | {n_all})
                drows = np.where(dv.dialect.to_numpy() != target)[0]
                tmask = np.where(d_te == target)[0]
                for k in ks:
                    ktag = "all" if k == n_all else str(k)
                    key = f"{model_name}|LC|{target}|{ktag}|{seed}"
                    if store.is_done(key):
                        continue
                    if deadline.expired():
                        print("[lc] deadline reached", flush=True)
                        store.backup()
                        return
                    t0 = time.time()
                    rows = np.concatenate([base, order[:k]])
                    m, f1, ep = C.train_one(ckpt, T["tr"], rows, tr.label.to_numpy()[rows], T["dv"], drows,
                                            dv.label.to_numpy()[drows], lr, seed, epochs, max_steps=max_steps)
                    p = C.predict(m, T["te"], tmask)
                    C.free(m)
                    fname = f"LC_{model_name}_{target}_k{ktag}_seed{seed}.npy"
                    store.save_pred(fname, p)
                    y = te.label.to_numpy()[tmask]
                    rec = {"run_key": key, "model": model_name, "target": target, "k": int(k), "k_tag": ktag,
                           "seed": seed, "lr": lr, "train_n": int(len(rows)), "dev_f1": f1, "best_epoch": ep,
                           "target_f1": float(f1_score(y, p, average="macro", zero_division=0)),
                           "target_acc": float((y == p).mean()), "pred_files": [fname],
                           "seconds": round(time.time() - t0, 1)}
                    store.append(rec)
                    print(f"  {key}: F1={rec['target_f1']:.4f} [{rec['seconds']:.0f}s]", flush=True)
    store.backup()
    print("[STEP 5c PASSED]", flush=True)


if __name__ == "__main__":
    main()
