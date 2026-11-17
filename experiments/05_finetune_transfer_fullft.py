# -*- coding: utf-8 -*-
"""Step 5 (GPU): full 12-layer fine-tuning, FP16 AMP, per-model LR selection, 5 seeds, full regime grid.

Regimes per (model, seed): PID, PID_weighted, LODO_matched (N=1,531), MSA_only (n=1,905),
PID_n1531 and PID_n1905 (size-matched pooled controls, stratified by dialect x label).

Crash-proof: a run is skipped when its JSON line and all its prediction files exist; predictions are
written atomically, every JSON line is fsynced, and a tar.gz backup is written every 10 runs.

Usage:
  python 05_finetune_transfer_fullft.py --models AraBERTv02 MARBERT --runs_name finetune_runs_gpu0.jsonl
  python 05_finetune_transfer_fullft.py --smoke_test
"""
from __future__ import annotations

import argparse
import json
import os
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
C = SourceFileLoader("fullft_common", str(ROOT / "_fullft_common.py")).load_module()

MAIN_DIALECTS = ["MSA", "Egyptian", "Levantine", "Gulf"]
SEEDS = [42, 43, 44, 45, 46]
LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]
EPOCHS = 4
N_LODO = 1531
REGIMES = ["PID", "PID_weighted", "LODO_matched", "MSA_only", "PID_n1531", "PID_n1905"]
PREFIX = {"PID": "PID", "PID_weighted": "PIDw", "LODO_matched": "LODO", "MSA_only": "MSAonly",
          "PID_n1531": "PIDn1531", "PID_n1905": "PIDn1905"}
MAG_REGIMES = {"PID", "PID_weighted", "MSA_only"}


def stratified_subset(tr: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    strata = tr["dialect"].astype(str) + "_" + tr["label"].astype(str)
    idx, _ = train_test_split(np.arange(len(tr)), train_size=n, random_state=seed, stratify=strata)
    return np.sort(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(step3.MODEL_REGISTRY))
    ap.add_argument("--out_dir", default=str(ROOT / "outputs_v2"))
    ap.add_argument("--runs_name", default="finetune_runs.jsonl")
    ap.add_argument("--deadline_hours", type=float, default=None)
    ap.add_argument("--backup", default=None)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    data_dir = ROOT / "outputs_v2"
    tr = pd.read_csv(data_dir / "train_4.csv")
    dv = pd.read_csv(data_dir / "dev_4.csv")
    te = pd.read_csv(data_dir / "test_4.csv")
    mg = pd.read_csv(data_dir / "test_maghrebi.csv")
    seeds, lr_grid, epochs, max_steps, models = SEEDS, LR_GRID, EPOCHS, None, a.models
    if a.smoke_test:
        # tiny, fast, CPU-friendly dry run of every code path including resume-skipping
        out = Path(os.environ.get("SMOKE_DIR", str(ROOT / "smoke_out")))
        tr = pd.concat([g.sample(12, random_state=0) for _, g in tr.groupby("dialect")]).reset_index(drop=True)
        dv = pd.concat([g.sample(4, random_state=0) for _, g in dv.groupby("dialect")]).reset_index(drop=True)
        seeds, lr_grid, epochs, max_steps, models = [42, 43], [2e-5, 5e-5], 1, 2, ["AraBERTv02"]
    n_lodo = N_LODO if not a.smoke_test else 20
    n_ctrl = {"PID_n1531": 1531, "PID_n1905": 1905} if not a.smoke_test else {"PID_n1531": 20, "PID_n1905": 24}
    out.mkdir(parents=True, exist_ok=True)
    lr_dir = out / "selected_lrs"
    lr_dir.mkdir(exist_ok=True)
    reads = sorted({p.name for p in out.glob("finetune_runs*.jsonl")} | {a.runs_name})
    store = C.RunStore(out, a.runs_name, reads, backup_path=Path(a.backup) if a.backup else None)
    deadline = C.Deadline(a.deadline_hours * 3600 if a.deadline_hours else None)
    y_te, d_te, y_mg = te.label.to_numpy(), te.dialect.to_numpy(), mg.label.to_numpy()
    print(f"[fullft] device={C.DEVICE} amp={C.USE_AMP} models={models} already_done={len(store.done)}", flush=True)

    for model_name in models:
        ckpt = step3.MODEL_REGISTRY[model_name]
        tok = AutoTokenizer.from_pretrained(ckpt)
        T = {k: C.Tokenized(tok, df["text"]) for k, df in [("tr", tr), ("dv", dv), ("te", te), ("mg", mg)]}
        all_tr, all_dv = np.arange(len(tr)), np.arange(len(dv))

        # ---- per-model LR selection on dev_4 (PID, seed 42); saved immediately ----
        lr_file = lr_dir / f"{model_name}.json"
        pid42 = f"{model_name}|PID|42"
        if lr_file.exists():
            sel = json.loads(lr_file.read_text())
        else:
            grid, best = {}, None
            for lr in lr_grid:
                t0 = time.time()
                m, f1, ep = C.train_one(ckpt, T["tr"], all_tr, tr.label, T["dv"], all_dv, dv.label, lr, 42, epochs,
                                        max_steps=max_steps)
                grid[str(lr)] = {"dev_f1": f1, "best_epoch": ep, "seconds": round(time.time() - t0, 1)}
                print(f"  [{model_name}] LR {lr:g}: dev F1={f1:.4f} (epoch {ep}) [{time.time() - t0:.0f}s]", flush=True)
                if best is None or f1 > best[1]:
                    if best is not None:
                        C.free(best[0])
                    best = (m, f1, ep, lr, time.time() - t0)
                else:
                    C.free(m)
            m, f1, ep, lr, secs = best
            sel = {"model": model_name, "selected_lr": lr, "grid": grid}
            # the selected sweep run *is* PID seed 42: record it instead of retraining
            if not store.is_done(pid42):
                p_te, p_mg = C.predict(m, T["te"], np.arange(len(te))), C.predict(m, T["mg"], np.arange(len(mg)))
                f_te, f_mg = f"PID_{model_name}_seed42.npy", f"PIDmag_{model_name}_seed42.npy"
                store.save_pred(f_te, p_te)
                store.save_pred(f_mg, p_mg)
                store.append({"run_key": pid42, "model": model_name, "regime": "PID", "seed": 42, "lr": lr,
                              "finetune": "full", "epochs_max": epochs, "dev_f1": f1, "best_epoch": ep,
                              "train_n": len(tr), "pred_files": [f_te, f_mg],
                              "maghrebi": step2.eval_maghrebi_exploratory(y_mg, p_mg),
                              **step2.compute_robustness_profile(y_te, p_te, d_te), "seconds": round(secs, 1)})
            C.free(m)
            tmp = lr_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(sel, indent=2))
            os.replace(tmp, lr_file)
        lr = float(sel["selected_lr"])
        print(f"  [{model_name}] selected LR = {lr:g}", flush=True)

        for seed in seeds:
            for regime in REGIMES:
                key = f"{model_name}|{regime}|{seed}"
                if store.is_done(key):
                    print(f"  [SKIP - ALREADY COMPLETED] {key}", flush=True)
                    continue
                if deadline.expired():
                    print("[fullft] deadline reached; stopping cleanly", flush=True)
                    store.backup()
                    return
                t0 = time.time()
                rec = {"run_key": key, "model": model_name, "regime": regime, "seed": seed, "lr": lr,
                       "finetune": "full", "epochs_max": epochs}
                tag = PREFIX[regime]
                files = []
                if regime in ("PID", "PID_weighted", "PID_n1531", "PID_n1905"):
                    rows = all_tr if regime in ("PID", "PID_weighted") else stratified_subset(tr, n_ctrl[regime], seed)
                    m, f1, ep = C.train_one(ckpt, T["tr"], rows, tr.label.to_numpy()[rows], T["dv"], all_dv, dv.label,
                                            lr, seed, epochs, class_weighted=(regime == "PID_weighted"), max_steps=max_steps)
                    p_te = C.predict(m, T["te"], np.arange(len(te)))
                    rec.update(dev_f1=f1, best_epoch=ep, train_n=int(len(rows)))
                    if regime in MAG_REGIMES:
                        p_mg = C.predict(m, T["mg"], np.arange(len(mg)))
                        rec["maghrebi"] = step2.eval_maghrebi_exploratory(y_mg, p_mg)
                        files.append(f"{tag}mag_{model_name}_seed{seed}.npy")
                        store.save_pred(files[-1], p_mg)
                    C.free(m)
                elif regime == "MSA_only":
                    rows = np.where(tr.dialect.to_numpy() == "MSA")[0]
                    drows = np.where(dv.dialect.to_numpy() == "MSA")[0]
                    m, f1, ep = C.train_one(ckpt, T["tr"], rows, tr.label.to_numpy()[rows], T["dv"], drows,
                                            dv.label.to_numpy()[drows], lr, seed, epochs, max_steps=max_steps)
                    p_te = C.predict(m, T["te"], np.arange(len(te)))
                    p_mg = C.predict(m, T["mg"], np.arange(len(mg)))
                    rec.update(dev_f1=f1, best_epoch=ep, train_n=int(len(rows)),
                               maghrebi=step2.eval_maghrebi_exploratory(y_mg, p_mg))
                    files.append(f"{tag}mag_{model_name}_seed{seed}.npy")
                    store.save_pred(files[-1], p_mg)
                    C.free(m)
                elif regime == "LODO_matched":
                    p_te = np.zeros(len(te), dtype=int)
                    rec["lodo_targets"] = {}
                    for target in MAIN_DIALECTS:
                        pool = tr[tr.dialect != target]
                        rows = np.sort(pool.sample(n=n_lodo, random_state=seed).index.to_numpy())
                        drows = np.where(dv.dialect.to_numpy() != target)[0]
                        m, f1, ep = C.train_one(ckpt, T["tr"], rows, tr.label.to_numpy()[rows], T["dv"], drows,
                                                dv.label.to_numpy()[drows], lr, seed, epochs, max_steps=max_steps)
                        tmask = np.where(d_te == target)[0]
                        p_te[tmask] = C.predict(m, T["te"], tmask)
                        rec["lodo_targets"][target] = {"dev_f1": f1, "best_epoch": ep, "train_n": int(len(rows))}
                        C.free(m)
                f_te = f"{tag}_{model_name}_seed{seed}.npy"
                store.save_pred(f_te, p_te)
                files.insert(0, f_te)
                rec["pred_files"] = files
                rec.update(step2.compute_robustness_profile(y_te, p_te, d_te))
                rec["seconds"] = round(time.time() - t0, 1)
                store.append(rec)
                print(f"  {key}: pooled F1={rec['pooled_macro_f1']:.4f} HM={rec['hm_f1']:.4f} "
                      f"WD={rec['wd_f1']:.4f} [{rec['seconds']:.0f}s]", flush=True)
    store.backup()
    print("\n[STEP 5 FULL-FT PASSED] all requested (model, regime, seed) runs present.", flush=True)


if __name__ == "__main__":
    main()
