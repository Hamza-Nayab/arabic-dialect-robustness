# -*- coding: utf-8 -*-
"""Step 5d (GPU): how much do the duplicated / leaked official ArSarcasm splits inflate scores?

"Leaky" protocol = what a user of the official release gets without our audit: the official train split
(4 main dialects, duplicates kept, including tweets whose normalized text also occurs in test), a stratified
85/15 dev split (seed 42, same rule as Step 1), and the official test split with its duplicates.
Each leaky model is evaluated on (a) the raw official test (as previously reported) and (b) our clean test_4,
so the inflation can be split into "contaminated training" and "duplicated test items".
Clean-protocol numbers are the Step 5 PID runs (same model, LR and seeds).

Usage: python 05d_leaky_splits.py --models AraBERTv02 MARBERT --runs_name leaky_runs_gpu0.jsonl
"""
from __future__ import annotations

import argparse
import json
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
S5 = SourceFileLoader("step5", str(ROOT / "05_finetune_transfer_fullft.py")).load_module()
CORE = ["AraBERTv02", "MARBERT", "CAMeLBERT-Mix", "XLM-R-Base"]


def leaky_frames():
    raw = pd.read_csv(ROOT / "outputs" / "arsarcasm_binary.csv")
    raw["tweet_id"] = np.arange(len(raw))
    raw = raw[raw.dialect.isin(S5.MAIN_DIALECTS)].copy()
    trp, te = raw[raw.split == "train"], raw[raw.split == "test"].reset_index(drop=True)
    strata = trp.dialect.astype(str) + "_" + trp.label.astype(str)
    tr, dv = train_test_split(trp, test_size=0.15, random_state=42, stratify=strata)
    return tr.reset_index(drop=True), dv.reset_index(drop=True), te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=CORE)
    ap.add_argument("--out_dir", default=str(ROOT / "outputs_v2"))
    ap.add_argument("--runs_name", default="leaky_runs.jsonl")
    ap.add_argument("--deadline_hours", type=float, default=None)
    ap.add_argument("--backup", default=None)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    tr, dv, te_raw = leaky_frames()
    te_clean = pd.read_csv(ROOT / "outputs_v2" / "test_4.csv")
    out, seeds, epochs, max_steps, models = Path(a.out_dir), S5.SEEDS, S5.EPOCHS, None, a.models
    if a.smoke_test:
        import os
        out = Path(os.environ.get("SMOKE_DIR", str(ROOT / "smoke_out")))
        tr, dv = tr.sample(40, random_state=0).reset_index(drop=True), dv.sample(16, random_state=0).reset_index(drop=True)
        seeds, epochs, max_steps, models = [42], 1, 2, ["AraBERTv02"]
    out.mkdir(parents=True, exist_ok=True)
    reads = sorted({p.name for p in out.glob("leaky_runs*.jsonl")} | {a.runs_name})
    store = C.RunStore(out, a.runs_name, reads, backup_path=Path(a.backup) if a.backup else None)
    deadline = C.Deadline(a.deadline_hours * 3600 if a.deadline_hours else None)
    lrs = json.loads((ROOT / "outputs_v2" / "selected_lrs.json").read_text())
    for model_name in models:
        ckpt = step3.MODEL_REGISTRY[model_name]
        lr = float(lrs[model_name]["selected_lr"])
        tok = AutoTokenizer.from_pretrained(ckpt)
        T = {k: C.Tokenized(tok, df["text"]) for k, df in [("tr", tr), ("dv", dv), ("tr_raw", te_raw), ("tc", te_clean)]}
        for seed in seeds:
            key = f"{model_name}|LEAKY|{seed}"
            if store.is_done(key):
                continue
            if deadline.expired():
                store.backup()
                return
            t0 = time.time()
            m, f1, ep = C.train_one(ckpt, T["tr"], np.arange(len(tr)), tr.label, T["dv"], np.arange(len(dv)), dv.label,
                                    lr, seed, epochs, max_steps=max_steps)
            p_raw = C.predict(m, T["tr_raw"], np.arange(len(te_raw)))
            p_cln = C.predict(m, T["tc"], np.arange(len(te_clean)))
            C.free(m)
            f_raw, f_cln = f"LEAKYraw_{model_name}_seed{seed}.npy", f"LEAKYclean_{model_name}_seed{seed}.npy"
            store.save_pred(f_raw, p_raw)
            store.save_pred(f_cln, p_cln)
            rec = {"run_key": key, "model": model_name, "seed": seed, "lr": lr, "train_n": len(tr), "dev_f1": f1,
                   "best_epoch": ep, "pred_files": [f_raw, f_cln],
                   "raw_test": step2.compute_robustness_profile(te_raw.label.to_numpy(), p_raw, te_raw.dialect.to_numpy()),
                   "clean_test": step2.compute_robustness_profile(te_clean.label.to_numpy(), p_cln,
                                                                  te_clean.dialect.to_numpy()),
                   "seconds": round(time.time() - t0, 1)}
            store.append(rec)
            print(f"  {key}: raw pooled={rec['raw_test']['pooled_macro_f1']:.4f} "
                  f"clean pooled={rec['clean_test']['pooled_macro_f1']:.4f} [{rec['seconds']:.0f}s]", flush=True)
    store.backup()
    print("[STEP 5d PASSED]", flush=True)


if __name__ == "__main__":
    main()
