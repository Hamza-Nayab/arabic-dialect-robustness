# -*- coding: utf-8 -*-
"""Step 5e: replication of the transfer protocol on ArSarcasm-v2 (Abu Farha et al., 2021), binary sentiment.

--prepare (CPU): official v2 train/test -> drop NEU -> same normalization/dedup/leak removal as Step 1
                 -> stratified 85/15 train/dev (seed 42) -> outputs_v2/arsarcasm_v2/*.csv + data_card_v2.json
                 + majority / TF-IDF baselines. Maghrebi is excluded (31 binary train, 0 binary test tweets).
default (GPU):   5 encoders x 5 seeds x {PID, LODO_matched, MSA_only, PID_nLODO, PID_nMSA}
                 (LR = the encoder's Step-5 selected LR; size-matched pooled controls stratified by dialect x label).
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

ROOT = Path(__file__).resolve().parent
S1 = SourceFileLoader("step1", str(ROOT / "01_prepare_splits.py")).load_module()
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()

RAW = ROOT / "outputs" / "arsarcasm_v2_raw"
V2 = ROOT / "outputs_v2" / "arsarcasm_v2"
DMAP = {"msa": "MSA", "egypt": "Egyptian", "levant": "Levantine", "gulf": "Gulf", "magreb": "Maghrebi"}
MAIN = ["MSA", "Egyptian", "Levantine", "Gulf"]
MODELS = ["AraBERTv02", "MARBERT", "MARBERTv2", "CAMeLBERT-Mix", "XLM-R-Base"]
SEEDS = [42, 43, 44, 45, 46]
REGIMES = ["PID", "LODO_matched", "MSA_only", "PID_nLODO", "PID_nMSA"]


def load_v2(name):
    d = pd.read_csv(RAW / name)
    d = d[d.sentiment != "NEU"].copy()
    d["text"] = d.tweet.astype(str)
    d["label"] = (d.sentiment == "POS").astype(int)
    d["dialect"] = d.dialect.map(DMAP)
    d["norm_text"] = d.text.map(S1.normalize_for_dedup)
    return d[d.norm_text.str.len() > 0]


def prepare() -> None:
    V2.mkdir(parents=True, exist_ok=True)
    tr_raw, te_raw = load_v2("training_data.csv"), load_v2("testing_data.csv")
    v1 = pd.read_csv(ROOT / "outputs" / "arsarcasm_binary.csv")
    v1n = set(v1.text.map(S1.normalize_for_dedup))
    cross = len(set(tr_raw.norm_text) & set(te_raw.norm_text))
    te = te_raw.drop_duplicates("norm_text").copy()
    trp = tr_raw[~tr_raw.norm_text.isin(set(te.norm_text))].drop_duplicates("norm_text").copy()
    n_mag_tr, n_mag_te = int((trp.dialect == "Maghrebi").sum()), int((te.dialect == "Maghrebi").sum())
    trp, te = trp[trp.dialect.isin(MAIN)], te[te.dialect.isin(MAIN)].reset_index(drop=True)
    strata = trp.dialect + "_" + trp.label.astype(str)
    tr, dv = train_test_split(trp, test_size=0.15, random_state=42, stratify=strata)
    tr, dv = tr.reset_index(drop=True), dv.reset_index(drop=True)
    for n, df in [("train", tr), ("dev", dv), ("test", te)]:
        df[["text", "label", "dialect", "sarcasm", "norm_text"]].to_csv(V2 / f"{n}.csv", index=False, encoding="utf-8-sig")
    lodo = {d: int((tr.dialect != d).sum()) for d in MAIN}
    card = {"raw_binary_train": int(len(tr_raw)), "raw_binary_test": int(len(te_raw)),
            "raw_cross_split_leak_texts": int(cross),
            "test_overlap_with_v1": int(te.norm_text.isin(v1n).sum()),
            "train_overlap_with_v1": int(tr.norm_text.isin(v1n).sum() + dv.norm_text.isin(v1n).sum()),
            "maghrebi_binary_train": n_mag_tr, "maghrebi_binary_test": n_mag_te,
            "train_total": len(tr), "dev_total": len(dv), "test_total": len(te),
            "lodo_train_sizes": lodo, "n_lodo_matched": min(lodo.values()), "n_msa_only": int((tr.dialect == "MSA").sum()),
            "by_dialect": {n: {d: {"total": int((df.dialect == d).sum()),
                                   "pos": int(((df.dialect == d) & (df.label == 1)).sum())} for d in MAIN}
                           for n, df in [("train", tr), ("dev", dv), ("test", te)]}}
    # classical baselines on v2 (same recipe as Step 2: majority + balanced TF-IDF word+char)
    base = {}
    y_te, d_te = te.label.to_numpy(), te.dialect.to_numpy()
    maj = int(np.bincount(tr.label).argmax())
    base["Majority"] = step2.compute_robustness_profile(y_te, np.full(len(te), maj), d_te)
    # identical recipe to Step 2's balanced TF-IDF (word 1-2 + char_wb 2-5, C grid on dev)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline, make_union
    best = None
    for c in [0.25, 0.5, 1.0, 2.0, 4.0]:
        pipe = make_pipeline(make_union(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                             LogisticRegression(max_iter=2000, C=c, class_weight="balanced", random_state=42))
        pipe.fit(tr.text, tr.label)
        f = f1_score(dv.label, pipe.predict(dv.text), average="macro", zero_division=0)
        if best is None or f > best[0]:
            best = (f, c, pipe)
    p = best[2].predict(te.text)
    card["tfidf_C"] = best[1]
    base["TF-IDF"] = step2.compute_robustness_profile(y_te, np.asarray(p), d_te)
    (V2 / "data_card_v2.json").write_text(json.dumps(card, indent=2))
    (V2 / "classical_v2.json").write_text(json.dumps(base, indent=2))
    (V2 / "preds_classical_v2.json").write_text(json.dumps({"Majority": [maj] * len(te), "TF-IDF": np.asarray(p).tolist()}))
    print(json.dumps(card, indent=2))
    print({k: round(v["pooled_macro_f1"], 4) for k, v in base.items()})
    assert set(tr.norm_text).isdisjoint(te.norm_text) and set(dv.norm_text).isdisjoint(te.norm_text)
    print("[STEP 5e PREPARE PASSED]")


def stratified(df, n, seed):
    s = df.dialect + "_" + df.label.astype(str)
    idx, _ = train_test_split(np.arange(len(df)), train_size=n, random_state=seed, stratify=s)
    return np.sort(idx)


def train(a) -> None:
    C = SourceFileLoader("fullft_common", str(ROOT / "_fullft_common.py")).load_module()
    from transformers import AutoTokenizer
    tr, dv, te = (pd.read_csv(V2 / f"{n}.csv") for n in ["train", "dev", "test"])
    card = json.loads((V2 / "data_card_v2.json").read_text())
    n_lodo, n_msa = card["n_lodo_matched"], card["n_msa_only"]
    out = Path(a.out_dir)
    seeds, epochs, max_steps, models = SEEDS, 4, None, a.models
    if a.smoke_test:
        import os
        out = Path(os.environ.get("SMOKE_DIR", str(ROOT / "smoke_out")))
        tr = pd.concat([g.sample(10, random_state=0) for _, g in tr.groupby("dialect")]).reset_index(drop=True)
        dv = pd.concat([g.sample(4, random_state=0) for _, g in dv.groupby("dialect")]).reset_index(drop=True)
        seeds, epochs, max_steps, models, n_lodo, n_msa = [42], 1, 2, ["AraBERTv02"], 16, 12
    out.mkdir(parents=True, exist_ok=True)
    reads = sorted({p.name for p in out.glob("v2_runs*.jsonl")} | {a.runs_name})
    store = C.RunStore(out, a.runs_name, reads, backup_path=Path(a.backup) if a.backup else None)
    deadline = C.Deadline(a.deadline_hours * 3600 if a.deadline_hours else None)
    lrs = json.loads((ROOT / "outputs_v2" / "selected_lrs.json").read_text())
    y_te, d_te = te.label.to_numpy(), te.dialect.to_numpy()
    toks = {}
    for seed in seeds:  # seeds outermost: a deadline still leaves every model with the same number of seeds
        for model_name in models:
            ckpt = step3.MODEL_REGISTRY[model_name]
            lr = float(lrs[model_name]["selected_lr"])
            if model_name not in toks:
                tok = AutoTokenizer.from_pretrained(ckpt)
                toks[model_name] = {k: C.Tokenized(tok, df["text"]) for k, df in [("tr", tr), ("dv", dv), ("te", te)]}
            T = toks[model_name]
            for regime in REGIMES:
                key = f"{model_name}|V2|{regime}|{seed}"
                if store.is_done(key):
                    continue
                if deadline.expired():
                    store.backup()
                    print("[v2] deadline reached", flush=True)
                    return
                t0 = time.time()
                rec = {"run_key": key, "model": model_name, "regime": regime, "seed": seed, "lr": lr}
                ylab = tr.label.to_numpy()
                if regime == "LODO_matched":
                    p_te = np.zeros(len(te), dtype=int)
                    rec["lodo_targets"] = {}
                    for target in MAIN:
                        rows = np.sort(tr[tr.dialect != target].sample(n=n_lodo, random_state=seed).index.to_numpy())
                        drows = np.where(dv.dialect.to_numpy() != target)[0]
                        m, f1, ep = C.train_one(ckpt, T["tr"], rows, ylab[rows], T["dv"], drows,
                                                dv.label.to_numpy()[drows], lr, seed, epochs, max_steps=max_steps)
                        tm = np.where(d_te == target)[0]
                        p_te[tm] = C.predict(m, T["te"], tm)
                        rec["lodo_targets"][target] = {"dev_f1": f1, "best_epoch": ep, "train_n": int(len(rows))}
                        C.free(m)
                else:
                    if regime == "PID":
                        rows, drows = np.arange(len(tr)), np.arange(len(dv))
                    elif regime == "MSA_only":
                        rows, drows = np.where(tr.dialect == "MSA")[0], np.where(dv.dialect == "MSA")[0]
                    else:
                        rows = stratified(tr, n_lodo if regime == "PID_nLODO" else n_msa, seed)
                        drows = np.arange(len(dv))
                    m, f1, ep = C.train_one(ckpt, T["tr"], rows, ylab[rows], T["dv"], drows, dv.label.to_numpy()[drows],
                                            lr, seed, epochs, max_steps=max_steps)
                    p_te = C.predict(m, T["te"], np.arange(len(te)))
                    rec.update(dev_f1=f1, best_epoch=ep, train_n=int(len(rows)))
                    C.free(m)
                fname = f"V2{regime}_{model_name}_seed{seed}.npy"
                store.save_pred(fname, p_te)
                rec["pred_files"] = [fname]
                rec.update(step2.compute_robustness_profile(y_te, p_te, d_te))
                rec["seconds"] = round(time.time() - t0, 1)
                store.append(rec)
                print(f"  {key}: pooled={rec['pooled_macro_f1']:.4f} HM={rec['hm_f1']:.4f} [{rec['seconds']:.0f}s]",
                      flush=True)
    store.backup()
    print("[STEP 5e PASSED]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out_dir", default=str(ROOT / "outputs_v2"))
    ap.add_argument("--runs_name", default="v2_runs.jsonl")
    ap.add_argument("--deadline_hours", type=float, default=None)
    ap.add_argument("--backup", default=None)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    prepare() if a.prepare else train(a)


if __name__ == "__main__":
    main()
