# -*- coding: utf-8 -*-
"""Step 5b: 3-class (negative / neutral / positive) replication of the robustness-metric findings.

Same leak-free protocol as Step 1 on the full ArSarcasm corpus: deduplicate the official test split first,
drop train tweets leaking into test, deduplicate train, hold Maghrebi out, carve a stratified 15% dev split.
Runs Majority and balanced TF-IDF+LR, then 5-seed full fine-tuning (PID) of the core encoders, and writes
per-dialect 3-class macro-F1, WD-F1, HM-F1 and Old-DRI to outputs_v2/three_class_replication.json.

  python 05b_three_class_replication.py --prepare            # splits + classical baselines (CPU)
  python 05b_three_class_replication.py --models MARBERT ...  # GPU worker(s), resumable
  python 05b_three_class_replication.py --aggregate           # summary JSON from all workers
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
from scipy.stats import kendalltau
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
step1 = SourceFileLoader("step1", str(ROOT / "01_prepare_splits.py")).load_module()
step3 = SourceFileLoader("step3", str(ROOT / "03_model_and_fertility_audit.py")).load_module()
C = SourceFileLoader("fullft_common", str(ROOT / "_fullft_common.py")).load_module()

MAIN = ["MSA", "Egyptian", "Levantine", "Gulf"]
CORE = ["AraBERTv02", "MARBERT", "CAMeLBERT-Mix", "XLM-R-Base"]
SEEDS = [42, 43, 44, 45, 46]
LABELS = [0, 1, 2]
DIALECT_MAP = {"egypt": "Egyptian", "gulf": "Gulf", "levant": "Levantine", "magreb": "Maghrebi", "msa": "MSA"}
DIR3 = "three_class"


def profile3(y, p, d) -> dict:
    per = {g: float(f1_score(y[d == g], p[d == g], average="macro", labels=LABELS, zero_division=0)) for g in MAIN}
    acc = np.array([accuracy_score(y[d == g], p[d == g]) for g in MAIN])
    v = np.array([per[g] for g in MAIN])
    return {"pooled_macro_f1": float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)),
            "pooled_acc": float(accuracy_score(y, p)), "dialect_macro_f1": float(v.mean()), "wd_f1": float(v.min()),
            "hm_f1": float(len(v) / np.sum(1.0 / np.maximum(v, 1e-6))), "max_gap_vs_msa": float(per["MSA"] - v.min()),
            "old_dri_acc": float(1.0 / (1.0 + np.var(acc))), "per_dialect_f1": per,
            "per_dialect_acc": dict(zip(MAIN, map(float, acc)))}


def prepare(out: Path) -> None:
    from datasets import load_dataset
    ds = load_dataset("iabufarha/ar_sarcasm")
    frames = []
    for split in ["train", "test"]:
        d, f = ds[split], ds[split].features
        frames.append(pd.DataFrame({
            "text": d["tweet"], "dialect": [DIALECT_MAP[f["dialect"].int2str(x)] for x in d["dialect"]],
            "label": [int(x) for x in d["sentiment"]],  # ClassLabel order: negative=0, neutral=1, positive=2
            "sarcasm": [bool(x) for x in d["sarcasm"]], "split": split}))
    raw = pd.concat(frames, ignore_index=True)
    assert ds["train"].features["sentiment"].names == ["negative", "neutral", "positive"]
    raw["tweet_id"] = np.arange(len(raw))
    raw["norm_text"] = raw["text"].apply(step1.normalize_for_dedup)
    raw = raw[raw["norm_text"].str.len() > 0].copy()
    leaks = len(set(raw[raw.split == "train"].norm_text) & set(raw[raw.split == "test"].norm_text))
    test = raw[raw.split == "test"].drop_duplicates("norm_text").copy()
    pool = raw[raw.split == "train"]
    pool = pool[~pool.norm_text.isin(set(test.norm_text))].drop_duplicates("norm_text").copy()
    mag = pd.concat([pool[pool.dialect == "Maghrebi"], test[test.dialect == "Maghrebi"]], ignore_index=True)
    tr4, te4 = pool[pool.dialect.isin(MAIN)].copy(), test[test.dialect.isin(MAIN)].copy()
    strata = tr4.dialect + "_" + tr4.label.astype(str)
    tr, dv = train_test_split(tr4, test_size=0.15, random_state=42, stratify=strata)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", tr), ("dev", dv), ("test", te4), ("maghrebi", mag)]:
        df.sort_values("tweet_id").reset_index(drop=True).to_csv(out / f"{name}_3cls.csv", index=False, encoding="utf-8-sig")
    card = {"raw_total": int(len(ds["train"]) + len(ds["test"])), "raw_cross_split_leak_texts": int(leaks),
            "unique_total": int(len(pool) + len(test)), "train": int(len(tr)), "dev": int(len(dv)), "test": int(len(te4)),
            "maghrebi": int(len(mag)),
            "test_by_dialect_label": {g: te4[te4.dialect == g].label.value_counts().sort_index().to_dict() for g in MAIN},
            "train_label_counts": tr.label.value_counts().sort_index().to_dict()}
    (out / "data_card_3cls.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))

    # classical baselines
    tr, dv, te = (pd.read_csv(out / f"{n}_3cls.csv") for n in ["train", "dev", "test"])
    res, preds = {}, {}
    maj = DummyClassifier(strategy="most_frequent").fit(tr.text, tr.label)
    preds["Majority"] = maj.predict(te.text)
    best = (-1, None, None)
    for c in [0.25, 0.5, 1.0, 2.0, 4.0]:
        pipe = make_pipeline(make_union(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                             LogisticRegression(max_iter=3000, C=c, class_weight="balanced", random_state=42))
        pipe.fit(tr.text, tr.label)
        f1 = f1_score(dv.label, pipe.predict(dv.text), average="macro")
        if f1 > best[0]:
            best = (f1, c, pipe)
    preds["TFIDF_LR_balanced"] = best[2].predict(te.text)
    for k, p in preds.items():
        res[k] = profile3(te.label.to_numpy(), np.asarray(p), te.dialect.to_numpy())
    res["TFIDF_LR_balanced"]["best_C"] = best[1]
    (out / "classical_3cls.json").write_text(json.dumps(res, indent=2))
    np.savez(out / "classical_3cls_preds.npz", **{k: np.asarray(v) for k, v in preds.items()})
    print({k: round(v["hm_f1"], 4) for k, v in res.items()})


def run_models(out_root: Path, models, runs_name, deadline_h, smoke) -> None:
    out = out_root / DIR3
    tr, dv, te, mg = (pd.read_csv(out / f"{n}_3cls.csv") for n in ["train", "dev", "test", "maghrebi"])
    seeds, epochs, max_steps = SEEDS, 4, None
    if smoke:
        tr = pd.concat([g.sample(9, random_state=0) for _, g in tr.groupby(["dialect"])]).reset_index(drop=True)
        dv = pd.concat([g.sample(6, random_state=0) for _, g in dv.groupby(["dialect"])]).reset_index(drop=True)
        seeds, epochs, max_steps = [42], 1, 2
    reads = sorted({p.name for p in out.glob("threeclass_runs*.jsonl")} | {runs_name})
    store = C.RunStore(out, runs_name, reads)
    deadline = C.Deadline(deadline_h * 3600 if deadline_h else None)
    lr_dir = out_root / "selected_lrs"
    y_te, d_te = te.label.to_numpy(), te.dialect.to_numpy()
    for model_name in models:
        ckpt = step3.MODEL_REGISTRY[model_name]
        f = lr_dir / f"{model_name}.json"
        lr = float(json.loads(f.read_text())["selected_lr"]) if f.exists() else 2e-5  # binary dev-selected LR
        tok = AutoTokenizer.from_pretrained(ckpt)
        T = {k: C.Tokenized(tok, df["text"]) for k, df in [("tr", tr), ("dv", dv), ("te", te), ("mg", mg)]}
        for seed in seeds:
            key = f"{model_name}|PID3|{seed}"
            if store.is_done(key):
                print(f"  [SKIP - ALREADY COMPLETED] {key}", flush=True)
                continue
            if deadline.expired():
                print("[3cls] deadline reached", flush=True)
                return
            t0 = time.time()
            m, f1, ep = C.train_one(ckpt, T["tr"], np.arange(len(tr)), tr.label, T["dv"], np.arange(len(dv)), dv.label,
                                    lr, seed, epochs, num_labels=3, max_steps=max_steps)
            p_te, p_mg = C.predict(m, T["te"], np.arange(len(te))), C.predict(m, T["mg"], np.arange(len(mg)))
            C.free(m)
            files = [f"3cls_{model_name}_seed{seed}.npy", f"3clsmag_{model_name}_seed{seed}.npy"]
            store.save_pred(files[0], p_te)
            store.save_pred(files[1], p_mg)
            rec = {"run_key": key, "model": model_name, "regime": "PID3", "seed": seed, "lr": lr, "finetune": "full",
                   "epochs_max": epochs, "dev_f1": f1, "best_epoch": ep, "pred_files": files,
                   "maghrebi_acc": float(np.mean(p_mg == mg.label.to_numpy())), **profile3(y_te, p_te, d_te),
                   "seconds": round(time.time() - t0, 1)}
            store.append(rec)
            print(f"  {key}: pooled F1={rec['pooled_macro_f1']:.4f} HM={rec['hm_f1']:.4f} WD={rec['wd_f1']:.4f} "
                  f"[{rec['seconds']:.0f}s]", flush=True)


def aggregate(out_root: Path) -> None:
    out = out_root / DIR3
    te = pd.read_csv(out / "test_3cls.csv")
    runs = []
    for p in sorted(out.glob("threeclass_runs*.jsonl")):
        runs += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    runs = list({r["run_key"]: r for r in runs}.values())
    classical = json.loads((out / "classical_3cls.json").read_text())
    summary = {"data_card": json.loads((out / "data_card_3cls.json").read_text()), "classical": classical, "models": {}}
    keys = ["pooled_macro_f1", "dialect_macro_f1", "hm_f1", "wd_f1", "max_gap_vs_msa", "old_dri_acc"]
    for m in CORE:
        g = [r for r in runs if r["model"] == m]
        if not g:
            continue
        summary["models"][m] = {"n_seeds": len(g), "lr": g[0]["lr"],
                                **{f"{k}_mean": float(np.mean([r[k] for r in g])) for k in keys},
                                **{f"{k}_std": float(np.std([r[k] for r in g], ddof=1)) if len(g) > 1 else 0.0 for k in keys},
                                **{f"f1_{d}_mean": float(np.mean([r["per_dialect_f1"][d] for r in g])) for d in MAIN},
                                **{f"f1_{d}_std": float(np.std([r["per_dialect_f1"][d] for r in g], ddof=1)) if len(g) > 1 else 0.0 for d in MAIN}}
    systems = [(k, v) for k, v in classical.items()] + [(r["run_key"], r) for r in runs]
    if len(systems) >= 3:
        hm = [s["hm_f1"] for _, s in systems]
        summary["kendall_tau_hm_vs_old_dri"] = float(kendalltau(hm, [s["old_dri_acc"] for _, s in systems])[0])
        summary["kendall_tau_hm_vs_wd"] = float(kendalltau(hm, [s["wd_f1"] for _, s in systems])[0])
        summary["n_systems"] = len(systems)
        dri_rank = sorted(systems, key=lambda kv: -kv[1]["old_dri_acc"])
        summary["majority_old_dri_rank"] = 1 + [k for k, _ in dri_rank].index("Majority")
        summary["majority_hm_rank"] = 1 + [k for k, _ in sorted(systems, key=lambda kv: -kv[1]["hm_f1"])].index("Majority")
    (out_root / "three_class_replication.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({m: {k: round(v[k], 4) for k in ["pooled_macro_f1_mean", "hm_f1_mean", "wd_f1_mean", "old_dri_acc_mean"]}
                      for m, v in summary["models"].items()}, indent=2))
    print("[STEP 5b] three_class_replication.json written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "outputs_v2"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--runs_name", default="threeclass_runs.jsonl")
    ap.add_argument("--deadline_hours", type=float, default=None)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir) if not a.smoke_test else Path(os.environ.get("SMOKE_DIR", str(ROOT / "smoke_out")))
    if a.prepare or a.smoke_test:
        prepare(out / DIR3)
    if a.models is not None or a.smoke_test:
        run_models(out, a.models if a.models else (["AraBERTv02"] if a.smoke_test else CORE), a.runs_name,
                   a.deadline_hours, a.smoke_test)
    if a.aggregate or a.smoke_test:
        aggregate(out)


if __name__ == "__main__":
    main()
