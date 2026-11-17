# -*- coding: utf-8 -*-
"""Step 7b (CPU): label-quality and near-duplicate audits requested in review.

1. Near duplicates: Jaccard similarity of character 4-gram shingle sets (normalized text, Step-1 normalization)
   between every test_4 tweet and every train_4/dev_4 tweet (exact sparse computation, no LSH approximation).
   Reports counts at J >= 0.5 / 0.7 / 0.9 and saves the pairs >= 0.7 for inspection.
   Sensitivity: every PID run is re-scored on test_4 minus near-duplicated tweets (no retraining).
2. Dialect labels vs automatic dialect identification: CAMeLBERT-Mix DID fine-tuned on MADAR-6
   (Beirut, Cairo, Doha, Rabat, Tunis, MSA -> LEV, EGY, GLF, MAG, MAG, MSA) and on NADI (country-level;
   mapped to regions, MSA not a NADI class). Reports agreement and Cohen's kappa per split and per gold dialect.
   Sensitivity: PID runs re-scored on the DID-agreeing test subset.
Outputs: outputs_v2/data_audit.json, near_duplicates.csv, dialect_id_predictions.csv
"""
from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2"
S1 = SourceFileLoader("step1", str(ROOT / "01_prepare_splits.py")).load_module()
step2 = SourceFileLoader("step2", str(ROOT / "02_classical_baselines.py")).load_module()
MAIN = ["MSA", "Egyptian", "Levantine", "Gulf"]
MADAR6 = "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus6"
NADI = "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi"
MADAR_MAP = {"bei": "Levantine", "cai": "Egyptian", "doh": "Gulf", "rab": "Maghrebi", "tun": "Maghrebi",
             "msa": "MSA"}  # MADAR-6 checkpoint emits 3-letter city codes; full names mapped by prefix
NADI_REGION = {"egypt": "Egyptian", "saudi_arabia": "Gulf", "kuwait": "Gulf", "qatar": "Gulf", "bahrain": "Gulf",
               "oman": "Gulf", "united_arab_emirates": "Gulf", "uae": "Gulf", "yemen": "Gulf", "iraq": "Iraqi",
               "syria": "Levantine", "lebanon": "Levantine", "jordan": "Levantine", "palestine": "Levantine",
               "morocco": "Maghrebi", "algeria": "Maghrebi", "tunisia": "Maghrebi", "libya": "Maghrebi",
               "mauritania": "Maghrebi", "sudan": "Other", "somalia": "Other", "djibouti": "Other"}


def shingles(texts, n=4):
    vocab, rows, cols = {}, [], []
    for i, t in enumerate(texts):
        t = f" {t} "
        grams = {t[j : j + n] for j in range(max(1, len(t) - n + 1))}
        for g in grams:
            rows.append(i)
            cols.append(vocab.setdefault(g, len(vocab)))
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), len(vocab)))


def near_duplicates(tr, dv, te):
    pool = pd.concat([tr.assign(split="train"), dv.assign(split="dev")], ignore_index=True)
    texts = list(te.norm_text.astype(str)) + list(pool.norm_text.astype(str))
    X = shingles(texts)
    A, B = X[: len(te)], X[len(te):]
    inter = (A @ B.T).toarray()
    sa, sb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
    jac = inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1)
    best = jac.max(1)
    arg = jac.argmax(1)
    pairs = []
    for i in np.where(best >= 0.7)[0]:
        j = arg[i]
        pairs.append({"test_tweet_id": int(te.tweet_id[i]), "pool_tweet_id": int(pool.tweet_id[j]),
                      "pool_split": pool.split[j], "jaccard": round(float(best[i]), 3),
                      "same_label": bool(te.label[i] == pool.label[j]), "same_dialect": bool(te.dialect[i] == pool.dialect[j]),
                      "test_text": te.text[i], "pool_text": pool.text[j]})
    pd.DataFrame(pairs).to_csv(OUT / "near_duplicates.csv", index=False, encoding="utf-8-sig")
    summ = {f"test_with_neighbour_ge_{t}": int((best >= t).sum()) for t in (0.5, 0.7, 0.8, 0.9)}
    summ["by_dialect_ge_0.7"] = {d: int(((best >= 0.7) & (te.dialect == d)).sum()) for d in MAIN}
    summ["same_label_rate_ge_0.7"] = float(np.mean([p["same_label"] for p in pairs])) if pairs else None
    return summ, best


@torch.no_grad()
def did_predict(ckpt, texts, bs=64):
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
    id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
    enc = tok(texts, truncation=True, max_length=128)
    order = np.argsort([len(x) for x in enc["input_ids"]])
    out = np.empty(len(texts), dtype=object)
    for i in range(0, len(order), bs):
        b = order[i : i + bs]
        batch = tok.pad({"input_ids": [enc["input_ids"][j] for j in b],
                         "attention_mask": [enc["attention_mask"][j] for j in b]}, return_tensors="pt")
        lab = model(**batch).logits.argmax(-1).numpy()
        out[b] = [id2[int(x)] for x in lab]
    return out


def rescore_pid(te, mask):
    """Seed-mean PID metrics on a subset of test_4 (no retraining)."""
    y, d = te.label.to_numpy()[mask], te.dialect.to_numpy()[mask]
    res = {}
    for f in sorted((OUT / "raw_preds").glob("PID_*_seed*.npy")):
        model = f.stem[len("PID_"):].rsplit("_seed", 1)[0]
        p = np.load(f)[mask]
        prof = step2.compute_robustness_profile(y, p, d)
        res.setdefault(model, []).append([prof["pooled_macro_f1"], prof["hm_f1"], prof["wd_f1"]])
    return {m: dict(zip(["pooled_macro_f1", "hm_f1", "wd_f1"], np.mean(v, 0).round(4).tolist())) for m, v in res.items()}


def main() -> None:
    tr, dv, te, mg = (pd.read_csv(OUT / f) for f in ["train_4.csv", "dev_4.csv", "test_4.csv", "test_maghrebi.csv"])
    report = {}
    nd, best = near_duplicates(tr, dv, te)
    report["near_duplicates"] = nd
    keep = best < 0.7
    report["near_duplicates"]["pid_rescored_without_ge_0.7"] = rescore_pid(te, keep)
    report["pid_full_test"] = rescore_pid(te, np.ones(len(te), dtype=bool))
    print(json.dumps(nd, indent=1)[:600], flush=True)

    alldf = pd.concat([tr.assign(split="train"), dv.assign(split="dev"), te.assign(split="test"),
                       mg.assign(split="maghrebi")], ignore_index=True)
    texts = alldf.text.astype(str).tolist()
    cache = OUT / "dialect_id_predictions.csv"
    if cache.exists() and len(pd.read_csv(cache)) == len(alldf):
        c = pd.read_csv(cache)
        m6 = [str(x) for x in c.did_madar6_raw] if "did_madar6_raw" in c else [str(x) for x in c.did_madar6]
        nadi = [str(x) for x in c.did_nadi_country]
    else:
        m6 = did_predict(MADAR6, texts)
        nadi = did_predict(NADI, texts)
    alldf["did_madar6_raw"] = m6
    alldf["did_madar6"] = [MADAR_MAP.get(str(x).lower()[:3], x) for x in m6]
    alldf["did_nadi_country"] = nadi
    alldf["did_nadi_region"] = [NADI_REGION.get(x, "Other") for x in nadi]
    alldf[["tweet_id", "split", "dialect", "did_madar6_raw", "did_madar6", "did_nadi_country", "did_nadi_region"]].to_csv(
        OUT / "dialect_id_predictions.csv", index=False)
    did = {}
    for col in ["did_madar6", "did_nadi_region"]:
        sub = alldf if col == "did_madar6" else alldf[alldf.dialect != "MSA"]  # NADI has no MSA class
        did[col] = {"agreement": float((sub.dialect == sub[col]).mean()),
                    "cohen_kappa": float(cohen_kappa_score(sub.dialect, sub[col])),
                    "per_gold_dialect_agreement": {d: float((g.dialect == g[col]).mean())
                                                   for d, g in sub.groupby("dialect")},
                    "confusion": pd.crosstab(sub.dialect, sub[col]).to_dict()}
    report["dialect_id"] = did
    t = alldf[alldf.split == "test"].reset_index(drop=True)
    agree = (t.did_madar6 == t.dialect).to_numpy()
    report["dialect_id"]["test_agreeing_counts"] = {d: int(((t.dialect == d) & agree).sum()) for d in MAIN}
    report["dialect_id"]["pid_rescored_on_madar6_agreeing_test"] = rescore_pid(te, agree)
    (OUT / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("agreement", "cohen_kappa", "per_gold_dialect_agreement")}
                      for k, v in did.items()}, indent=1))
    print("[STEP 7b PASSED]")


if __name__ == "__main__":
    main()
